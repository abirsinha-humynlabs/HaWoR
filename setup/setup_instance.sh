#!/usr/bin/env bash
# One-shot, idempotent setup of the HaWoR v3 pipeline on a fresh GPU instance.
# Reproduces the environment the 2026-10-09 v3 batch ran in (see docs/HANDOFF.md).
#
# Prerequisites
#   * AWS "Deep Learning OSS Nvidia Driver AMI GPU PyTorch (Amazon Linux 2023)", GPU instance WITH an
#     NVMe instance store (g4dn.xlarge was used; g5 works too). Root volume >= 50 GB.
#   * This repo cloned on branch my-changes, with its submodules (the script inits them).
#   * SSO login for profile 'stage' (needed for the MANO step and for running clips):
#         aws sso login --profile stage --use-device-code
#
#   bash setup/setup_instance.sh           # everything; ~45-60 min on a g4dn.xlarge, mostly compiling
#   bash setup/setup_instance.sh verify    # only the checks at the end
#
# Every step skips itself when its result already exists, so re-running is safe.
# Build logs go to /opt/dlami/nvme/logs/setup_*.log.
set -eo pipefail
REPO=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
NVME=/opt/dlami/nvme; L=$NVME/logs
CONDA=~/miniconda3
B=s3://stage-humyn-egocentric-stereo-data
EGOFORCE_COMMIT=9edd455f982fd552bb9478aa9df0622f042f431d       # run/type2-fusion, as used for v3
say(){ echo "=== [$(date -u +%H:%M:%S)] $*"; }

step_scratch(){
  say "scratch: NVMe instance store at $NVME (re-created on every boot by hawor-scratch.service)"
  if ! cmp -s "$REPO/setup/hawor-scratch.sh" /usr/local/sbin/hawor-scratch.sh 2>/dev/null \
     || ! systemctl is-enabled hawor-scratch.service >/dev/null 2>&1; then
    sudo install -m 0755 "$REPO/setup/hawor-scratch.sh" /usr/local/sbin/hawor-scratch.sh
    sudo install -m 0644 "$REPO/setup/hawor-scratch.service" /etc/systemd/system/hawor-scratch.service
    sudo systemctl daemon-reload && sudo systemctl enable hawor-scratch.service
  fi
  sudo systemctl start hawor-scratch.service
  mountpoint -q $NVME || { echo "FATAL: nothing mounted at $NVME - instance has no NVMe instance store?" >&2; exit 1; }
  mkdir -p $L $NVME/dl
  command -v tmux >/dev/null || sudo dnf install -y -q tmux
}

step_layout(){
  say "layout: ~/projects/{HaWoR,EgoForce}, ~/hawor_runs"
  [ "$(git -C "$REPO" rev-parse --abbrev-ref HEAD)" = my-changes ] || echo "WARNING: $REPO is not on branch my-changes"
  mkdir -p ~/projects ~/hawor_runs/out ~/hawor_runs/logs ~/hawor_runs/scripts
  [ -e ~/projects/HaWoR ] || ln -s "$REPO" ~/projects/HaWoR
  for d in clips work scratch render; do ln -sfn $NVME/hawor_runs/$d ~/hawor_runs/$d; done
  # run_v3_clip.sh calls ~/hawor_runs/scripts/qc_hawor.py; v1v2 drivers call run_hawor_clip.py there
  for f in run_hawor_clip.py qc_hawor.py; do ln -sfn ~/projects/HaWoR/delivery/$f ~/hawor_runs/scripts/$f; done
  [ -d ~/projects/EgoForce/.git ] || git clone -q https://github.com/abirsinha-humynlabs/EgoForce.git ~/projects/EgoForce
  git -C ~/projects/EgoForce fetch -q origin
  if [ -z "$(git -C ~/projects/EgoForce status --porcelain --untracked-files=no)" ]; then
    git -C ~/projects/EgoForce checkout -q -B run/type2-fusion origin/run/type2-fusion
  fi
  [ "$(git -C ~/projects/EgoForce rev-parse HEAD)" = $EGOFORCE_COMMIT ] || \
    echo "NOTE: EgoForce is at $(git -C ~/projects/EgoForce rev-parse --short HEAD); the v3 batch used ${EGOFORCE_COMMIT:0:7}"
  git -C "$REPO" submodule update --init --recursive
}

dl(){   # url dest expected_size
  [ "$(stat -c%s "$2" 2>/dev/null)" = "$3" ] && return 0
  mkdir -p "$(dirname "$2")"; curl -sSL --fail -o "$2.part" "$1" && mv "$2.part" "$2"
  [ "$(stat -c%s "$2")" = "$3" ] || { echo "FATAL: size mismatch for $2" >&2; exit 1; }
}
step_weights(){
  say "weights (public Hugging Face; same files and sizes as the v3 batch)"
  local HF=https://huggingface.co/ThunderVVV/HaWoR/resolve/main
  dl https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt "$REPO/weights/external/detector.pt" 53582271
  dl $HF/external/droid.pth                     "$REPO/weights/external/droid.pth"                          16061701
  dl $HF/hawor/checkpoints/hawor.ckpt           "$REPO/weights/hawor/checkpoints/hawor.ckpt"                3267481572
  dl $HF/hawor/checkpoints/infiller.pt          "$REPO/weights/hawor/checkpoints/infiller.pt"               418603497
  dl $HF/hawor/model_config.yaml                "$REPO/weights/hawor/model_config.yaml"                     2743
  dl $HF/external/metric_depth_vit_large_800k.pth "$REPO/thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth" 1647972663
}

step_mano(){
  say "MANO (org copy in S3; licence: non-commercial research only - see its LICENSE.txt)"
  local M=$B/labelling_results/implementation/mano
  [ "$(stat -c%s "$REPO/_DATA/data/mano/MANO_RIGHT.pkl" 2>/dev/null)" = 3821356 ] || \
    AWS_PROFILE=stage aws s3 cp $M/MANO_RIGHT.pkl "$REPO/_DATA/data/mano/MANO_RIGHT.pkl" --only-show-errors
  [ "$(stat -c%s "$REPO/_DATA/data_left/mano_left/MANO_LEFT.pkl" 2>/dev/null)" = 3821391 ] || \
    AWS_PROFILE=stage aws s3 cp $M/MANO_LEFT.pkl "$REPO/_DATA/data_left/mano_left/MANO_LEFT.pkl" --only-show-errors
}

step_tools(){
  say "Miniforge (conda-forge only) at $CONDA, ninja"
  if [ ! -x $CONDA/bin/conda ]; then
    local V=26.7.2-0 SHA=281b0ac7d550802efc81af633225a5e6116d29ae72f3ab4eae7168c3931a4c05
    curl -sSL --fail -o $NVME/dl/Miniforge3.sh \
      https://github.com/conda-forge/miniforge/releases/download/$V/Miniforge3-$V-Linux-x86_64.sh
    echo "$SHA  $NVME/dl/Miniforge3.sh" | sha256sum -c -
    bash $NVME/dl/Miniforge3.sh -b -p $CONDA > $L/setup_miniforge.log 2>&1
  fi
  $CONDA/bin/conda config --show pkgs_dirs | grep "$NVME/cache/conda_pkgs" >/dev/null || \
    $CONDA/bin/conda config --add pkgs_dirs $NVME/cache/conda_pkgs
  mkdir -p $NVME/cache/conda_pkgs $NVME/cache/pip ~/.local/bin
  if ! command -v ninja >/dev/null; then   # build speed only (parallel nvcc); not part of any env
    curl -sSL --fail -o $NVME/dl/ninja-linux.zip https://github.com/ninja-build/ninja/releases/download/v1.13.2/ninja-linux.zip
    echo "5749cbc4e668273514150a80e387a957f933c6ed3f5f11e03fb30955e2bbead6  $NVME/dl/ninja-linux.zip" | sha256sum -c -
    python3 -c "import zipfile,sys;zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" $NVME/dl/ninja-linux.zip $NVME/dl
    install -m 0755 $NVME/dl/ninja ~/.local/bin/ninja
  fi
  export PATH=~/.local/bin:$PATH
}

split_yml(){   # yml conda_out pip_out [names to leave out of pip_out]
  python3 - "$@" <<'EOF'
import sys, re
src, conda_out, pip_out, skip = sys.argv[1], sys.argv[2], sys.argv[3], set(sys.argv[4:])
conda, pins, in_pip = [], [], False
for line in open(src):
    s = line.rstrip('\n')
    if s.strip() == '- pip:': in_pip = True; continue
    if in_pip and s.startswith('      - '): pins.append(s.strip()[2:]); continue
    in_pip = False
    if s.startswith('prefix:') or s.strip() == '- defaults': continue   # Miniforge: conda-forge only
    conda.append(s)
open(conda_out, 'w').write('\n'.join(conda) + '\n')
open(pip_out, 'w').write('\n'.join(p for p in pins if re.split('==| @ ', p)[0].lower() not in skip) + '\n')
EOF
}
env_exists(){ $CONDA/bin/conda env list | grep -E "^$1[[:space:]]" >/dev/null; }

step_env_hawor(){ (   # subshell: the build-time exports below must not leak into later steps
  say "env hawor: py3.10.21, torch 1.13.0+cu117, nvcc 11.7 (exact capture: environment/hawor.*)"
  split_yml "$REPO/environment/hawor.conda.yml" $NVME/dl/hawor_conda.yml $NVME/dl/hawor_pip.txt \
            torch torchvision torch-scatter pytorch3d chumpy lietorch droid-backends
  env_exists hawor || $CONDA/bin/mamba env create -y -f $NVME/dl/hawor_conda.yml > $L/setup_hawor_conda.log 2>&1
  source $CONDA/etc/profile.d/conda.sh; conda activate hawor
  export PIP_CACHE_DIR=$NVME/cache/pip CUDA_HOME=$CONDA_PREFIX PATH=$CONDA_PREFIX/bin:$PATH
  export TORCH_CUDA_ARCH_LIST="7.5;8.6" FORCE_CUDA=1 MAX_JOBS=${MAX_JOBS:-4}
  ok(){ (cd /tmp && python -c "$1" >/dev/null 2>&1); }
  ok "import torch; assert torch.__version__ == '1.13.0+cu117'" || \
    pip install --no-deps torch==1.13.0+cu117 torchvision==0.14.0+cu117 \
      --extra-index-url https://download.pytorch.org/whl/cu117 > $L/setup_hawor_torch.log 2>&1
  pip install -q --no-deps setuptools==80.9.0          # mmcv 1.3.9's setup.py needs pkg_resources
  pip install --no-deps --no-build-isolation -r $NVME/dl/hawor_pip.txt > $L/setup_hawor_pip.log 2>&1
  ok "import chumpy" || pip install --no-deps --no-build-isolation \
    "chumpy @ git+https://github.com/mattloper/chumpy@580566eafc9ac68b2614b64d6f7aaa84eebb70da" > $L/setup_chumpy.log 2>&1
  ok "import torch_scatter" || pip install --no-deps --no-build-isolation --no-binary torch-scatter \
    torch-scatter==2.1.2 > $L/setup_scatter.log 2>&1
  ok "import torch, droid_backends, lietorch" || \
    (cd "$REPO/thirdparty/DROID-SLAM" && rm -rf build && python setup.py install > $L/setup_droid.log 2>&1)
  if ! ok "import torch, pytorch3d._C, pytorch3d.renderer"; then   # torch first (loads libc10.so); wheel host 403s: build v0.7.5
    rm -rf $NVME/dl/pytorch3d
    git -c advice.detachedHead=false clone -q --depth 1 --branch v0.7.5 https://github.com/facebookresearch/pytorch3d.git $NVME/dl/pytorch3d
    pip install --no-deps --no-build-isolation $NVME/dl/pytorch3d > $L/setup_pytorch3d.log 2>&1
  fi
  conda deactivate
) }

step_env_mp310(){
  say "env mp310: MediaPipe 0.10.14 (exact capture: environment/mp310.conda.yml)"
  split_yml "$REPO/environment/mp310.conda.yml" $NVME/dl/mp310_conda.yml $NVME/dl/mp310_pip.txt
  env_exists mp310 || $CONDA/bin/mamba env create -y -f $NVME/dl/mp310_conda.yml > $L/setup_mp310_conda.log 2>&1
  PIP_CACHE_DIR=$NVME/cache/pip $CONDA/envs/mp310/bin/pip install -q --no-deps -r $NVME/dl/mp310_pip.txt
}

step_env_egoforce(){
  # NOT the full EgoForce env (CUDA 12.6 toolkit + torch 2.8 + TensorRT, ~12-15 GB): only what
  # run_v3_clip.sh stage 6 needs (renderer + evaluate_run.py), at EgoForce's exact pinned versions.
  say "env egoforce (render/score subset of EgoForce run/type2-fusion environment/egoforce.*)"
  env_exists egoforce || $CONDA/bin/mamba create -y -n egoforce -c conda-forge \
    "python=3.10.21=h267e890_0_cpython" "ffmpeg=6.1.1=gpl_h0db5852_117" pip > $L/setup_egoforce_conda.log 2>&1
  PIP_CACHE_DIR=$NVME/cache/pip $CONDA/envs/egoforce/bin/pip install -q --no-deps numpy==2.2.6 \
    opencv-python==4.12.0.88 matplotlib==3.10.6 scipy==1.15.3 contourpy==1.3.2 cycler==0.12.1 \
    fonttools==4.65.0 kiwisolver==1.5.1 pillow==11.3.0 pyparsing==3.3.2 python-dateutil==2.9.0.post0 \
    pyyaml==6.0.2 six==1.17.0 packaging
  echo "Subset of the EgoForce env for HaWoR run_v3_clip.sh stage 6 only; see setup/setup_instance.sh" \
    > $CONDA/envs/egoforce/SUBSET_OF_EGOFORCE.txt
}

step_verify(){
  say "verify"
  local fail=0
  for f in weights/external/detector.pt weights/external/droid.pth weights/hawor/checkpoints/hawor.ckpt \
           weights/hawor/checkpoints/infiller.pt weights/hawor/model_config.yaml \
           thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth \
           _DATA/data/mano/MANO_RIGHT.pkl _DATA/data_left/mano_left/MANO_LEFT.pkl; do
    [ -s "$REPO/$f" ] || { echo "MISSING $f"; fail=1; }
  done
  for p in ~/projects/HaWoR/delivery/run_v3_clip.sh ~/projects/EgoForce/fusion/evaluate_run.py \
           ~/hawor_runs/scripts/qc_hawor.py; do [ -e "$p" ] || { echo "MISSING $p"; fail=1; }; done
  mountpoint -q $NVME || { echo "NVMe scratch not mounted"; fail=1; }
  source $CONDA/etc/profile.d/conda.sh
  conda activate hawor
  (cd /tmp && PYTHONPATH="$REPO/thirdparty/DROID-SLAM/droid_slam:$REPO/thirdparty/DROID-SLAM" python - <<'EOF') || fail=1
import warnings; warnings.filterwarnings('ignore')
import torch, lietorch, droid_backends, torch_scatter, pytorch3d, pytorch3d.renderer, chumpy, ultralytics, mmcv
print('hawor   torch', torch.__version__, '| GPU', torch.cuda.get_device_name(0), 'sm_%d%d' % torch.cuda.get_device_capability(0))
x = 0.1 * torch.randn(64, 6, dtype=torch.float64)
d = (lietorch.SE3.exp(x.cuda()).data.cpu() - lietorch.SE3.exp(x).data).abs().max().item()
assert d < 1e-9, d
from droid_slam.modules.corr import CorrBlock
f1 = torch.randn(1, 2, 128, 16, 20, device='cuda'); f2 = torch.randn(1, 2, 128, 16, 20, device='cuda')
yy, xx = torch.meshgrid(torch.arange(16.), torch.arange(20.), indexing='ij')
c = CorrBlock(f1, f2)(torch.stack([xx, yy], -1)[None, None].expand(1, 2, 16, 20, 2).contiguous().cuda())
assert torch.isfinite(c).all()
from torch_scatter import scatter_sum
assert scatter_sum(torch.ones(4, device='cuda'), torch.tensor([0, 0, 1, 1], device='cuda')).tolist() == [2.0, 2.0]
print('hawor   lietorch GPU==CPU (%.1e), droid corr kernel, torch_scatter, pytorch3d %s: OK' % (d, pytorch3d.__version__))
EOF
  conda activate mp310
  (cd /tmp && python -c "import mediapipe as m; print('mp310   mediapipe', m.__version__, ': OK')") || fail=1
  conda activate egoforce
  (cd /tmp && python -c "import numpy, cv2, matplotlib; print('egoforce numpy', numpy.__version__, 'cv2', cv2.__version__, ': OK')") || fail=1
  [ "$($CONDA/envs/egoforce/bin/ffmpeg -hide_banner -encoders 2>/dev/null | grep -c libx264)" -gt 0 ] || { echo "egoforce ffmpeg has no libx264"; fail=1; }
  conda deactivate
  AWS_PROFILE=stage aws sts get-caller-identity >/dev/null 2>&1 || echo "NOTE: SSO not logged in (aws sso login --profile stage --use-device-code)"
  [ $fail -eq 0 ] && say "VERIFY_OK - run a clip with: delivery/batch/hawor_tmux.sh <clip> <group>" || { say "VERIFY_FAILED"; exit 1; }
}

case "${1:-all}" in
  verify) step_verify ;;
  all)    step_scratch; step_layout; step_weights; step_mano; step_tools
          step_env_hawor; step_env_mp310; step_env_egoforce; step_verify ;;
  *)      echo "usage: $0 [all|verify]"; exit 2 ;;
esac
