#!/usr/bin/env bash
# Full per-clip pipeline: fetch -> HaWoR inference -> QC -> render -> score -> upload -> reclaim.
# NOTE: not `set -u` (the egoforce env's activate.d trips on it).
set -eo pipefail

CLIP="${1:?usage: run_one_clip.sh <clip> <group>}"
GROUP="${2:?group is home|others}"
B=stage-humyn-egocentric-stereo-data
PFX=labelling_results/hand_pose_HaWoR
ROOT=~/hawor_runs
IN=$ROOT/clips/$CLIP; WORK=$ROOT/work/$CLIP; OUT=$ROOT/out/$CLIP; REND=$ROOT/render/$CLIP
export AWS_PROFILE=stage
mkdir -p "$IN" "$WORK" "$OUT" "$REND" "$OUT/qc"
source ~/miniconda3/etc/profile.d/conda.sh

say(){ echo "=========== [$CLIP] $* ==========="; }

say "disk before"; df -h / | tail -1

say "fetch"
S3IN="s3://$B/validation-result/ZED/$GROUP/$CLIP"
for f in left_eye.mp4 calibration.json imu_accel.csv; do
  [ -f "$IN/$f" ] || aws s3 cp "$S3IN/$f" "$IN/$f" --only-show-errors
done
# head pose by discovery, exactly as EgoForce does it: exactly one dir ending in _<clip>
if [ ! -f "$IN/head_pose_6dof.npz" ]; then
  mapfile -t HC < <(aws s3 ls "s3://$B/labelling_results/6dof_head_pose_v2/" | awk '{print $2}' | grep -E "_${CLIP}/$" || true)
  [ "${#HC[@]}" -eq 1 ] || { echo "FATAL: expected 1 head-pose dir ending in _$CLIP, got ${#HC[@]}: ${HC[*]:-none}" >&2; exit 1; }
  echo "  head pose: ${HC[0]%/} (discovered)"
  aws s3 cp "s3://$B/labelling_results/6dof_head_pose_v2/${HC[0]}head_pose_6dof.npz" "$IN/head_pose_6dof.npz" --only-show-errors
fi
ln -sf "$IN/left_eye.mp4" "$WORK/left_eye.mp4"

say "inference (hawor env)"
conda activate hawor
export CUDA_HOME=$CONDA_PREFIX PATH=$CONDA_PREFIX/bin:$PATH
python $ROOT/scripts/run_hawor_clip.py --video "$WORK/left_eye.mp4" --calib "$IN/calibration.json" \
    --clip "$CLIP" --out "$OUT" --check-roundtrip 2>&1 | grep -avE "it/s\]|%\|" | tail -30

say "reclaim heavy intermediates (SLAM result is cached; masks no longer needed)"
rm -f "$WORK/left_eye/tracks_0_1823/model_masks.npy"
rm -rf "$WORK/left_eye/extracted_images"
df -h / | tail -1

say "QC + drawn frames"
python $ROOT/scripts/qc_hawor.py --npz "$OUT/${CLIP}_hand21_keypoints.npz" --video "$IN/left_eye.mp4" \
    --clip "$CLIP" --out "$OUT/qc" | tail -45

say "GATE: reprojection self-consistency and bbox containment"
python - "$CLIP" <<'PY'
import json,sys,numpy as np
clip=sys.argv[1]
q=json.load(open(f'/home/ec2-user/hawor_runs/out/{clip}/qc/{clip}_qc.json'))
m=json.load(open(f'/home/ec2-user/hawor_runs/out/{clip}/{clip}_hawor_metadata.json'))
assert q['reproj']['median_px'] < 1.0, f"FAIL reproj {q['reproj']['median_px']}"
rt=m['roundtrip']['max_m']
assert rt < 1e-4, f"FAIL world->cam roundtrip {rt} m"
print(f"  reproj median {q['reproj']['median_px']:.4f} px (self-consistent by construction)")
print(f"  world->cam roundtrip max {rt:.3e} m  OK")
nd=m['roundtrip'].get('frames_in_multiple_chunks', 0)
if nd: print(f"  NOTE {nd} (hand,frame) covered by >1 chunk (HaWoR overlap; last-write-wins): "
             f"{m['roundtrip'].get('multi_chunk_keys')}")
print(f"  behind-lens {q['depth']['frac_any_joint_behind_lens']*100:.2f}%  bone CV {q['bones_observed_only']['bone_cv_median']:.5f}")
PY

say "render + encode + score (egoforce env)"
conda activate egoforce
python ~/projects/EgoForce/viz_delivery/render_v4_panels.py --video "$IN/left_eye.mp4" \
    --npz "$OUT/${CLIP}_hand21_keypoints.npz" --head "$IN/head_pose_6dof.npz" --out "$REND" \
    --future-sec 3 --past-sec 0 --imu "$IN/imu_accel.csv" 2>&1 | grep -avE "it/s\]|%\|" | tail -8
ffmpeg -y -loglevel error -i "$REND/left_eye_wrist_traj_panels.mp4" \
  -vf "scale=1920:-2,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black" \
  -c:v libx264 -profile:v high -crf 18 -preset veryfast -movflags +faststart \
  "$OUT/${CLIP}_hawor_wrist_traj_panels.mp4"
python ~/projects/EgoForce/fusion/evaluate_run.py --run "hawor=$OUT" --out "$OUT/eval" 2>&1 | tail -3
# side-by-side against EgoForce's raw run + the shared RTMPose 2D reference, when present
EGO=$HOME/egoforce_runs/$CLIP/type1_egoforce_only/out
RTM=$HOME/egoforce_runs/$CLIP/type2_egoforce_rtmpose/out/${CLIP}_2d_keypoints.npz
ARGS=(--run "hawor=$OUT"); [ -d "$EGO" ] && ARGS+=(--run "egoforce_type1_raw=$EGO")
[ -f "$RTM" ] && ARGS+=(--rtmpose "$RTM")
python ~/projects/EgoForce/fusion/evaluate_run.py "${ARGS[@]}" --out "$OUT/eval_vs_egoforce" 2>&1 | tail -3

say "upload"
aws s3 cp "$OUT/${CLIP}_hand21_keypoints.npz"        "s3://$B/$PFX/$CLIP/" --only-show-errors
aws s3 cp "$OUT/${CLIP}_3d_keypoints.npz"            "s3://$B/$PFX/$CLIP/" --only-show-errors
aws s3 cp "$OUT/${CLIP}_hawor_metadata.json"         "s3://$B/$PFX/$CLIP/" --only-show-errors
aws s3 cp "$OUT/${CLIP}_hawor_wrist_traj_panels.mp4" "s3://$B/$PFX/$CLIP/" --only-show-errors
aws s3 cp "$OUT/eval/"             "s3://$B/$PFX/$CLIP/eval/"             --recursive --only-show-errors
aws s3 cp "$OUT/eval_vs_egoforce/" "s3://$B/$PFX/$CLIP/eval_vs_egoforce/" --recursive --only-show-errors
aws s3 cp "$OUT/qc/"               "s3://$B/$PFX/$CLIP/qc/"               --recursive --only-show-errors

say "reclaim clip media"
rm -f "$IN/left_eye.mp4" "$WORK/left_eye.mp4"
rm -rf "$REND"
df -h / | tail -1
say "CLIP-DONE"
