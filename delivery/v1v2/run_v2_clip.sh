#!/usr/bin/env bash
# <clip>_v2: identical to the v1 run except HAWOR_TRACK_FIX=1 (track-id handedness in detect_track).
# Reuses the v1 inputs. v1 artefacts are never touched. Upload is a separate step so it can be
# retried after an SSO re-login without redoing the compute.
set -eo pipefail
BASE="${1:?usage: run_v2_clip.sh <clip> [upload]}"
DO_UPLOAD="${2:-no}"
CLIP=${BASE}_v2
SRC=~/hawor_runs/clips/$BASE
WORK=~/hawor_runs/work/$CLIP
OUT=~/hawor_runs/out/$CLIP
REND=~/hawor_runs/render/$CLIP
B=stage-humyn-egocentric-stereo-data; PFX=labelling_results/hand_pose_HaWoR
mkdir -p "$WORK" "$OUT/qc" "$REND"
source ~/miniconda3/etc/profile.d/conda.sh
say(){ echo "=========== [$CLIP] $* ==========="; }
VIDEO="$WORK/left_eye.mp4"     # pre-linked by the caller if the mp4 is not under clips/

say "disk before"; df -h / | tail -1

if [ ! -e "$VIDEO" ]; then
  export AWS_PROFILE=stage
  aws s3 cp "s3://$B/validation-result/ZED/${GROUP:?set GROUP=home|others}/$BASE/left_eye.mp4" "$SRC/left_eye.mp4" --only-show-errors
  ln -sf "$SRC/left_eye.mp4" "$VIDEO"
fi

say "inference WITH FIX (HAWOR_TRACK_FIX=1)"
conda activate hawor
export CUDA_HOME=$CONDA_PREFIX PATH=$CONDA_PREFIX/bin:$PATH HAWOR_TRACK_FIX=1
python ~/hawor_runs/scripts/run_hawor_clip.py --video "$VIDEO" --calib "$SRC/calibration.json" \
    --clip "$CLIP" --out "$OUT" --check-roundtrip 2>&1 | grep -avE "it/s\]|%\|" | tail -30

say "reclaim masks"; rm -f "$WORK/left_eye/tracks_0_1823/model_masks.npy"; df -h / | tail -1

say "QC"
python ~/hawor_runs/scripts/qc_hawor.py --npz "$OUT/${CLIP}_hand21_keypoints.npz" --video "$VIDEO" \
    --clip "$CLIP" --out "$OUT/qc" | tail -40
# frames that were fabricated in v1, for a direct before/after
python ~/hawor_runs/scripts/qc_hawor.py --npz "$OUT/${CLIP}_hand21_keypoints.npz" --video "$VIDEO" \
    --clip "${CLIP}_WASINFILLED" --out "$OUT/qc" --frames "${CHECK_FRAMES:-713,908,1059}" >/dev/null || true

say "render + encode + score"
conda activate egoforce
python ~/projects/EgoForce/viz_delivery/render_v4_panels.py --video "$VIDEO" \
    --npz "$OUT/${CLIP}_hand21_keypoints.npz" --head "$SRC/head_pose_6dof.npz" --out "$REND" \
    --future-sec 3 --past-sec 0 --imu "$SRC/imu_accel.csv" 2>&1 | grep -avE "it/s\]|%\|" | tail -8
ffmpeg -y -loglevel error -i "$REND/left_eye_wrist_traj_panels.mp4" \
  -vf "scale=1920:-2,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black" \
  -c:v libx264 -profile:v high -crf 18 -preset veryfast -movflags +faststart \
  "$OUT/${CLIP}_hawor_wrist_traj_panels.mp4"
python ~/projects/EgoForce/fusion/evaluate_run.py --run "hawor_v2=$OUT" --out "$OUT/eval" 2>&1 | tail -2
EGO=$HOME/egoforce_runs/$BASE/type1_egoforce_only/out
RTM=$HOME/egoforce_runs/$BASE/type2_egoforce_rtmpose/out/${BASE}_2d_keypoints.npz
A=(--run "hawor_v1_stock=$HOME/hawor_runs/out/$BASE" --run "hawor_v2_trackfix=$OUT")
[ -d "$EGO" ] && A+=(--run "egoforce_type1_raw=$EGO"); [ -f "$RTM" ] && A+=(--rtmpose "$RTM")
python ~/projects/EgoForce/fusion/evaluate_run.py "${A[@]}" --out "$OUT/eval_v1_vs_v2" 2>&1 | tail -2
rm -rf "$REND"
say "COMPUTE-DONE"

if [ "$DO_UPLOAD" = "upload" ]; then
  say "upload"; export AWS_PROFILE=stage
  for f in "${CLIP}_hand21_keypoints.npz" "${CLIP}_3d_keypoints.npz" "${CLIP}_hawor_metadata.json" \
           "${CLIP}_hawor_wrist_traj_panels.mp4"; do
    aws s3 cp "$OUT/$f" "s3://$B/$PFX/$CLIP/" --only-show-errors; done
  aws s3 cp "$OUT/eval/"          "s3://$B/$PFX/$CLIP/eval/"          --recursive --only-show-errors
  aws s3 cp "$OUT/eval_v1_vs_v2/" "s3://$B/$PFX/$CLIP/eval_v1_vs_v2/" --recursive --only-show-errors
  aws s3 cp "$OUT/qc/"            "s3://$B/$PFX/$CLIP/qc/"            --recursive --only-show-errors
  say "UPLOADED"
fi
