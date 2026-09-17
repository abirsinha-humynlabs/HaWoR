#!/usr/bin/env bash
# Render + score one HaWoR clip with EgoForce's own tooling, then (optionally) upload.
# Inference runs in `hawor`; the renderer and scorer run in `egoforce` (matplotlib/OpenCV live
# there). The egoforce env is only ACTIVATED, never installed into.
# NOTE: not -u. The egoforce env's activate.d/activate-gcc_linux-64.sh references an unset
# SYS_SYSROOT, which aborts under `set -u`. That env is read-only to us, so relax the flag
# here rather than touching it.
set -eo pipefail

CLIP="${1:?usage: deliver_clip.sh <clip> [upload]}"
DO_UPLOAD="${2:-no}"

ROOT=~/hawor_runs
IN="$ROOT/clips/$CLIP"
OUT="$ROOT/out/$CLIP"
REND="$ROOT/render/$CLIP"
BUCKET=stage-humyn-egocentric-stereo-data
PREFIX="labelling_results/hand_pose_HaWoR"

mkdir -p "$REND" "$OUT"
source ~/miniconda3/etc/profile.d/conda.sh

echo "############ render (egoforce env) ############"
conda activate egoforce
# NO --clip-foot: it needs open_clip (absent) and it drops tracks, which would change what is
# being compared rather than what the model produced.
python ~/projects/EgoForce/viz_delivery/render_v4_panels.py \
    --video "$IN/left_eye.mp4" \
    --npz   "$OUT/${CLIP}_hand21_keypoints.npz" \
    --head  "$IN/head_pose_6dof.npz" \
    --out   "$REND" \
    --future-sec 3 --past-sec 0 --imu "$IN/imu_accel.csv" 2>&1 | tail -25

echo "############ web encode ############"
ffmpeg -y -loglevel error -i "$REND/left_eye_wrist_traj_panels.mp4" \
  -vf "scale=1920:-2,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black" \
  -c:v libx264 -profile:v high -crf 18 -preset veryfast -movflags +faststart \
  "$OUT/${CLIP}_hawor_wrist_traj_panels.mp4"
ls -la "$OUT/${CLIP}_hawor_wrist_traj_panels.mp4"

echo "############ score ############"
python ~/projects/EgoForce/fusion/evaluate_run.py \
    --run "hawor=$OUT" --out "$OUT/eval" 2>&1 | tail -20

if [[ "$DO_UPLOAD" == "upload" ]]; then
  echo "############ upload -> s3://$BUCKET/$PREFIX/$CLIP/ ############"
  export AWS_PROFILE=stage
  aws s3 cp "$OUT/${CLIP}_hand21_keypoints.npz"          "s3://$BUCKET/$PREFIX/$CLIP/" --only-show-errors
  aws s3 cp "$OUT/${CLIP}_3d_keypoints.npz"              "s3://$BUCKET/$PREFIX/$CLIP/" --only-show-errors
  aws s3 cp "$OUT/${CLIP}_hawor_wrist_traj_panels.mp4"   "s3://$BUCKET/$PREFIX/$CLIP/" --only-show-errors
  aws s3 cp "$OUT/${CLIP}_hawor_metadata.json"           "s3://$BUCKET/$PREFIX/$CLIP/" --only-show-errors
  aws s3 cp "$OUT/eval/"                                 "s3://$BUCKET/$PREFIX/$CLIP/eval/" --recursive --only-show-errors
  aws s3 cp "$OUT/qc/"                                   "s3://$BUCKET/$PREFIX/$CLIP/qc/" --recursive --only-show-errors
  aws s3 ls "s3://$BUCKET/$PREFIX/$CLIP/" --recursive
fi
echo "done: $CLIP"
