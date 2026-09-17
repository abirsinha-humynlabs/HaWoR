#!/usr/bin/env bash
# episode_048 v2: identical to v1 except HAWOR_TRACK_FIX=1 (track-id handedness in detect_track).
# Inputs are the SAME files v1 used. v1 artefacts are never touched.
set -eo pipefail
CLIP=episode_048_v2
SRC=~/hawor_runs/clips/episode_048
WORK=~/hawor_runs/work/$CLIP
OUT=~/hawor_runs/out/$CLIP
REND=~/hawor_runs/render/$CLIP
B=stage-humyn-egocentric-stereo-data
PFX=labelling_results/hand_pose_HaWoR
export AWS_PROFILE=stage
mkdir -p "$WORK" "$OUT/qc" "$REND"
ln -sf "$SRC/left_eye.mp4" "$WORK/left_eye.mp4"
source ~/miniconda3/etc/profile.d/conda.sh
say(){ echo "=========== [$CLIP] $* ==========="; }

say "inference WITH FIX (HAWOR_TRACK_FIX=1)"
conda activate hawor
export CUDA_HOME=$CONDA_PREFIX PATH=$CONDA_PREFIX/bin:$PATH
export HAWOR_TRACK_FIX=1
python ~/hawor_runs/scripts/run_hawor_clip.py --video "$WORK/left_eye.mp4" --calib "$SRC/calibration.json" \
    --clip "$CLIP" --out "$OUT" --check-roundtrip 2>&1 | grep -avE "it/s\]|%\|" | tail -30

say "reclaim masks"
rm -f "$WORK/left_eye/tracks_0_1823/model_masks.npy"; df -h / | tail -1

say "QC"
python ~/hawor_runs/scripts/qc_hawor.py --npz "$OUT/${CLIP}_hand21_keypoints.npz" --video "$SRC/left_eye.mp4" \
    --clip "$CLIP" --out "$OUT/qc" | tail -40
# the two frames that drifted in v1, for a direct before/after
python ~/hawor_runs/scripts/qc_hawor.py --npz "$OUT/${CLIP}_hand21_keypoints.npz" --video "$SRC/left_eye.mp4" \
    --clip "${CLIP}_DRIFT" --out "$OUT/qc" --frames "1310,1320,1575" >/dev/null

say "render + encode + score"
conda activate egoforce
python ~/projects/EgoForce/viz_delivery/render_v4_panels.py --video "$SRC/left_eye.mp4" \
    --npz "$OUT/${CLIP}_hand21_keypoints.npz" --head "$SRC/head_pose_6dof.npz" --out "$REND" \
    --future-sec 3 --past-sec 0 --imu "$SRC/imu_accel.csv" 2>&1 | grep -avE "it/s\]|%\|" | tail -8
ffmpeg -y -loglevel error -i "$REND/left_eye_wrist_traj_panels.mp4" \
  -vf "scale=1920:-2,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black" \
  -c:v libx264 -profile:v high -crf 18 -preset veryfast -movflags +faststart \
  "$OUT/${CLIP}_hawor_wrist_traj_panels.mp4"
python ~/projects/EgoForce/fusion/evaluate_run.py --run "hawor_v2=$OUT" --out "$OUT/eval" 2>&1 | tail -3
# v1 vs v2 vs EgoForce, on the shared RTMPose reference
python ~/projects/EgoForce/fusion/evaluate_run.py \
   --run "hawor_v1_stock=$HOME/hawor_runs/out/episode_048" \
   --run "hawor_v2_trackfix=$OUT" \
   --run "egoforce_type1_raw=$HOME/egoforce_runs/episode_048/type1_egoforce_only/out" \
   --rtmpose "$HOME/egoforce_runs/episode_048/type2_egoforce_rtmpose/out/episode_048_2d_keypoints.npz" \
   --out "$OUT/eval_v1_vs_v2" 2>&1 | tail -3

say "upload"
for f in "${CLIP}_hand21_keypoints.npz" "${CLIP}_3d_keypoints.npz" "${CLIP}_hawor_metadata.json" \
         "${CLIP}_hawor_wrist_traj_panels.mp4"; do
  aws s3 cp "$OUT/$f" "s3://$B/$PFX/$CLIP/" --only-show-errors
done
aws s3 cp "$OUT/eval/"          "s3://$B/$PFX/$CLIP/eval/"          --recursive --only-show-errors
aws s3 cp "$OUT/eval_v1_vs_v2/" "s3://$B/$PFX/$CLIP/eval_v1_vs_v2/" --recursive --only-show-errors
aws s3 cp "$OUT/qc/"            "s3://$B/$PFX/$CLIP/qc/"            --recursive --only-show-errors
rm -rf "$REND"
say "V2-DONE"
