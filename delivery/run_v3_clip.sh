#!/usr/bin/env bash
# End-to-end improved pipeline for one clip, v3.
#
#   1. HaWoR inference with HAWOR_TRACK_FIX=1   (track-id handedness in detect_track)
#   2. postprocess_v3.py   drop CMIB-infiller rows; zero-phase smooth keypoints and head pose
#   3. mp_handedness.py    MediaPipe, in its own env, as an independent handedness opinion
#   4. fix_handedness.py   per-physical-hand vote; writes npz['hand'] for the renderer
#   5. render_v4_panels_handfix.py   delivery renderer that honours npz['hand']
#   6. score + upload
#
# Usage: run_v3_clip.sh <clip> <group> [upload] [--fresh]
#   --fresh clears HaWoR's stage caches so inference genuinely re-runs.
set -eo pipefail
BASE="${1:?usage: run_v3_clip.sh <clip> <group> [upload] [--fresh]}"
GROUP="${2:?home|others}"
DO_UPLOAD="${3:-no}"
FRESH="${4:-}"

C=${BASE}_v3
SRC=~/hawor_runs/clips/$BASE
WORK=~/hawor_runs/work/$C
OUT=~/hawor_runs/out/$C
REND=~/hawor_runs/render/$C
SCRATCH=~/hawor_runs/scratch/$C
B=stage-humyn-egocentric-stereo-data; PFX=labelling_results/hand_pose_HaWoR
D=~/projects/HaWoR/delivery
mkdir -p "$WORK" "$OUT/qc" "$REND" "$SCRATCH" "$SRC"
source ~/miniconda3/etc/profile.d/conda.sh
say(){ echo "=========== [$C] $* ==========="; }

say "inputs"; df -h / | tail -1
export AWS_PROFILE=stage
for f in calibration.json imu_accel.csv; do
  [ -f "$SRC/$f" ] || aws s3 cp "s3://$B/validation-result/ZED/$GROUP/$BASE/$f" "$SRC/$f" --only-show-errors
done
if [ ! -f "$SRC/head_pose_6dof.npz" ]; then
  mapfile -t HC < <(aws s3 ls "s3://$B/labelling_results/6dof_head_pose_v2/" | awk '{print $2}' | grep -E "_${BASE}/$" || true)
  [ "${#HC[@]}" -eq 1 ] || { echo "FATAL: expected 1 head-pose dir ending in _$BASE, got ${#HC[@]}" >&2; exit 1; }
  echo "  head pose: ${HC[0]%/}"
  aws s3 cp "s3://$B/labelling_results/6dof_head_pose_v2/${HC[0]}head_pose_6dof.npz" "$SRC/head_pose_6dof.npz" --only-show-errors
fi
if [ ! -e "$WORK/left_eye.mp4" ]; then
  LOCAL=$(ls -S ~/egoforce_runs/$BASE/*/input/left_eye.mp4 2>/dev/null | head -1 || true)
  if [ -n "$LOCAL" ] && [ "$(stat -c%s "$LOCAL")" -gt 1000000 ]; then ln -sf "$LOCAL" "$WORK/left_eye.mp4"
  else aws s3 cp "s3://$B/validation-result/ZED/$GROUP/$BASE/left_eye.mp4" "$SRC/left_eye.mp4" --only-show-errors
       ln -sf "$SRC/left_eye.mp4" "$WORK/left_eye.mp4"; fi
fi

if [ "$FRESH" = "--fresh" ]; then say "clearing HaWoR stage caches (genuine re-inference)"; rm -rf "$WORK/left_eye"; fi

say "1/6 HaWoR inference (HAWOR_TRACK_FIX=1)"
conda activate hawor
export CUDA_HOME=$CONDA_PREFIX PATH=$CONDA_PREFIX/bin:$PATH HAWOR_TRACK_FIX=1
python $D/run_hawor_clip.py --video "$WORK/left_eye.mp4" --calib "$SRC/calibration.json" \
  --clip "${BASE}_raw" --out "$SCRATCH" --check-roundtrip 2>&1 | grep -avE "it/s\]|%\|" | tail -22
rm -f "$WORK/left_eye/tracks_0_1823/model_masks.npy"

say "2/6 post-process (drop fabricated, smooth keypoints + head)"
python $D/postprocess_v3.py --npz "$SCRATCH/${BASE}_raw_hand21_keypoints.npz" \
  --out "$SCRATCH/pp.npz" --head "$SRC/head_pose_6dof.npz" \
  --head-out "$OUT/head_pose_6dof_smoothed.npz" --head-win 3 2>&1 | tail -25

say "3/6 MediaPipe second opinion (isolated env)"
conda activate mp310
python $D/mp_handedness.py --video "$WORK/left_eye.mp4" --out "$SCRATCH/mp.npz" 2>&1 | tail -8

say "4/6 per-hand handedness vote"
conda activate hawor
python $D/fix_handedness.py --npz "$SCRATCH/pp.npz" \
  --out "$OUT/${C}_hand21_keypoints.npz" --mp "$SCRATCH/mp.npz" 2>&1 | tail -25
cp "$SCRATCH/pp_v3_report.json" "$OUT/${C}_v3_report.json" 2>/dev/null || true
cp "$OUT/${C}_hand21_keypoints_handedness_report.json" "$OUT/${C}_handedness_report.json" 2>/dev/null || true
cp "$SCRATCH/${BASE}_raw_3d_keypoints.npz" "$OUT/${C}_3d_keypoints.npz"

say "5/6 QC + gate"
python ~/hawor_runs/scripts/qc_hawor.py --npz "$OUT/${C}_hand21_keypoints.npz" \
  --video "$WORK/left_eye.mp4" --clip "$C" --out "$OUT/qc" | tail -30
python - "$OUT" "$C" <<'PY'
import json,sys
o,c=sys.argv[1:3]
q=json.load(open(f'{o}/qc/{c}_qc.json'))
r=q['reproj']['median_px']
assert r < 1.0, f"FAIL reproj {r} px"
print(f"  GATE reproj {r:.4f} px OK | behind-lens {q['depth']['frac_any_joint_behind_lens']*100:.2f}%"
      f" | boneCV {q['bones_observed_only']['bone_cv_median']:.5f} | rows {q['rows']['total']}")
PY

say "6/6 render (handedness honoured) + score"
conda activate egoforce
python $D/render_v4_panels_handfix.py --video "$WORK/left_eye.mp4" \
  --npz "$OUT/${C}_hand21_keypoints.npz" --head "$OUT/head_pose_6dof_smoothed.npz" --out "$REND" \
  --future-sec 3 --past-sec 0 --imu "$SRC/imu_accel.csv" 2>&1 | grep -avE "it/s\]|%\|" | tail -8
ffmpeg -y -loglevel error -i "$REND/left_eye_wrist_traj_panels.mp4" \
  -vf "scale=1920:-2,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black" \
  -c:v libx264 -profile:v high -crf 18 -preset veryfast -movflags +faststart \
  "$OUT/${C}_hawor_wrist_traj_panels.mp4"
python ~/projects/EgoForce/fusion/evaluate_run.py --run "hawor_v3=$OUT" --out "$OUT/eval" 2>&1 | tail -2

if [ "$DO_UPLOAD" = "upload" ]; then
  say "upload"; export AWS_PROFILE=stage
  for f in "${C}_hand21_keypoints.npz" "${C}_3d_keypoints.npz" "${C}_hawor_wrist_traj_panels.mp4" \
           "${C}_v3_report.json" "${C}_handedness_report.json" "head_pose_6dof_smoothed.npz"; do
    [ -f "$OUT/$f" ] && aws s3 cp "$OUT/$f" "s3://$B/$PFX/$C/" --only-show-errors; done
  for d in eval qc; do aws s3 cp "$OUT/$d/" "s3://$B/$PFX/$C/$d/" --recursive --only-show-errors; done
fi
rm -rf "$REND"
say "V3-DONE"
