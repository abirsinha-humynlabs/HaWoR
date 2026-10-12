#!/usr/bin/env bash
# Pre-fetch every input of the v3 batch to the NVMe so the run itself needs S3 only for uploads.
# Lays files out exactly where delivery/run_v3_clip.sh looks first, so it skips its own downloads.
# Job list: delivery/batch/v3_jobs.tsv next to this script, or override with JOBS=<file>.
set -eo pipefail
JOBS=${JOBS:-$(dirname "$(readlink -f "$0")")/v3_jobs.tsv}
export AWS_PROFILE=stage; B=s3://stage-humyn-egocentric-stereo-data
grep -v '^#' "$JOBS" | while IFS=$'\t' read -r C D H; do
  SRC=~/hawor_runs/clips/$C; WORK=~/hawor_runs/work/${C}_v3; mkdir -p "$SRC" "$WORK"
  for f in left_eye.mp4 calibration.json imu_accel.csv; do
    [ -s "$SRC/$f" ] || aws s3 cp "$B/$D/$f" "$SRC/$f" --only-show-errors
  done
  [ -s "$SRC/head_pose_6dof.npz" ] || aws s3 cp "$B/$H" "$SRC/head_pose_6dof.npz" --only-show-errors
  ln -sfn "$SRC/left_eye.mp4" "$WORK/left_eye.mp4"
  echo "fetched $C $(du -sh "$SRC" | cut -f1)"
done
echo PREFETCH_DONE
