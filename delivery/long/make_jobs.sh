#!/usr/bin/env bash
# Build a long_batch.sh job list from a clips CSV whose first column is the clip name, e.g.
#   akai__2026-07-22__akai-ego-005_2026-07-22_02-21-55__004__seg_000
#   -> source s3://<prod bucket>/chunking/akai/2026-07-22/akai-ego-005_2026-07-22_02-21-55/004/seg_000/
#   -> dst_rel akai/2026-07-22/akai-ego-005_2026-07-22_02-21-55/004/seg_000   (under hand_pose_HaWoR/v3/)
# Every source is checked to hold left_rectified.mp4 + calibration.json (AWS profile prod, read-only).
#   make_jobs.sh clips.csv > jobs.tsv
set -o pipefail
CSV=${1:?usage: make_jobs.sh <clips.csv>}; SRC_ROOT=${SRC_ROOT:-s3://prod-egc-stereo-v2-data/chunking}
echo -e "# clip\tsource\tdst_rel   (from $(basename "$CSV"), $(date -u +%F))"
tail -n +2 "$CSV" | cut -d, -f1 | tr -d '\r' | while read -r C; do
  [ -n "$C" ] || continue
  REL=${C//__//}
  have=$(AWS_PROFILE=prod aws s3 ls "$SRC_ROOT/$REL/" 2>/dev/null | awk '{print $4}' | grep -cxE 'left_rectified\.mp4|calibration\.json')
  if [ "$have" = 2 ]; then echo -e "$C\t$SRC_ROOT/$REL\t$REL"; else echo "MISSING inputs for $C at $SRC_ROOT/$REL/" >&2; fi
done
