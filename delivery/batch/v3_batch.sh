#!/usr/bin/env bash
# Run every clip in v3_jobs.tsv (next to this script; override with JOBS=<file>) through the v3
# pipeline and publish it to
#   s3://stage-humyn-egocentric-stereo-data/labelling_results/hand_pose_HaWoR/v3/<clip>/
#
#   v3_batch.sh                 run + upload everything not yet uploaded (resumable)
#   v3_batch.sh --upload-only   only retry uploads of clips that ran but did not upload
#                               (e.g. the SSO token expired mid-batch: aws sso login, then this)
#
# delivery/run_v3_clip.sh is called UNCHANGED with upload=no, so it never writes to S3 itself.
# This script uploads the same file set that run_v3_clip.sh's own upload block would, only to the
# v3/ prefix, and verifies every object's size against the local file before marking it uploaded.
# Inputs are expected to be pre-fetched by v3_prefetch.sh, so S3 is needed only for the uploads.
#
# State (root disk, survives a stop/start): ~/hawor_runs/v3_state/<clip>.{done,uploaded,failed}
set -o pipefail
JOBS=${JOBS:-$(dirname "$(readlink -f "$0")")/v3_jobs.tsv}
RUN=~/projects/HaWoR/delivery/run_v3_clip.sh
ST=~/hawor_runs/v3_state; LOG=~/hawor_runs/logs/v3batch; mkdir -p "$ST" "$LOG"
B=stage-humyn-egocentric-stereo-data
DST_PFX=labelling_results/hand_pose_HaWoR/v3        # the ONLY place this script writes
export AWS_PROFILE=stage
ts(){ date -u +%Y-%m-%dT%H:%M:%SZ; }

upload_clip(){                                       # $1 = clip
  local C="$1" OUT=~/hawor_runs/out/$1_v3 D="s3://$B/$DST_PFX/$1/" f rc=0
  case "$D" in "s3://$B/labelling_results/hand_pose_HaWoR/v3/"?*/) ;; *) echo "refusing dest $D"; return 1;; esac
  for f in "${C}_v3_hand21_keypoints.npz" "${C}_v3_3d_keypoints.npz" "${C}_v3_hawor_wrist_traj_panels.mp4" \
           "${C}_v3_v3_report.json" "${C}_v3_handedness_report.json" "head_pose_6dof_smoothed.npz"; do
    [ -f "$OUT/$f" ] && { aws s3 cp "$OUT/$f" "$D" --only-show-errors || rc=1; }
  done
  for f in eval qc; do aws s3 cp "$OUT/$f/" "$D$f/" --recursive --only-show-errors || rc=1; done
  [ $rc -eq 0 ] || return 1
  # verify: every local file that should be there is there remotely with the same byte size
  local want have
  want=$( cd "$OUT" && { for f in "${C}_v3_hand21_keypoints.npz" "${C}_v3_3d_keypoints.npz" \
            "${C}_v3_hawor_wrist_traj_panels.mp4" "${C}_v3_v3_report.json" "${C}_v3_handedness_report.json" \
            "head_pose_6dof_smoothed.npz"; do [ -f "$f" ] && echo "$f $(stat -c%s "$f")"; done
          find eval qc -type f -printf '%p %s\n'; } | sort )
  have=$(aws s3 ls "$D" --recursive | awk -v p="$DST_PFX/$C/" '{k=$4; for(i=5;i<=NF;i++)k=k" "$i; sub("^"p,"",k); print k, $3}' | sort)
  [ "$want" = "$(comm -12 <(echo "$want") <(echo "$have"))" ] || { echo "  verify mismatch for $C"; return 1; }
  echo "  uploaded $(echo "$want" | wc -l) files -> $D"
}

if [ "${1:-}" = "--upload-only" ]; then
  for d in "$ST"/*.done; do
    [ -e "$d" ] || continue; C=$(basename "$d" .done); [ -f "$ST/$C.uploaded" ] && continue
    echo "[$(ts)] upload $C"; upload_clip "$C" && touch "$ST/$C.uploaded" || echo "  UPLOAD-FAIL $C"
  done; exit 0
fi

mountpoint -q /opt/dlami/nvme || { echo "NVMe scratch not mounted"; exit 1; }
N=$(grep -vc '^#' "$JOBS"); i=0
echo "[$(ts)] v3 batch start: $N clips"
while IFS=$'\t' read -r C DIR HEAD <&3; do
  [ -z "$C" ] || [ "${C:0:1}" = "#" ] && continue
  i=$((i+1))
  [ -f "$ST/$C.uploaded" ] && { echo "[$(ts)] [$i/$N] skip $C (already uploaded)"; continue; }
  if [ ! -f "$ST/$C.done" ]; then
    G=${DIR#validation-result/ZED/}; G=${G%/$C}
    echo "[$(ts)] [$i/$N] run  $C  (group '$G')"
    T0=$(date +%s)
    ( cd ~ && bash "$RUN" "$C" "$G" no ) > "$LOG/$C.log" 2>&1; rc=$?
    T1=$(date +%s); printf '%s\t%s\t%d\t%d\n' "$C" "$(ts)" $((T1-T0)) $rc >> "$LOG/timing.tsv"
    OUT=~/hawor_runs/out/${C}_v3
    if [ $rc -eq 0 ] && grep -q 'V3-DONE' "$LOG/$C.log" && [ -s "$OUT/${C}_v3_hand21_keypoints.npz" ] \
       && [ -s "$OUT/${C}_v3_hawor_wrist_traj_panels.mp4" ]; then
      touch "$ST/$C.done"; rm -f "$ST/$C.failed"
      echo "  done in $(( (T1-T0)/60 )) min | $(grep -a 'GATE' "$LOG/$C.log" | sed 's/^ *//')"
    else
      touch "$ST/$C.failed"; echo "  RUN-FAIL $C rc=$rc :: $(grep -avE '^\s*$' "$LOG/$C.log" | tail -2 | tr '\n' ' ' | cut -c1-300)"
      continue
    fi
  fi
  if upload_clip "$C"; then
    touch "$ST/$C.uploaded"
    rm -rf ~/hawor_runs/work/${C}_v3/left_eye          # frames + 3.8 GB mask memmap; inputs kept
  else
    echo "  UPLOAD-PENDING $C (run: v3_batch.sh --upload-only after aws sso login)"
  fi
done 3< "$JOBS"
echo "[$(ts)] v3 batch end: done=$(ls "$ST"/*.done 2>/dev/null | wc -l) uploaded=$(ls "$ST"/*.uploaded 2>/dev/null | wc -l) failed=$(ls "$ST"/*.failed 2>/dev/null | wc -l) of $N"
