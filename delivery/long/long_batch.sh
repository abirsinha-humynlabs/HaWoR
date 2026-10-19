#!/usr/bin/env bash
# Batch the v3 keypoint pipeline over LONG clips read from the prod bucket (read-only) and publish to
#   s3://stage-humyn-egocentric-stereo-data/labelling_results/hand_pose_HaWoR/v3/<dst_rel>/
# where <dst_rel> mirrors the source path below chunking/ (e.g. akai/2026-07-22/<session>/004/seg_000).
#
#   JOBS=<tsv> long_batch.sh                 run + upload everything not yet uploaded (resumable)
#   JOBS=<tsv> long_batch.sh --upload-only   only retry uploads (after: aws sso login --profile stage)
#
# JOBS columns: clip <TAB> s3 source prefix (folder with left_rectified.mp4) <TAB> dst_rel. Build it with
# make_jobs.sh. Reads use AWS profile prod, writes use stage and are refused outside the v3/ prefix.
# The next clip is pre-fetched while the current one runs. If the prod or stage SSO token expires the
# batch waits (fetch) or queues the upload (retried before every clip) instead of failing; log in again.
# State (root disk): ~/hawor_runs/long_state/<clip>.{done,uploaded,failed}; logs: ~/hawor_runs/logs/long/
set -o pipefail
JOBS=${JOBS:?JOBS=<tsv> required}
RUN=~/projects/HaWoR/delivery/long/run_v3_long.sh
ST=~/hawor_runs/long_state; LOG=~/hawor_runs/logs/long; mkdir -p "$ST" "$LOG"
B=stage-humyn-egocentric-stereo-data
DST_PFX=labelling_results/hand_pose_HaWoR/v3        # the ONLY place this script writes
ts(){ date -u +%Y-%m-%dT%H:%M:%SZ; }
job(){ grep -v '^#' "$JOBS" | awk -F'\t' -v c="$1" '$1==c'; }

fetch(){                                             # $1 clip $2 src prefix; prod, read-only
  local d=~/hawor_runs/clips/$1 f
  mkdir -p "$d"
  for f in calibration.json meta.json left_rectified.mp4; do
    [ -s "$d/$f" ] && continue
    AWS_PROFILE=prod aws s3 cp "$2/$f" "$d/$f.part" --only-show-errors && mv "$d/$f.part" "$d/$f" || return 1
  done
}
fetch_wait(){                                        # retry every 10 min (e.g. prod SSO expired)
  until fetch "$1" "$2"; do
    echo "  [$(ts)] fetch failed for $1 - waiting 10 min (aws sso login --profile prod --use-device-code?)"
    sleep 600
  done
}

upload_clip(){                                       # $1 clip $2 dst_rel
  local C="$1" OUT=~/hawor_runs/out/$1_v3 D="s3://$B/$DST_PFX/$2/" f rc=0 want have
  case "$D" in "s3://$B/$DST_PFX/"?*/) ;; *) echo "  refusing dest $D"; return 1;; esac
  export AWS_PROFILE=stage
  for f in "${C}_v3_hand21_keypoints.npz" "${C}_v3_3d_keypoints.npz" "${C}_v3_v3_report.json" \
           "${C}_v3_handedness_report.json" "${C}_v3_hawor_metadata.json"; do
    [ -f "$OUT/$f" ] && { aws s3 cp "$OUT/$f" "$D" --only-show-errors || rc=1; }
  done
  aws s3 cp "$OUT/qc/" "${D}qc/" --recursive --only-show-errors || rc=1
  [ $rc -eq 0 ] || return 1
  want=$( cd "$OUT" && find . -maxdepth 1 -type f \( -name "${C}_v3_*.npz" -o -name "${C}_v3_*.json" \) -printf '%P %s\n'
          cd "$OUT" && find qc -type f -printf '%p %s\n' )
  want=$(echo "$want" | sort)
  have=$(aws s3api list-objects-v2 --bucket $B --prefix "$DST_PFX/$2/" --query 'Contents[].[Key,Size]' --output text \
         | awk -F'\t' -v p="$DST_PFX/$2/" '{k=$1; sub("^"p,"",k); print k, $2}' | sort)
  [ "$want" = "$(comm -12 <(echo "$want") <(echo "$have"))" ] || { echo "  verify mismatch for $C"; return 1; }
  echo "  uploaded $(echo "$want" | wc -l) files -> $D"
}
retry_uploads(){
  local d C
  for d in "$ST"/*.done; do
    [ -e "$d" ] || continue; C=$(basename "$d" .done); [ -f "$ST/$C.uploaded" ] && continue
    echo "[$(ts)] upload $C"
    upload_clip "$C" "$(job "$C" | cut -f3)" && { touch "$ST/$C.uploaded"; cleanup "$C"; } || echo "  UPLOAD-PENDING $C"
  done
}
cleanup(){ rm -rf ~/hawor_runs/work/"$1"_v3 ~/hawor_runs/scratch/"$1"_v3 ~/hawor_runs/clips/"$1"; }

if [ "${1:-}" = "--upload-only" ]; then retry_uploads; exit 0; fi

mountpoint -q /opt/dlami/nvme || { echo "NVMe scratch not mounted"; exit 1; }
mapfile -t TODO < <(grep -v '^#' "$JOBS" | cut -f1 | while read -r c; do [ -n "$c" ] && [ ! -f "$ST/$c.uploaded" ] && [ ! -f "$ST/$c.failed" ] && echo "$c"; done)
N=${#TODO[@]}
echo "[$(ts)] long v3 batch start: $N clips to do ($(grep -vc '^#' "$JOBS") in $JOBS)"
for i in "${!TODO[@]}"; do
  C=${TODO[$i]}; IFS=$'\t' read -r _ SRC DST < <(job "$C")
  retry_uploads
  if [ ! -f "$ST/$C.done" ]; then
    fetch_wait "$C" "$SRC"
    NEXT=${TODO[$((i+1))]:-}
    if [ -n "$NEXT" ]; then IFS=$'\t' read -r _ NSRC _ < <(job "$NEXT"); fetch "$NEXT" "$NSRC" > /dev/null 2>&1 & fi
    echo "[$(ts)] [$((i+1))/$N] run  $C"
    T0=$(date +%s)
    bash "$RUN" "$C" ~/hawor_runs/clips/"$C" > "$LOG/$C.log" 2>&1; rc=$?
    T1=$(date +%s); printf '%s\t%s\t%d\t%d\n' "$C" "$(ts)" $((T1-T0)) $rc >> "$LOG/timing.tsv"
    wait
    if [ $rc -eq 0 ] && grep -q 'V3-LONG-DONE' "$LOG/$C.log" && [ -s ~/hawor_runs/out/${C}_v3/${C}_v3_hand21_keypoints.npz ]; then
      touch "$ST/$C.done"; rm -f "$ST/$C.failed"
      echo "  done in $(( (T1-T0)/60 )) min | $(grep -a 'GATE' "$LOG/$C.log" | sed 's/^ *//')"
    else
      touch "$ST/$C.failed"; echo "  RUN-FAIL $C rc=$rc :: $(grep -avE '^\s*$' "$LOG/$C.log" | tail -3 | tr '\n' ' ' | cut -c1-400)"
      continue
    fi
  fi
  if upload_clip "$C" "$DST"; then touch "$ST/$C.uploaded"; cleanup "$C"
  else echo "  UPLOAD-PENDING $C (retried before the next clip, or: long_batch.sh --upload-only)"; fi
done
retry_uploads
echo "[$(ts)] long v3 batch end: uploaded=$(ls "$ST"/*.uploaded 2>/dev/null | wc -l) failed=$(ls "$ST"/*.failed 2>/dev/null | wc -l) pending=$(comm -23 <(ls "$ST" | grep '\.done$' | sed 's/\.done$//' | sort) <(ls "$ST" | grep '\.uploaded$' | sed 's/\.uploaded$//' | sort) | wc -l)"
