#!/usr/bin/env bash
# Wait for any in-flight clip, then run the remaining clips one at a time.
# A clip that fails a QC gate exits non-zero BEFORE uploading; we record it and carry on
# to the next clip rather than abandoning the whole queue.
set -o pipefail
cd ~
while pgrep -f "run_one_clip.sh episode_" >/dev/null 2>&1; do sleep 30; done

STATUS=~/hawor_runs/out/queue_status.txt
: > "$STATUS"
for spec in "episode_009 home" "episode_002 others" "episode_088 home" "episode_053 home"; do
  set -- $spec; CLIP=$1; GRP=$2
  echo "########## QUEUE START $CLIP ##########"
  if bash ~/hawor_runs/scripts/run_one_clip.sh "$CLIP" "$GRP" > ~/hawor_runs/out/${CLIP}_pipeline.log 2>&1; then
    echo "$CLIP OK" >> "$STATUS"; echo "########## QUEUE OK $CLIP ##########"
  else
    echo "$CLIP FAILED" >> "$STATUS"; echo "########## QUEUE FAILED $CLIP (see ${CLIP}_pipeline.log) ##########"
    tail -20 ~/hawor_runs/out/${CLIP}_pipeline.log
  fi
  df -h / | tail -1
done
echo "########## QUEUE COMPLETE ##########"
cat "$STATUS"
