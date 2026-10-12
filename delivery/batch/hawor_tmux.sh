#!/usr/bin/env bash
# Launch HaWoR v3 runs inside a detached tmux session, so they survive SSH / VS Code / network drops.
#
#   hawor_tmux.sh <clip> <home|others> [upload] [--fresh]     one clip   (args as run_v3_clip.sh)
#   hawor_tmux.sh --queue <file>                              many clips, one after another;
#                                                             file lines: <clip> <group> [upload]
#
# Watch:   tmux attach -t <session>     (detach again with Ctrl-b d; the run keeps going)
# List:    tmux ls
# Logs:    ~/hawor_runs/logs/<clip>_v3_<UTC timestamp>.log   (root disk, survives a stop/start)
#
# Uploads only ever go where run_v3_clip.sh puts them:
#   s3://stage-humyn-egocentric-stereo-data/labelling_results/hand_pose_HaWoR/<clip>_v3/
set -eo pipefail
RUN=~/projects/HaWoR/delivery/run_v3_clip.sh
LOGDIR=~/hawor_runs/logs; RUNNERS=$LOGDIR/.runners; mkdir -p "$RUNNERS"

# The SSO token has to be valid at the start (input download) and, with "upload", again at the end.
if ! AWS_PROFILE=stage aws sts get-caller-identity >/dev/null 2>&1; then
  echo "AWS SSO token for profile 'stage' is not valid. Run: aws sso login --profile stage --use-device-code" >&2
  exit 1
fi
mountpoint -q /opt/dlami/nvme || { echo "/opt/dlami/nvme is not mounted (sudo systemctl start hawor-scratch)" >&2; exit 1; }

TS=$(date -u +%Y%m%dT%H%M%SZ)
if [ "${1:-}" = "--queue" ]; then
  Q=$(readlink -f "${2:?usage: hawor_tmux.sh --queue <file>}"); [ -f "$Q" ] || { echo "no such queue file" >&2; exit 1; }
  S="hawor_queue_$TS"
  cat > "$RUNNERS/$S.sh" <<EOF
#!/usr/bin/env bash
while read -r c g u; do
  [ -z "\$c" ] && continue; [ "\${c:0:1}" = "#" ] && continue
  echo "=== \$(date -u +%H:%M:%S) UTC  START \$c (\$g, upload=\${u:-no})"
  bash $RUN "\$c" "\$g" "\${u:-no}" 2>&1 | tee "$LOGDIR/\${c}_v3_$TS.log"
  echo "=== \$(date -u +%H:%M:%S) UTC  END \$c rc=\${PIPESTATUS[0]}"
done < "$Q"
echo "=== queue finished"
EOF
else
  CLIP="${1:?usage: hawor_tmux.sh <clip> <home|others> [upload] [--fresh]}"
  GROUP="${2:?usage: hawor_tmux.sh <clip> <home|others> [upload] [--fresh]}"
  S="hawor_${CLIP}"
  cat > "$RUNNERS/$S.sh" <<EOF
#!/usr/bin/env bash
bash $RUN "$CLIP" "$GROUP" "${3:-no}" ${4:-} 2>&1 | tee "$LOGDIR/${CLIP}_v3_$TS.log"
echo "=== $CLIP finished rc=\${PIPESTATUS[0]} at \$(date -u +%H:%M:%S) UTC"
EOF
fi

tmux has-session -t "$S" 2>/dev/null && { echo "tmux session '$S' already exists (tmux attach -t $S)" >&2; exit 1; }
tmux new-session -d -s "$S" "bash $RUNNERS/$S.sh"
tmux set-option -t "$S" remain-on-exit on >/dev/null   # keep the final output visible after it ends
echo "started tmux session: $S"
echo "  watch:  tmux attach -t $S    (Ctrl-b d to detach)"
echo "  logs:   $LOGDIR/"
