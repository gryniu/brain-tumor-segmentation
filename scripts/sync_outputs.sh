#!/usr/bin/env bash
# Copy run outputs (checkpoints, logs, configs, metrics, figures) to storage that outlives the instance.
# Usage:  bash scripts/sync_outputs.sh <destination> [--loop SECONDS]
#   <destination>  any rsync target: /workspace/backup, user@host:/path, or an rclone remote
#                  written as rclone:remote:path (e.g. rclone:gdrive:brats_outputs).
#   --loop N       repeat every N seconds (run it in a second terminal / tmux window during training).
# Never syncs the preprocessing cache or datasets.
set -euo pipefail
DEST="${1:?usage: sync_outputs.sh <destination> [--loop SECONDS]}"
INTERVAL=""
[[ "${2:-}" == "--loop" ]] && INTERVAL="${3:?--loop needs SECONDS}"
SRC="${OUTPUT_DIR:-outputs}"

sync_once() {
  if [[ "$DEST" == rclone:* ]]; then
    rclone copy "$SRC" "${DEST#rclone:}" --exclude "cache/**" --exclude "*.tmp" --transfers 4
  else
    rsync -a --exclude 'cache/' --exclude '*.tmp' "$SRC"/ "$DEST"/
  fi
  echo "$(date +%H:%M:%S) synced $SRC -> $DEST"
}

sync_once
while [[ -n "$INTERVAL" ]]; do sleep "$INTERVAL"; sync_once; done
