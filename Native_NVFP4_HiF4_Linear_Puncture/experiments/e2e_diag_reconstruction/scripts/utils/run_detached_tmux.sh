#!/usr/bin/env bash
# Launch a long job in a detached tmux session (survives Cursor/SSH disconnect).
# Usage:
#   run_detached_tmux.sh <session_name> <logfile> -- <command...>
#   run_detached_tmux.sh <session_name> <logfile> bash /path/to/script.sh
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <session_name> <logfile> -- <command...>" >&2
  exit 2
fi

SESSION="$1"
LOGFILE="$2"
shift 2
if [[ "${1:-}" == "--" ]]; then
  shift
fi
if [[ $# -lt 1 ]]; then
  echo "usage: $0 <session_name> <logfile> -- <command...>" >&2
  exit 2
fi

mkdir -p "$(dirname "${LOGFILE}")"
if tmux has-session -t "=${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  echo "attach: tmux attach -t ${SESSION}" >&2
  exit 1
fi

# Keep a login-free non-interactive bash; pipefail; log everything.
CMD=$(printf '%q ' "$@")
tmux new-session -d -s "${SESSION}" "bash -lc 'set -o pipefail; ${CMD} 2>&1 | tee -a $(printf %q "${LOGFILE}"); ec=\$?; echo EXIT:\$ec | tee -a $(printf %q "${LOGFILE}"); exec bash'"
echo "detached tmux session=${SESSION}"
echo "logfile=${LOGFILE}"
echo "attach: tmux attach -t ${SESSION}"
