#!/usr/bin/env bash
# One-shot status read; never loops or tails.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
RUNS_ROOT="${ROOT}/Native_NVFP4_HiF4_Linear_Puncture/results/internal_error_accumulation/qwen3_30b_a3b_e0_e1_formal/runs"

if [[ $# -ne 1 ]]; then
  echo "usage: show_status.sh RUN_ID" >&2
  exit 2
fi
RUN_ID="$1"
RUN_ROOT="${RUNS_ROOT}/${RUN_ID}"
STATE="${RUN_ROOT}/run_state.json"
PID_FILE="${RUN_ROOT}/launcher.pid"

echo "RUN_ID=${RUN_ID}"
echo "run_root=${RUN_ROOT}"
if [[ -f "${PID_FILE}" ]]; then
  PID="$(cat "${PID_FILE}")"
  if kill -0 "${PID}" 2>/dev/null; then
    echo "launcher_pid=${PID} alive=yes"
  else
    echo "launcher_pid=${PID} alive=no"
  fi
else
  echo "launcher_pid=missing"
fi
if [[ -f "${STATE}" ]]; then
  python3 - <<PY
import json
from pathlib import Path
s=json.loads(Path(${STATE@Q}).read_text())
for k in ["status","current_stage","completed_stages","through_stage","next_allowed_stage","waiting_review_gate","exit_code","failure_or_gate_reason","pid","updated_at"]:
    print(f"{k}={s.get(k)}")
PY
else
  echo "run_state=missing"
fi
echo "launcher_log=${RUN_ROOT}/logs/launcher.log"
