#!/usr/bin/env bash
# Detached launcher: nohup + setsid; survives Cursor/SSH/terminal exit.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
RUNS_ROOT="${ROOT}/Native_NVFP4_HiF4_Linear_Puncture/results/internal_error_accumulation/qwen3_30b_a3b_e0_e1_formal/runs"

RUN_ID=""
THROUGH_STAGE=""
FROM_STAGE=""
CREATE_NEW=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --through-stage) THROUGH_STAGE="$2"; shift 2 ;;
    --from-stage) FROM_STAGE="$2"; shift 2 ;;
    --create-new) CREATE_NEW=1; shift 1 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${THROUGH_STAGE}" ]]; then
  echo "usage: launch_detached.sh --through-stage STAGE [--run-id ID | --create-new]" >&2
  exit 2
fi

if [[ "${CREATE_NEW}" -eq 1 ]]; then
  if [[ -n "${RUN_ID}" ]]; then
    echo "--create-new and --run-id are mutually exclusive" >&2
    exit 2
  fi
  RUN_ID="iea_$(date -u +%Y%m%dT%H%M%SZ)_$$"
fi

if [[ -z "${RUN_ID}" ]]; then
  echo "must pass --run-id or --create-new" >&2
  exit 2
fi

RUN_ROOT="${RUNS_ROOT}/${RUN_ID}"
mkdir -p "${RUN_ROOT}/logs"
PID_FILE="${RUN_ROOT}/launcher.pid"
STATE_FILE="${RUN_ROOT}/run_state.json"
LOG_FILE="${RUN_ROOT}/logs/launcher.log"

if [[ -f "${PID_FILE}" ]]; then
  OLD_PID="$(cat "${PID_FILE}" || true)"
  if [[ -n "${OLD_PID}" ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
    echo "REFUSE duplicate start: RUN_ID=${RUN_ID} still alive pid=${OLD_PID}" >&2
    exit 3
  fi
fi

EXTRA=()
if [[ -n "${FROM_STAGE}" ]]; then
  EXTRA+=(--from-stage "${FROM_STAGE}")
fi

cd "${ROOT}"
nohup setsid bash "${SCRIPT_DIR}/run_formal.sh" \
  --run-id "${RUN_ID}" \
  --through-stage "${THROUGH_STAGE}" \
  "${EXTRA[@]}" \
  </dev/null >>"${LOG_FILE}" 2>&1 &
echo $! > "${PID_FILE}"
PID="$(cat "${PID_FILE}")"

# Seed a minimal run_state if pipeline has not written yet.
if [[ ! -f "${STATE_FILE}" ]]; then
  python3 - <<PY
import json
from datetime import datetime, timezone
from pathlib import Path
path = Path(${STATE_FILE@Q})
payload = {
  "run_id": ${RUN_ID@Q},
  "status": "RUNNING",
  "current_stage": None,
  "completed_stages": [],
  "pid": int(${PID}),
  "started_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
  "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
  "exit_code": None,
  "failure_or_gate_reason": None,
  "next_allowed_stage": "S0_INFRA",
  "through_stage": ${THROUGH_STAGE@Q},
  "waiting_review_gate": None,
}
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
fi

echo "RUN_ID=${RUN_ID}"
echo "PID=${PID}"
echo "run_state=${STATE_FILE}"
echo "launcher_log=${LOG_FILE}"
echo "through_stage=${THROUGH_STAGE}"
