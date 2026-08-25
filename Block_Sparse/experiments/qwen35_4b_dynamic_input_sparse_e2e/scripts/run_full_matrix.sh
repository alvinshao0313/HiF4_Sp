#!/usr/bin/env bash
# Full e2e matrix. Supports multi-GPU parallel M8/M1 ratios after dense.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

ensure_run_dir
export RUN_DIR
write_manifest
LOG="${RUN_DIR}/logs/full_matrix.log"
exec > >(tee -a "${LOG}") 2>&1

echo "RUN_DIR=${RUN_DIR}"
echo "GPU_POOL=${GPU_POOL}"

# 1) unit + smoke (unless skipped)
if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
  bash "${SCRIPT_DIR}/run_unit_and_smoke.sh"
fi

# 2) dense baseline (must finish before sparse comparisons)
bash "${SCRIPT_DIR}/run_dense.sh"

# 3) M8 keep ratios in parallel, one free GPU each
echo "=== M8 parallel sweep ==="
IFS=',' read -r -a gpus <<< "${GPU_POOL}"
pids=()
i=0
for keep in 0.75 0.50 0.25; do
  gpu="${gpus[$((i % ${#gpus[@]}))]}"
  (
    export RUN_DIR PIN_GPU="${gpu}"
    bash "${SCRIPT_DIR}/run_m8.sh" "${keep}"
  ) > "${RUN_DIR}/logs/m8_${keep}_worker.log" 2>&1 &
  pids+=($!)
  i=$((i + 1))
  sleep 15
done
ec=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    ec=1
  fi
done
if [[ "${ec}" -ne 0 ]]; then
  echo "M8 parallel sweep had failures" >&2
  exit 1
fi

# Interim summary after M8
run_py python "${EXP_DIR}/summarize_results.py" "${RUN_DIR}" || true

# 4) M1 (with stop rule)
bash "${SCRIPT_DIR}/run_m1.sh"

# 5) final summary
run_py python "${EXP_DIR}/summarize_results.py" "${RUN_DIR}"
date -u +%Y-%m-%dT%H:%M:%SZ > "${RUN_DIR}/logs/full_matrix.DONE"
echo "FULL MATRIX DONE -> ${RUN_DIR}"
