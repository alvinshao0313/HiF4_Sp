#!/usr/bin/env bash
# Re-run MLP hierarchical perm with s1k calib (128 full samples, no truncate).
# Algorithm unchanged. Only report whether calib-set quant error decreased.
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${EXP_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/HiFloat4:${PYTHONPATH:-}"

HIF4_PY="${HIF4_PY:-/home/shaoyuantian/anaconda3/envs/hif4/bin/python}"
export PATH="$(dirname "${HIF4_PY}"):${PATH}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "hif4" ]]; then
  set +u
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate hif4
  set -u
fi

MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
GPUS="${GPUS:-4}"
PERM_OUT="${EXP_DIR}/results/perm_search"
LOG="${EXP_DIR}/logs/perm_s1k.log"

mkdir -p "${PERM_OUT}" "$(dirname "${LOG}")"
: > "${LOG}"

echo "[$(date --iso-8601=seconds)] === MLP reorder on s1k (128, no truncate) ===" | tee -a "${LOG}"
CUDA_VISIBLE_DEVICES="${GPUS}" "${HIF4_PY}" -m permutation_optimization.run_mlp_reorder \
  --model "${MODEL}" \
  --calibration-dataset s1k \
  --calibration-nsamples 128 \
  --calibration-seqlen 0 \
  --activation-rows 512 \
  --weight-rows 512 \
  --refine-passes 0 \
  --refine-bad-blocks 8 \
  --num-workers "${NUM_WORKERS:-8}" \
  --device cuda \
  --layers all \
  --output-dir "${PERM_OUT}" \
  --trust-remote-code \
  2>&1 | tee -a "${LOG}"

echo "[$(date --iso-8601=seconds)] === Summarize calib-set error deltas ===" | tee -a "${LOG}"
"${HIF4_PY}" "${EXP_DIR}/summarize_error_audit.py" \
  --metrics "${PERM_OUT}/layer_metrics.jsonl" \
  --output_dir "${EXP_DIR}/results" \
  2>&1 | tee -a "${LOG}"

echo "[$(date --iso-8601=seconds)] done" | tee -a "${LOG}"
cat "${EXP_DIR}/results/error_audit.md"
