#!/usr/bin/env bash
# Stage 1: code/proxy audit on representative layers 0, 15, 31 (single worker).
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${EXP_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/Block_Sparse:${REPO_ROOT}/HiFloat4:${PYTHONPATH:-}"

HIF4_PY="${HIF4_PY:-/home/shaoyuantian/anaconda3/envs/hif4/bin/python}"
export PATH="$(dirname "${HIF4_PY}"):${PATH}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "hif4" ]]; then
  set +u
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate hif4
  set -u
fi

MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
GPUS="${GPUS:-4}"
OUT="${EXP_DIR}/results/stage1_layer_audit/perm_search"
LOG="${EXP_DIR}/logs/stage1_layer_audit.log"

mkdir -p "${OUT}" "$(dirname "${LOG}")"
: > "${LOG}"

echo "[$(date --iso-8601=seconds)] Stage 1: layers 0,15,31 audit" | tee -a "${LOG}"
CUDA_VISIBLE_DEVICES="${GPUS}" "${HIF4_PY}" -m permutation_optimization.run_mlp_reorder \
  --model "${MODEL}" \
  --calibration-dataset s1k \
  --calibration-nsamples 128 \
  --calibration-seqlen 0 \
  --activation-rows 512 \
  --weight-rows 512 \
  --candidate-window 128 \
  --neighbor-k 32 \
  --beam-width-g4 4 \
  --beam-width-g64 4 \
  --refine-passes 0 \
  --refine-enabled \
  --refine-max-rounds 2 \
  --refine-candidates-per-round 64 \
  --validation-seeds 42,43,44 \
  --min-relative-improvement 0.001 \
  --max-bf16-reorder-drift 0.002 \
  --num-workers 1 \
  --device cuda \
  --layers 0,15,31 \
  --output-dir "${OUT}" \
  --overwrite-output \
  --trust-remote-code \
  2>&1 | tee -a "${LOG}"

echo "[$(date --iso-8601=seconds)] Stage 1 done" | tee -a "${LOG}"
