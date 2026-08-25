#!/usr/bin/env bash
# Stage 2: full 32-layer search (only after Stage 1 gates pass). 4 workers.
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
OUT="${EXP_DIR}/results/stage2_full_search/perm_search"
REORDERED="${EXP_DIR}/tmp/stage2_reordered_bf16"
IDENTITY="${EXP_DIR}/tmp/stage2_identity_bf16"
LOG="${EXP_DIR}/logs/stage2_full_search.log"

mkdir -p "${OUT}" "$(dirname "${LOG}")" "${EXP_DIR}/tmp"
: > "${LOG}"

echo "[$(date --iso-8601=seconds)] Stage 2: full 32-layer search" | tee -a "${LOG}"
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
  --num-workers "${NUM_WORKERS:-4}" \
  --device cuda \
  --layers all \
  --output-dir "${OUT}" \
  --overwrite-output \
  --save-reordered-model "${REORDERED}" \
  --save-identity-copy "${IDENTITY}" \
  --bf16-control-probes 16 \
  --bf16-control-seqlen 128 \
  --bf16-control-seed 42 \
  --trust-remote-code \
  2>&1 | tee -a "${LOG}"

echo "[$(date --iso-8601=seconds)] Stage 2 done" | tee -a "${LOG}"
