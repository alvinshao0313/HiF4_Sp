#!/usr/bin/env bash
# One-shot 32-layer candidate search. All threshold maps derive from this run.
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
SEARCH_SEED="${SEARCH_SEED:-42}"
OUT="${EXP_DIR}/results/search"
LOG="${EXP_DIR}/logs/01_search_once.log"

mkdir -p "${OUT}" "$(dirname "${LOG}")"
: > "${LOG}"

echo "[$(date --iso-8601=seconds)] search once (seed=${SEARCH_SEED})" | tee -a "${LOG}"
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
  --no-proxy-audit \
  --num-workers "${NUM_WORKERS:-4}" \
  --device cuda \
  --layers all \
  --seed "${SEARCH_SEED}" \
  --output-dir "${OUT}" \
  --overwrite-output \
  --bf16-control-probes 0 \
  --trust-remote-code \
  2>&1 | tee -a "${LOG}"

if [[ ! -f "${OUT}/candidate_permutations.pt" ]]; then
  echo "candidate_permutations.pt missing; abort" >&2
  exit 1
fi
if [[ ! -f "${OUT}/selected_permutations.pt" ]]; then
  echo "selected_permutations.pt missing; abort" >&2
  exit 1
fi
echo "[$(date --iso-8601=seconds)] search done" | tee -a "${LOG}"
