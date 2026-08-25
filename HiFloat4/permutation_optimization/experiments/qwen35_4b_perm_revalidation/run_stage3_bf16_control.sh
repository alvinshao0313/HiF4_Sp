#!/usr/bin/env bash
# Stage 3: BF16-only paired control (identity vs permuted, NO quantization).
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

GPUS="${GPUS:-4}"
IDENTITY="${EXP_DIR}/tmp/stage2_identity_bf16"
REORDERED="${EXP_DIR}/tmp/stage2_reordered_bf16"
OUT="${EXP_DIR}/results/stage3_bf16_control"
LOG="${EXP_DIR}/logs/stage3_bf16_control.log"
TASKS="${TASKS:-arc_easy,arc_challenge,mmlu,wikitext}"

mkdir -p "${OUT}" "$(dirname "${LOG}")"
: > "${LOG}"

for pair in "identity:${IDENTITY}" "permuted:${REORDERED}"; do
  name="${pair%%:*}"
  ckpt="${pair##*:}"
  if [[ ! -f "${ckpt}/config.json" ]]; then
    echo "missing ckpt ${ckpt}; run Stage 2 first" >&2
    exit 1
  fi
  echo "[$(date --iso-8601=seconds)] Stage 3 bf16 ${name}" | tee -a "${LOG}"
  CUDA_VISIBLE_DEVICES="${GPUS}" "${HIF4_PY}" "${EXP_DIR}/eval_paired.py" \
    --model_path "${ckpt}" \
    --mode bf16 \
    --tasks "${TASKS}" \
    --num_fewshot 0 \
    --batch_size 8 \
    --output_json "${OUT}/bf16_${name}.json" \
    2>&1 | tee -a "${LOG}"
done

echo "[$(date --iso-8601=seconds)] Stage 3 done" | tee -a "${LOG}"
