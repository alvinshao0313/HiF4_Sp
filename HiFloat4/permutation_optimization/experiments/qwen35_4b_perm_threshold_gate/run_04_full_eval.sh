#!/usr/bin/env bash
# Full validation on tasks NOT used for threshold selection:
# BoolQ / HellaSwag / WinoGrande / MMLU.
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
EVAL_PAIRED="${REPO_ROOT}/HiFloat4/permutation_optimization/experiments/qwen35_4b_perm_revalidation/eval_paired.py"
FAST_SUMMARY="${EXP_DIR}/results/fast_eval/summary.json"
OUT="${EXP_DIR}/results/full_eval"
META="${EXP_DIR}/results/threshold_maps/variant_metadata.json"
LOG="${EXP_DIR}/logs/04_full_eval.log"

mkdir -p "${OUT}" "$(dirname "${LOG}")"
: > "${LOG}"

if [[ ! -f "${FAST_SUMMARY}" ]]; then
  echo "fast summary missing; run run_03_fast_eval.sh first" >&2
  exit 1
fi

SELECTED="$("${HIF4_PY}" - <<PY
import json
s = json.loads(open("${FAST_SUMMARY}").read())
print(" ".join(s["selected_thresholds"]))
PY
)"

if [[ -z "${SELECTED}" ]]; then
  echo "fast stage selected no thresholds; stop full eval per plan" | tee -a "${LOG}"
  echo "结论：当前候选排序在阈值门控后仍未显示稳定下游收益（fast 阶段已判定）" | tee -a "${LOG}"
  exit 0
fi

VARIANTS=(identity selected_default ${SELECTED})
echo "[$(date --iso-8601=seconds)] full eval variants: ${VARIANTS[*]}" | tee -a "${LOG}"

# Re-quantize any tau winners whose ckpts were freed after fast eval.
MISSING=""
for name in "${VARIANTS[@]}"; do
  if [[ ! -f "${EXP_DIR}/tmp/w4a4_${name}/config.json" ]]; then
    MISSING="${MISSING} ${name}"
  fi
done
if [[ -n "${MISSING}" ]]; then
  echo "[$(date --iso-8601=seconds)] re-quantizing:${MISSING}" | tee -a "${LOG}"
  VARIANTS="${MISSING}" bash "${EXP_DIR}/run_02_quantize_variants.sh"
fi

for name in "${VARIANTS[@]}"; do
  ckpt="${EXP_DIR}/tmp/w4a4_${name}"
  if [[ -f "${OUT}/${name}.json" ]]; then
    echo "[$(date --iso-8601=seconds)] reuse ${OUT}/${name}.json" | tee -a "${LOG}"
    continue
  fi
  echo "[$(date --iso-8601=seconds)] full eval ${name}" | tee -a "${LOG}"
  CUDA_VISIBLE_DEVICES="${GPUS}" "${HIF4_PY}" "${EVAL_PAIRED}" \
    --model_path "${ckpt}" \
    --mode w4a4 \
    --tasks boolq,hellaswag,winogrande,mmlu \
    --num_fewshot 0 \
    --batch_size 8 \
    --output_json "${OUT}/${name}.json" \
    2>&1 | tee -a "${LOG}"
done

"${HIF4_PY}" "${EXP_DIR}/summarize_threshold_results.py" \
  --stage full \
  --eval-dir "${OUT}" \
  --threshold-metadata "${META}" \
  --output-dir "${EXP_DIR}/results" | tee -a "${LOG}"

echo "[$(date --iso-8601=seconds)] full eval done" | tee -a "${LOG}"
