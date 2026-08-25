#!/usr/bin/env bash
# Fast screening: ARC-Easy / ARC-Challenge / PIQA on all W4A4 variants.
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
OUT="${EXP_DIR}/results/fast_eval"
META="${EXP_DIR}/results/threshold_maps/variant_metadata.json"
LOG="${EXP_DIR}/logs/03_fast_eval.log"

mkdir -p "${OUT}" "$(dirname "${LOG}")"
: > "${LOG}"

if [[ -n "${VARIANTS:-}" ]]; then
  read -r -a VARIANT_LIST <<< "${VARIANTS}"
else
  VARIANT_LIST=(identity selected_default tau_0p00 tau_0p25 tau_0p50 tau_1p00 tau_2p00)
fi
for name in "${VARIANT_LIST[@]}"; do
  ckpt="${EXP_DIR}/tmp/w4a4_${name}"
  if [[ -f "${OUT}/${name}.json" ]]; then
    echo "[$(date --iso-8601=seconds)] reuse ${OUT}/${name}.json" | tee -a "${LOG}"
    continue
  fi
  if [[ ! -f "${ckpt}/config.json" ]]; then
    echo "missing ${ckpt}; run run_02_quantize_variants.sh first" >&2
    exit 1
  fi
  echo "[$(date --iso-8601=seconds)] fast eval ${name}" | tee -a "${LOG}"
  CUDA_VISIBLE_DEVICES="${GPUS}" "${HIF4_PY}" "${EVAL_PAIRED}" \
    --model_path "${ckpt}" \
    --mode w4a4 \
    --tasks arc_easy,arc_challenge,piqa \
    --num_fewshot 0 \
    --batch_size 8 \
    --output_json "${OUT}/${name}.json" \
    2>&1 | tee -a "${LOG}"
  # Free the W4A4 ckpt for tau variants after their scores are persisted;
  # run_04 re-materializes only the winning thresholds if needed.
  if [[ "${name}" == tau_* && "${KEEP_TAU_CKPT:-0}" != "1" ]]; then
    rm -rf "${ckpt}"
    echo "[$(date --iso-8601=seconds)] freed ${ckpt} (scores kept at ${OUT}/${name}.json)" | tee -a "${LOG}"
  fi
done

if [[ "${SUMMARIZE:-1}" == "1" ]]; then
  # threshold metadata for the summarizer
  "${HIF4_PY}" - <<PY
import json
from pathlib import Path
exp = Path("${EXP_DIR}")
report = json.loads((exp / "results/threshold_maps/threshold_report.json").read_text())
meta = {}
for name, info in report["per_threshold"].items():
    meta[name] = {"threshold_pct": info["threshold_pct"], "n_reordered": info["n_reordered"]}
meta["selected_default"] = {"threshold_pct": None, "n_reordered": None}
(exp / "results/threshold_maps/variant_metadata.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))
PY

  "${HIF4_PY}" "${EXP_DIR}/summarize_threshold_results.py" \
    --stage fast \
    --eval-dir "${OUT}" \
    --threshold-metadata "${META}" \
    --output-dir "${OUT}" | tee -a "${LOG}"
fi

echo "[$(date --iso-8601=seconds)] fast eval done" | tee -a "${LOG}"
