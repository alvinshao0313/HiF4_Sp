#!/usr/bin/env bash
# Materialize BF16 variants and run IDENTICAL HiF4 W4A4 RTN on each.
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
DTYPE="${DTYPE:-bfloat16}"
SEARCH_DIR="${EXP_DIR}/results/search"
MAPS_DIR="${EXP_DIR}/results/threshold_maps"
LOG="${EXP_DIR}/logs/02_quantize_variants.log"

mkdir -p "${EXP_DIR}/logs" "${EXP_DIR}/tmp"
: > "${LOG}"

if [[ ! -f "${MAPS_DIR}/threshold_report.json" ]]; then
  echo "threshold maps missing; run build_threshold_maps.py first" >&2
  exit 1
fi

CAND_SHA="$("${HIF4_PY}" - <<PY
import hashlib
print(hashlib.sha256(open("${SEARCH_DIR}/candidate_permutations.pt","rb").read()).hexdigest())
PY
)"

# variant_name:perm_file pairs. "identity" uses the base model as-is.
# Override with VARIANTS="tau_0p25 tau_0p50" to process a subset serially
# (disk-peak control: RTN deletes the BF16 staging dir right after success).
if [[ -n "${VARIANTS:-}" ]]; then
  read -r -a _names <<< "${VARIANTS}"
  VARIANT_LIST=()
  for n in "${_names[@]}"; do
    case "${n}" in
      identity) VARIANT_LIST+=("identity:") ;;
      selected_default) VARIANT_LIST+=("selected_default:${SEARCH_DIR}/selected_permutations.pt") ;;
      tau_*) VARIANT_LIST+=("${n}:${MAPS_DIR}/${n}.pt") ;;
      *) echo "unknown variant ${n}" >&2; exit 1 ;;
    esac
  done
else
  VARIANT_LIST=(
    "identity:"
    "selected_default:${SEARCH_DIR}/selected_permutations.pt"
    "tau_0p00:${MAPS_DIR}/tau_0p00.pt"
    "tau_0p25:${MAPS_DIR}/tau_0p25.pt"
    "tau_0p50:${MAPS_DIR}/tau_0p50.pt"
    "tau_1p00:${MAPS_DIR}/tau_1p00.pt"
    "tau_2p00:${MAPS_DIR}/tau_2p00.pt"
  )
fi

for entry in "${VARIANT_LIST[@]}"; do
  name="${entry%%:*}"
  perm_file="${entry##*:}"
  bf16_dir="${EXP_DIR}/tmp/bf16_${name}"
  w4a4_dir="${EXP_DIR}/tmp/w4a4_${name}"

  if [[ "${name}" == "identity" ]]; then
    SRC_MODEL="${MODEL}"
  else
    if [[ -f "${bf16_dir}/config.json" ]]; then
      echo "[$(date --iso-8601=seconds)] reuse materialized ${bf16_dir}" | tee -a "${LOG}"
    else
      echo "[$(date --iso-8601=seconds)] materialize ${name} <- ${perm_file}" | tee -a "${LOG}"
      rm -rf "${bf16_dir}"
      "${HIF4_PY}" "${EXP_DIR}/materialize_threshold_model.py" \
        --model "${MODEL}" \
        --permutations "${perm_file}" \
        --output-dir "${bf16_dir}" \
        --dtype "${DTYPE}" \
        --device cpu \
        --metadata "{\"variant\": \"${name}\", \"candidate_permutations_sha256\": \"${CAND_SHA}\", \"source_search_seed\": 42}" \
        --trust-remote-code \
        2>&1 | tee -a "${LOG}"
    fi
    SRC_MODEL="${bf16_dir}"
  fi

  if [[ -f "${w4a4_dir}/config.json" ]]; then
    echo "[$(date --iso-8601=seconds)] reuse RTN ${w4a4_dir}" | tee -a "${LOG}"
    continue
  fi
  echo "[$(date --iso-8601=seconds)] RTN ${name}" | tee -a "${LOG}"
  rm -rf "${w4a4_dir}"
  mkdir -p "${w4a4_dir}"
  CUDA_VISIBLE_DEVICES="${GPUS}" "${HIF4_PY}" HiFloat4/main.py \
    --model "${SRC_MODEL}" \
    --dtype "${DTYPE}" \
    --hif4w true \
    --hif4_weight_format hif4 \
    --gptq false \
    --gptq_save_path "${w4a4_dir}" \
    --exclude-layers lm_head \
    --trust-remote-code true \
    2>&1 | tee -a "${LOG}"
  "${HIF4_PY}" - <<PY
from transformers import AutoTokenizer
import json
tok = AutoTokenizer.from_pretrained("${MODEL}", trust_remote_code=True)
tok.save_pretrained("${w4a4_dir}")
meta = {
    "variant": "${name}",
    "quantization": "HiF4 W4A4 RTN",
    "excluded_layers": ["lm_head"],
    "source_search_seed": 42,
    "candidate_permutations_sha256": "${CAND_SHA}",
    "src_model": "${SRC_MODEL}",
}
open("${w4a4_dir}/variant_metadata.json", "w").write(json.dumps(meta, indent=2))
print("tokenizer+metadata saved")
PY
  # BF16 staging dir is no longer needed once RTN succeeded; free 8GB peak.
  if [[ "${name}" != "identity" ]]; then
    rm -rf "${bf16_dir}"
    echo "[$(date --iso-8601=seconds)] freed ${bf16_dir}" | tee -a "${LOG}"
  fi
done

echo "[$(date --iso-8601=seconds)] quantize variants done" | tee -a "${LOG}"
