#!/usr/bin/env bash
# Shared paths, structure presets, and launch helpers. No algorithm logic.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
E2E_MODULE_TRAIN="Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.cli.train"
E2E_MODULE_EVAL="Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.cli.evaluate"
E2E_MODULE_SUMMARIZE="Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.cli.summarize"
E2E_MODULE_PREPARE_CALIB="Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.cli.prepare_calibration"
RESULTS_ROOT="${REPO_ROOT}/Native_NVFP4_HiF4_Linear_Puncture/results/e2e_diag_reconstruction"
DIAG_BATCH_SIZE="${DIAG_BATCH_SIZE:-4}"
EVAL_SEED="${EVAL_SEED:-42}"
FAST_EVAL_GROUPS="${FAST_EVAL_GROUPS:-arc,mmlu_pro_300}"
PARALLEL_EVAL_SLOTS="${PARALLEL_EVAL_SLOTS:-1}"

# shellcheck source=utils/gpu_pool.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/utils/gpu_pool.sh"

e2e_train() {
  conda run --no-capture-output -n hif4 python -m "${E2E_MODULE_TRAIN}" "$@"
}

e2e_eval() {
  conda run --no-capture-output -n hif4 python -m "${E2E_MODULE_EVAL}" "$@"
}

e2e_summarize() {
  conda run --no-capture-output -n hif4 python -m "${E2E_MODULE_SUMMARIZE}" "$@"
}

e2e_prepare_calib() {
  conda run --no-capture-output -n hif4 python -m "${E2E_MODULE_PREPARE_CALIB}" "$@"
}

set_structure_args() {
  local preset="$1"
  STRUCTURE_ARGS=()
  case "${preset}" in
    fusable)
      STRUCTURE_ARGS=(--diag_mode fusable)
      ;;
    fusable_r64)
      STRUCTURE_ARGS=(--diag_mode fusable --use_r64)
      ;;
    online)
      STRUCTURE_ARGS=(--diag_mode online)
      ;;
    online_diag_then_r64)
      STRUCTURE_ARGS=(--diag_mode online --use_r64 --rot_order diag_then_rot)
      ;;
    online_r64_then_diag)
      STRUCTURE_ARGS=(--diag_mode online --use_r64 --rot_order rot_then_diag)
      ;;
    *)
      echo "unknown structure preset: ${preset}" >&2
      return 2
      ;;
  esac
}

phase1_run_name() {
  case "$1" in
    fusable) echo "E3_fusable" ;;
    fusable_r64) echo "E4_fusable_r64" ;;
    online) echo "E5_online" ;;
    online_diag_then_r64) echo "E6_online_diag_then_r64" ;;
    online_r64_then_diag) echo "E7_online_r64_then_diag" ;;
    *) return 2 ;;
  esac
}

shared_calib_dir_for() {
  local source="$1"
  local policy="${2:-all}"
  case "${source}" in
    s1k_teacher_cot)
      echo "${RESULTS_ROOT}/shared_calibration/s1k_teacher_cot_${policy}_n128_v32_seed42"
      ;;
    s1k_original)
      echo "${RESULTS_ROOT}/shared_calibration/s1k_original_n128_v32_seed42"
      ;;
    s1k_question)
      echo "${RESULTS_ROOT}/shared_calibration/s1k_question_n128_v32_seed42"
      ;;
    wikitext2)
      echo "${RESULTS_ROOT}/shared_calibration/wikitext2_n128_v32_seed42_len1024"
      ;;
    c4)
      echo "${RESULTS_ROOT}/shared_calibration/c4_n128_v32_seed42_len1024"
      ;;
    *)
      echo "unknown calib source: ${source}" >&2
      return 2
      ;;
  esac
}

init_stage_gpu_pool() {
  mapfile -t AVAILABLE_GPUS < <(detect_available_gpus)
  if [[ ${#AVAILABLE_GPUS[@]} -eq 0 ]]; then
    echo "no available GPU" >&2
    return 1
  fi
  PARALLEL_SLOTS="$(resolve_parallel_slots "${#AVAILABLE_GPUS[@]}")"
  local joined
  joined="$(IFS=,; echo "${AVAILABLE_GPUS[*]}")"
  echo "available_gpus=${joined}"
  echo "parallel_slots=${PARALLEL_SLOTS}"
  echo "reasoning_eval_slots=$(( ${#AVAILABLE_GPUS[@]} / 2 ))"
}

reasoning_eval_slots() {
  echo $(( ${#AVAILABLE_GPUS[@]} / 2 ))
}

require_reasoning_eval_pool() {
  local slots
  slots="$(reasoning_eval_slots)"
  if [[ "${slots}" -le 0 ]]; then
    echo "MMLU-Pro/AIME need 2 GPUs; available=${AVAILABLE_GPUS[*]}" >&2
    return 1
  fi
}

reasoning_eval_pair() {
  local slot="$1"
  local a=$((slot * 2))
  local b=$((a + 1))
  if [[ -z "${AVAILABLE_GPUS[a]:-}" || -z "${AVAILABLE_GPUS[b]:-}" ]]; then
    echo "MMLU-Pro/AIME need 2 GPUs per job; slot=${slot} available=${AVAILABLE_GPUS[*]}" >&2
    return 1
  fi
  printf '%s,%s\n' "${AVAILABLE_GPUS[a]}" "${AVAILABLE_GPUS[b]}"
}

fast_eval_slots() {
  local slots
  slots="$(reasoning_eval_slots)"
  if [[ "${slots}" -le 0 ]]; then
    echo 0
    return 0
  fi
  if [[ "${PARALLEL_EVAL_SLOTS}" -lt "${slots}" ]]; then
    echo "${PARALLEL_EVAL_SLOTS}"
    return 0
  fi
  echo "${slots}"
}

require_fast_eval_pool() {
  require_reasoning_eval_pool
}

fast_eval_gpu() {
  local slot="$1"
  reasoning_eval_pair "${slot}"
}

require_complete_train_run() {
  local dir="$1"
  [[ -f "${dir}/config.json" ]] || { echo "missing ${dir}/config.json" >&2; return 1; }
  [[ -f "${dir}/summary.json" ]] || { echo "missing ${dir}/summary.json" >&2; return 1; }
  [[ -f "${dir}/checkpoint/final_model/conversion_state.pt" ]] || {
    echo "missing ${dir}/checkpoint/final_model/conversion_state.pt" >&2
    return 1
  }
  [[ -f "${dir}/checkpoint/final_model/manifest.json" ]] || {
    echo "missing ${dir}/checkpoint/final_model/manifest.json" >&2
    return 1
  }
  local last_layer=35
  if grep -q '"model_type": "qwen3_moe"' "${dir}/checkpoint/final_model/manifest.json"; then
    last_layer=47
  fi
  local i
  for i in $(seq -w 0 "${last_layer}"); do
    [[ -d "${dir}/layers/layer_${i}" ]] || { echo "missing ${dir}/layers/layer_${i}" >&2; return 1; }
  done
}

require_shared_calib_cache() {
  local root="$1"
  [[ -f "${root}/calibration/train.pt" ]] || { echo "missing shared cache ${root}/calibration/train.pt" >&2; return 1; }
  [[ -f "${root}/calibration/val.pt" ]] || { echo "missing shared cache ${root}/calibration/val.pt" >&2; return 1; }
  [[ -f "${root}/calibration/train_manifest.json" ]] || { echo "missing shared cache train_manifest.json" >&2; return 1; }
  [[ -f "${root}/calibration/val_manifest.json" ]] || { echo "missing shared cache val_manifest.json" >&2; return 1; }
}

write_run_map_and_summarize() {
  local base="$1"
  local map_json="$2"
  printf '%s\n' "${map_json}" > "${base}/run_map.json"
  e2e_summarize \
    --run_map "${base}/run_map.json" \
    --output_json "${base}/summary.json" \
    --output_md "${base}/report.md"
}

fast_eval_run() {
  local gpu="$1"
  local out_dir="$2"
  local variant="$3"
  local artifact="${4:-}"
  local extra=()
  if [[ -n "${artifact}" ]]; then
    extra+=(--artifact_path "${artifact}")
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" e2e_eval \
    --variant "${variant}" \
    --output_dir "${out_dir}" \
    --groups "${FAST_EVAL_GROUPS}" \
    --eval_seed "${EVAL_SEED}" \
    "${extra[@]}"
}
