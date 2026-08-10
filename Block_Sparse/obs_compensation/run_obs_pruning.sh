#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

SOURCE_ARTIFACTS_DIR="${SOURCE_ARTIFACTS_DIR:-Block_Sparse/outputs/qwen35_4b_mask_source/pruning_artifacts}"
OUTPUT_DIR="${OUTPUT_DIR:-Block_Sparse/outputs/qwen35_4b_obs_init}"
MODEL_PATH="${MODEL_PATH:-}"
CALIBRATION_DATASET="${CALIBRATION_DATASET:-s1k}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-128}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-1024}"
OBS_PERCDAMP="${OBS_PERCDAMP:-0.01}"
SOLVER_BLOCK_SIZE="${SOLVER_BLOCK_SIZE:-128}"
OBS_ORDER_POLICY="${OBS_ORDER_POLICY:-auto}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"

ARGS=(
  --source_artifacts_dir "${SOURCE_ARTIFACTS_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --calibration_dataset "${CALIBRATION_DATASET}"
  --calibration_samples "${CALIBRATION_SAMPLES}"
  --sequence_length "${SEQUENCE_LENGTH}"
  --obs_percdamp "${OBS_PERCDAMP}"
  --solver_block_size "${SOLVER_BLOCK_SIZE}"
  --obs_order_policy "${OBS_ORDER_POLICY}"
  --dtype "${DTYPE}"
  --device "${DEVICE}"
  --seed "${SEED}"
)

if [[ -n "${MODEL_PATH}" ]]; then
  ARGS+=(--model_path "${MODEL_PATH}")
fi

conda run -n hif4 --no-capture-output \
  python Block_Sparse/obs_compensation/run_obs_pruning.py "${ARGS[@]}"
