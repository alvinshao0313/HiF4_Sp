#!/usr/bin/env bash
# Build shared s1k_teacher_cot (policy=all) cache on 4 GPUs, then merge.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"

E2E_MODULE_BUILD="fp4_convert.experiments.e2e_diag_reconstruction.cli.build_teacher_cot_cache"
NUM_SHARDS="${NUM_SHARDS:-4}"
GPU_POOL="${GPU_POOL:-0,1,6,7}"
CALIB="${CALIB_CACHE_DIR:-$(shared_calib_dir_for s1k_teacher_cot all)}"
SHARDS_ROOT="${SHARDS_ROOT:-${CALIB}/shards}"
TEACHER_MAX_NEW_TOKENS="${TEACHER_MAX_NEW_TOKENS:-32768}"

e2e_build_teacher_cot() {
  conda run --no-capture-output -n hif4 python -m "${E2E_MODULE_BUILD}" "$@"
}

mapfile -t GPUS < <(detect_available_gpus)
if [[ ${#GPUS[@]} -ne "${NUM_SHARDS}" ]]; then
  echo "expected ${NUM_SHARDS} GPUs from GPU_POOL=${GPU_POOL}, got: ${GPUS[*]:-}" >&2
  exit 1
fi

if require_shared_calib_cache "${CALIB}" 2>/dev/null; then
  echo "shared calib cache already complete: ${CALIB}"
  exit 0
fi

if [[ -e "${CALIB}/calibration" ]]; then
  echo "incomplete calibration at ${CALIB}/calibration; refusing to continue" >&2
  exit 1
fi

mkdir -p "${SHARDS_ROOT}"
echo "=== Teacher-CoT 4-GPU cache build ==="
echo "calib=${CALIB}"
echo "shards_root=${SHARDS_ROOT}"
echo "gpus=${GPUS[*]}"
echo "num_shards=${NUM_SHARDS}"
echo "teacher_max_new_tokens=${TEACHER_MAX_NEW_TOKENS}"

pids=()
for shard_id in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu="${GPUS[${shard_id}]}"
  shard_dir="${SHARDS_ROOT}/shard_${shard_id}"
  mkdir -p "${shard_dir}"
  log="${shard_dir}/worker.log"
  echo "launch shard_id=${shard_id} gpu=${gpu} log=${log}"
  (
    CUDA_VISIBLE_DEVICES="${gpu}" e2e_build_teacher_cot worker \
      --calib_cache_dir "${CALIB}" \
      --output_shard_dir "${shard_dir}" \
      --shard_id "${shard_id}" \
      --num_shards "${NUM_SHARDS}" \
      --teacher_trace_policy all \
      --teacher_max_new_tokens "${TEACHER_MAX_NEW_TOKENS}"
  ) >"${log}" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" -ne 0 ]]; then
  echo "one or more shard workers failed; see ${SHARDS_ROOT}/shard_*/worker.log" >&2
  exit 1
fi

echo "=== merge shards ==="
e2e_build_teacher_cot merge \
  --calib_cache_dir "${CALIB}" \
  --shards_root "${SHARDS_ROOT}" \
  --num_shards "${NUM_SHARDS}" \
  --teacher_trace_policy all \
  --teacher_max_new_tokens "${TEACHER_MAX_NEW_TOKENS}"

require_shared_calib_cache "${CALIB}"
echo "teacher_cot_cache_ready=${CALIB}"
