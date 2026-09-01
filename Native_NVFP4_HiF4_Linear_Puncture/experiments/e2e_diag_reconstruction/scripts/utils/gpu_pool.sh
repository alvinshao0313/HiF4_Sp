#!/usr/bin/env bash
# GPU pool detection and wave wait. No model/tensor logic.

detect_available_gpus() {
  local smi
  if ! smi="$(nvidia-smi --query-gpu=index,memory.free,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)"; then
    return 0
  fi
  local known=()
  local line idx free total util
  while IFS=',' read -r idx free total util; do
    idx="${idx// /}"
    free="${free// /}"
    total="${total// /}"
    util="${util// /}"
    known+=("${idx}")
  done <<< "${smi}"

  local project_pool="${PROJECT_GPU_POOL:-0,1,2,3}"

  if [[ -n "${GPU_POOL:-}" ]]; then
    local tok
    IFS=',' read -r -a toks <<< "${GPU_POOL}"
    for tok in "${toks[@]}"; do
      tok="${tok// /}"
      [[ -z "${tok}" ]] && continue
      local ok=0
      local k
      for k in "${known[@]}"; do
        if [[ "${k}" == "${tok}" ]]; then
          ok=1
          break
        fi
      done
      if [[ "${ok}" -ne 1 ]]; then
        echo "GPU_POOL id ${tok} is not a nvidia-smi GPU" >&2
        return 1
      fi
      local project_ok=0 project_id
      IFS=',' read -r -a project_ids <<< "${project_pool}"
      for project_id in "${project_ids[@]}"; do
        project_id="${project_id// /}"
        if [[ "${project_id}" == "${tok}" ]]; then
          project_ok=1
          break
        fi
      done
      if [[ "${project_ok}" -ne 1 ]]; then
        echo "GPU_POOL id ${tok} is outside PROJECT_GPU_POOL=${project_pool}" >&2
        return 1
      fi
      printf '%s\n' "${tok}"
    done
    return 0
  fi

  local min_ratio="${GPU_MIN_FREE_RATIO:-0.90}"
  local max_util="${GPU_MAX_UTIL:-10}"
  while IFS=',' read -r idx free total util; do
    idx="${idx// /}"
    free="${free// /}"
    total="${total// /}"
    util="${util// /}"
    local project_ok=0 project_id
    IFS=',' read -r -a project_ids <<< "${project_pool}"
    for project_id in "${project_ids[@]}"; do
      project_id="${project_id// /}"
      if [[ "${project_id}" == "${idx}" ]]; then
        project_ok=1
        break
      fi
    done
    [[ "${project_ok}" -eq 1 ]] || continue
    if awk -v f="${free}" -v t="${total}" -v r="${min_ratio}" -v u="${util}" -v mu="${max_util}" 'BEGIN {
      if (t + 0 <= 0) exit 1
      if ((f + 0) / (t + 0) >= (r + 0) && (u + 0) <= (mu + 0)) exit 0
      exit 1
    }'; then
      printf '%s\n' "${idx}"
    fi
  done <<< "${smi}"
}

resolve_parallel_slots() {
  local n_gpus="$1"
  if [[ -z "${n_gpus}" ]] || [[ "${n_gpus}" -le 0 ]]; then
    echo "no available GPU" >&2
    return 1
  fi
  local slots="${n_gpus}"
  if [[ -n "${MAX_PARALLEL_JOBS:-}" ]]; then
    if [[ "${MAX_PARALLEL_JOBS}" -lt "${slots}" ]]; then
      slots="${MAX_PARALLEL_JOBS}"
    fi
  fi
  if [[ "${slots}" -le 0 ]]; then
    echo "no available GPU" >&2
    return 1
  fi
  printf '%s\n' "${slots}"
}

wait_gpu_wave() {
  local status=0
  local pid
  for pid in "$@"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  return "${status}"
}
