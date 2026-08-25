#!/usr/bin/env bash
# Shared helpers for dynamic input sparse e2e harness.
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${EXP_DIR}/../../.." && pwd)"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-4B}"
CONDA_ENV="${CONDA_ENV:-hif4}"

# Prefer free GPUs; override with CUDA_VISIBLE_DEVICES / GPU_POOL.
GPU_POOL="${GPU_POOL:-0,1,6,7}"

run_py() {
  conda run -n "${CONDA_ENV}" --no-capture-output "$@"
}

ensure_run_dir() {
  if [[ -n "${RUN_DIR:-}" ]]; then
    mkdir -p "${RUN_DIR}"/{logs,telemetry}
    return
  fi
  local ts
  ts="$(date -u +%Y%m%dT%H%M%SZ)"
  RUN_DIR="${EXP_DIR}/results/${ts}"
  mkdir -p "${RUN_DIR}"/{logs,telemetry}
  echo "${RUN_DIR}" > "${EXP_DIR}/results/LATEST_RUN_DIR.txt"
}

marker_done() {
  local d="$1"
  [[ -f "${d}/DONE" ]]
}

mark_done() {
  local d="$1"
  mkdir -p "${d}"
  date -u +%Y-%m-%dT%H:%M:%SZ > "${d}/DONE"
}

write_manifest() {
  ensure_run_dir
  run_py python - <<'PY' "${RUN_DIR}" "${MODEL_PATH}"
import json, sys, platform
from pathlib import Path
run_dir, model = Path(sys.argv[1]), sys.argv[2]
info = {
    "timestamp": run_dir.name,
    "model_path": model,
    "k_block_size": 64,
    "mask_granularity": "per_token",
    "m1_token_chunk_size": 8,
    "tp_pp_dp": [1, 1, 1],
    "enforce_eager": True,
    "platform": platform.platform(),
}
try:
    import torch
    info["torch"] = torch.__version__
    info["cuda"] = torch.version.cuda
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
except Exception as e:
    info["torch_error"] = str(e)
try:
    import transformers
    info["transformers"] = transformers.__version__
except Exception:
    pass
try:
    import vllm
    info["vllm"] = getattr(vllm, "__version__", "unknown")
    info["vllm_file"] = getattr(vllm, "__file__", "")
except Exception as e:
    info["vllm_error"] = str(e)
(run_dir / "manifest.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
print(run_dir / "manifest.json")
PY
}

pick_gpu() {
  # Round-robin from GPU_POOL using a counter file under RUN_DIR.
  ensure_run_dir
  local counter_file="${RUN_DIR}/.gpu_rr"
  local IFS=','
  read -r -a gpus <<< "${GPU_POOL}"
  local n=${#gpus[@]}
  local idx=0
  if [[ -f "${counter_file}" ]]; then
    idx="$(cat "${counter_file}")"
  fi
  local gpu="${gpus[$((idx % n))]}"
  echo $((idx + 1)) > "${counter_file}"
  echo "${gpu}"
}

run_arc() {
  local method="$1" keep="$2" out_dir="$3" gpu="$4" limit="${5:-}"
  mkdir -p "${out_dir}/arc"
  local out_json="${out_dir}/arc/lm_eval.json"
  if [[ -f "${out_json}" && -f "${out_dir}/arc/DONE" ]]; then
    echo "[skip] ARC ${method} keep=${keep}"
    return 0
  fi
  local limit_args=()
  if [[ -n "${limit}" ]]; then
    limit_args=(--limit "${limit}")
  fi
  echo "[arc] method=${method} keep=${keep} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" run_py python \
    "${EXP_DIR}/eval_lm_eval_dynamic_input_sparse.py" \
    --model_path "${MODEL_PATH}" \
    --tasks arc_easy,arc_challenge \
    --num_fewshot 0 \
    --batch_size 8 \
    --dtype bfloat16 \
    --device cuda \
    --dynamic_input_sparse_method "${method}" \
    --dynamic_input_keep_ratio "${keep}" \
    --output_json "${out_json}" \
    "${limit_args[@]}"
  date -u +%Y-%m-%dT%H:%M:%SZ > "${out_dir}/arc/DONE"
}

run_vllm_task() {
  local method="$1" keep="$2" task_name="$3" out_dir="$4" gpu="$5"
  local max_samples="${6:-}"
  local disable_thinking="${7:-0}"
  mkdir -p "${out_dir}/${task_name}"
  if [[ -f "${out_dir}/${task_name}/DONE" ]]; then
    echo "[skip] ${task_name} ${method} keep=${keep}"
    return 0
  fi
  local datasets
  if [[ "${task_name}" == "mmlu_pro" ]]; then
    datasets="mmlu_pro|0"
  else
    datasets="aime25_avg5"
  fi
  local tele_dir="${RUN_DIR}/telemetry/${method}_keep${keep//./}_${task_name}"
  mkdir -p "${tele_dir}"
  local extra=()
  if [[ "${method}" != "none" ]]; then
    extra+=(
      --dynamic_input_sparse_method "${method}"
      --dynamic_input_keep_ratio "${keep}"
      --dynamic_input_telemetry_dir "${tele_dir}"
    )
  fi
  if [[ -n "${max_samples}" ]]; then
    extra+=(--max_samples "${max_samples}")
  fi
  if [[ "${disable_thinking}" == "1" ]]; then
    extra+=(--disable_thinking)
  fi
  echo "[vllm] task=${datasets} method=${method} keep=${keep} gpu=${gpu}"
  local attempt=1
  local max_attempts="${VLLM_MAX_ATTEMPTS:-5}"
  while true; do
    set +e
    CUDA_VISIBLE_DEVICES="${gpu}" \
      HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
      run_py python "${REPO_ROOT}/main.py" \
      --model_path "${MODEL_PATH}" \
      --datasets "${datasets}" \
      --tensor_parallel_size 1 \
      --pipeline_parallel_size 1 \
      --data_parallel_size 1 \
      --enforce_eager \
      --fake_act_quant none \
      --kv_quant_format none \
      --max_model_length 32768 \
      --max_new_tokens 32768 \
      --temperature 0.7 \
      --top_p 0.8 \
      --top_k 20 \
      --gpu_memory_utilization 0.90 \
      --output_dir "${out_dir}/${task_name}" \
      "${extra[@]}"
    local rc=$?
    set -e
    if [[ "${rc}" -eq 0 ]]; then
      break
    fi
    if [[ "${attempt}" -ge "${max_attempts}" ]]; then
      echo "[vllm] FAILED after ${max_attempts} attempts: ${datasets} method=${method}" >&2
      return "${rc}"
    fi
    echo "[vllm] attempt ${attempt}/${max_attempts} failed (rc=${rc}); retry in 30s..." >&2
    attempt=$((attempt + 1))
    sleep 30
  done
  date -u +%Y-%m-%dT%H:%M:%SZ > "${out_dir}/${task_name}/DONE"
}

run_method_full() {
  local tag="$1" method="$2" keep="$3"
  local gpu="${4:-}"
  ensure_run_dir
  local out_dir="${RUN_DIR}/${tag}"
  mkdir -p "${out_dir}"
  if marker_done "${out_dir}"; then
    echo "[skip] method group ${tag}"
    return 0
  fi
  # Sequential within method. If gpu arg omitted, pick once and reuse (avoids RR races).
  if [[ -z "${gpu}" ]]; then
    gpu="$(pick_gpu)"
  fi
  echo "[method] ${tag} pinned_gpu=${gpu}"
  run_arc "${method}" "${keep}" "${out_dir}" "${gpu}"
  run_vllm_task "${method}" "${keep}" "mmlu_pro" "${out_dir}" "${gpu}" 300 1
  run_vllm_task "${method}" "${keep}" "aime25" "${out_dir}" "${gpu}" "" 0
  mark_done "${out_dir}"
}
