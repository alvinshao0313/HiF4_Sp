#!/usr/bin/env bash
# Detached lm_eval runner: survives terminal/Cursor exit.
# Skips models that already have JSON; runs remaining on GPUs 4/5/7.
set -eo pipefail

REPO=/home/shaoyuantian/program/HiF4_Sp
CONDA_BASE=/home/shaoyuantian/anaconda3
HIF4_PY="${CONDA_BASE}/envs/hif4/bin/python"
cd "${REPO}"

# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
set +u
conda activate hif4
set -u

unset HF_ENDPOINT || true
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export TOKENIZERS_PARALLELISM=false
export PATH="${CONDA_BASE}/envs/hif4/bin:${PATH}"

OUT="${REPO}/Block_Sparse/experiments/wikitext2_calib/results/lm_eval"
CKPT_ROOT="${REPO}/Block_Sparse/experiments/wikitext2_calib/outputs"
mkdir -p "${OUT}"
LOG="${OUT}/detached_runner.log"

MODELS=(
  qwen35_27b_magnitude_s0.20_b64
  qwen35_27b_magnitude_s0.20_b128
  qwen35_27b_random_s0.20_b64
  qwen35_27b_random_s0.20_b128
  qwen35_27b_fisher_s0.20_b64
  qwen35_27b_fisher_s0.20_b128
  qwen35_27b_fisher_s0.20_b64x32
)
GPUS=(4 5)
BATCH_SIZE="${BATCH_SIZE:-8}"

log() { echo "[$(date -Is)] $*" | tee -a "${LOG}"; }

needs_run() {
  local tag="$1"
  [[ ! -f "${OUT}/${tag}_arc_mmlu.json" ]]
}

gpu_busy() {
  local gpu="$1"
  local pid
  for pid in $(pgrep -f "${REPO}/Block_Sparse/scripts/eval_lm_eval.py" 2>/dev/null || true); do
    if [[ -r "/proc/${pid}/environ" ]] && \
       tr '\0' '\n' < "/proc/${pid}/environ" 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=${gpu}"; then
      return 0
    fi
  done
  return 1
}

run_model() {
  local gpu="$1" tag="$2"
  local model="${CKPT_ROOT}/${tag}"
  local out_json="${OUT}/${tag}_arc_mmlu.json"
  local model_log="${OUT}/${tag}_arc_mmlu.log"
  local lock="${OUT}/${tag}.lock"

  if [[ -f "${out_json}" ]]; then
    log "SKIP ${tag} (json exists)"
    return 0
  fi
  if [[ -f "${lock}" ]]; then
    local oldpid
    oldpid="$(cat "${lock}" 2>/dev/null || true)"
    if [[ -n "${oldpid}" ]] && kill -0 "${oldpid}" 2>/dev/null; then
      log "SKIP ${tag} (lock held by pid ${oldpid})"
      return 0
    fi
  fi

  log "START gpu=${gpu} ${tag}"
  (
    echo $$ > "${lock}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${HIF4_PY}" \
      "${REPO}/Block_Sparse/scripts/eval_lm_eval.py" \
      --model_path "${model}" \
      --tasks arc_easy,arc_challenge,mmlu \
      --batch_size "${BATCH_SIZE}" \
      --output_json "${out_json}" \
      >"${model_log}" 2>&1
    local rc=$?
    rm -f "${lock}"
    log "DONE gpu=${gpu} ${tag} rc=${rc}"
  ) &
}

log "==== detached runner started pid=$$ python=${HIF4_PY} ===="

while true; do
  pending=()
  for tag in "${MODELS[@]}"; do
    if needs_run "${tag}"; then
      pending+=("${tag}")
    fi
  done

  if [[ "${#pending[@]}" -eq 0 ]]; then
    log "ALL COMPLETE"
    date -Is > "${OUT}/all_done.flag"
    exit 0
  fi

  log "pending=${#pending[@]}: ${pending[*]}"

  for gpu in "${GPUS[@]}"; do
    if gpu_busy "${gpu}"; then
      continue
    fi
    for tag in "${pending[@]}"; do
      if needs_run "${tag}"; then
        lock="${OUT}/${tag}.lock"
        if [[ -f "${lock}" ]]; then
          oldpid="$(cat "${lock}" 2>/dev/null || true)"
          if [[ -n "${oldpid}" ]] && kill -0 "${oldpid}" 2>/dev/null; then
            continue
          fi
        fi
        run_model "${gpu}" "${tag}"
        sleep 3
        break
      fi
    done
  done

  sleep 60
done
