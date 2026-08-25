#!/usr/bin/env bash
# Driver for qwen35_4b_lora_distill_recovery: smoke(20 steps) -> M3 qad500 training.
# Retries ONLY on CUDA OutOfMemoryError (external GPU preemption on shared node);
# any deterministic failure stops immediately (let-it-crash). KL mode deviation
# from plan: eakld -> eakld_topk k=128 (full-vocab EAKLD graph accumulates ~77GB
# on the loss GPU at 32k seq; top-k keeps graph on k dims. gamma unchanged).
set -o pipefail
# NOTE: no `set -u`: hif4's activate.d/gcc script references unbound SYS_SYSROOT.

source /home/shaoyuantian/anaconda3/etc/profile.d/conda.sh
conda activate hif4
cd /home/shaoyuantian/program/HiF4_Sp

LOGD=Block_Sparse/experiments/qwen35_4b_lora_distill_recovery/run_logs
export PRUNED_MODEL_DIR=Block_Sparse/outputs/qwen35_4b_fisher_budget_wanda_s0.20_b64x32_s1k_permwanda_shared_rpermnone
export TEACHER_MODEL_DIR=Qwen/Qwen3.5-4B
export KL_MODE=eakld_topk
export KL_TOPK=128

M3_OUT=Block_Sparse/outputs/qwen35_4b_rpermnone_mlplora16_qad500
SMOKE_OUT=/tmp/qwen35_4b_lora_smoke

# Single-GPU mode (user decision 2026-07-31 16:48): eakld_topk peak ~60GB < 80GB,
# frees the other cards on this shared node. Picks the single least-used GPU.
pick_gpus3() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | sort -t, -k2 -nb | head -1 | cut -d, -f1 | tr -d ' '
}

# $1=output_dir $2=max_steps $3=max_attempts $4=retry_sleep_s $5=tag
run_arm() {
  local out=$1 steps=$2 max_att=$3 sleep_s=$4 tag=$5
  local attempt=1 gpus attlog rc
  while [ "${attempt}" -le "${max_att}" ]; do
    gpus=$(pick_gpus3)
    attlog=/tmp/${tag}_attempt_${attempt}.log
    echo "=== ${tag} attempt ${attempt} $(date '+%F %T') gpus=${gpus} ==="
    if OUTPUT_DIR=${out} MAX_STEPS=${steps} CUDA_VISIBLE_DEVICES=${gpus} \
       bash Block_Sparse/scripts/run_mlp_lora_sft.sh > "${attlog}" 2>&1; then
      cat "${attlog}"
      echo "=== ${tag} SUCCESS attempt ${attempt} ==="
      return 0
    fi
    rc=$?
    cat "${attlog}"
    if grep -q "OutOfMemoryError" "${attlog}"; then
      echo "=== ${tag} attempt ${attempt} rc=${rc} OOM, retry in ${sleep_s}s ==="
      sleep "${sleep_s}"
    else
      echo "=== ${tag} attempt ${attempt} rc=${rc} NON-OOM failure, stop ==="
      return 1
    fi
    attempt=$((attempt + 1))
  done
  echo "=== ${tag} exhausted ${max_att} attempts, stop ==="
  return 2
}

{
  echo "driver start $(date '+%F %T') pid=$$"
  if run_arm "${SMOKE_OUT}" 20 8 120 smoke20 2>&1 | tee -a "${LOGD}/smoke_qad20.log"; then
    rm -rf "${SMOKE_OUT}"
    run_arm "${M3_OUT}" 500 20 180 qad500 2>&1 | tee -a "${LOGD}/train_qad500.log"
  else
    echo "smoke failed; M3 not started"
  fi
  echo "driver end $(date '+%F %T')"
} >> "${LOGD}/driver_qad.log" 2>&1
