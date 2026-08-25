#!/usr/bin/env bash
# v2 (2026-08-03): M3 qad500 training, 2-GPU layer parallel (user pinned GPUs 0,1).
# Hot-card peak estimate ~51GB/80GB @32k (weights 8.3 + CE graph 32.6 + rest ~10).
#
# Retry policy (fixes v1's 20x wall-banging, ~17h wasted):
#   - OOM where ANOTHER process holds >5GiB on the card  -> preemption: retry,
#     re-pick cards (prefer 0,1; fall back to 2 least-used).
#   - OOM with no other big process                     -> self-OOM (systematic):
#     STOP immediately, no retry.
#   - non-OOM failure                                   -> STOP (let it crash).
set -o pipefail
# NOTE: no `set -u`: hif4's activate.d/gcc script references unbound SYS_SYSROOT.

source /home/shaoyuantian/anaconda3/etc/profile.d/conda.sh
conda activate hif4
cd /home/shaoyuantian/program/HiF4_Sp

export PRUNED_MODEL_DIR=Block_Sparse/outputs/qwen35_4b_fisher_budget_wanda_s0.20_b64x32_s1k_permwanda_shared_rpermnone
export TEACHER_MODEL_DIR=Qwen/Qwen3.5-4B
export KL_MODE=eakld_topk
export KL_TOPK=128
M3_OUT=Block_Sparse/outputs/qwen35_4b_rpermnone_mlplora16_qad500
MAX_ATT=10

pick_gpus2() {
  local m0 m1
  m0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
  m1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
  if [ "${m0}" -lt 5000 ] && [ "${m1}" -lt 5000 ]; then
    echo "0,1"
    return
  fi
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | sort -t, -k2 -nb | head -2 | cut -d, -f1 | sort -n | tr '\n' ',' | sed 's/,$//'
}

# $1=attempt log; exit 0 if another process holds >5GiB (preemption), else 1
other_big_process() {
  grep -oE "Process [0-9]+ has [0-9.]+ (MiB|GiB)" "$1" \
    | awk '{v=$4; if ($5=="MiB") v=v/1024; if (v>5) found=1} END {exit !found}'
}

attempt=1
while [ "${attempt}" -le "${MAX_ATT}" ]; do
  gpus=$(pick_gpus2)
  attlog=/tmp/qad500v2_attempt_${attempt}.log
  echo "=== qad500v2 attempt ${attempt} $(date '+%F %T') gpus=${gpus} ==="
  if OUTPUT_DIR=${M3_OUT} MAX_STEPS=500 CUDA_VISIBLE_DEVICES=${gpus} \
     bash Block_Sparse/scripts/run_mlp_lora_sft.sh > "${attlog}" 2>&1; then
    cat "${attlog}"
    echo "=== qad500v2 SUCCESS attempt ${attempt} ==="
    exit 0
  fi
  rc=$?
  cat "${attlog}"
  if grep -q "OutOfMemoryError" "${attlog}"; then
    if other_big_process "${attlog}"; then
      echo "=== qad500v2 attempt ${attempt} rc=${rc} preemption OOM, retry in 180s ==="
      sleep 180
    else
      echo "=== qad500v2 attempt ${attempt} rc=${rc} SELF OOM (systematic), stop ==="
      exit 3
    fi
  else
    echo "=== qad500v2 attempt ${attempt} rc=${rc} NON-OOM failure, stop ==="
    exit 1
  fi
  attempt=$((attempt + 1))
done
echo "=== qad500v2 exhausted ${MAX_ATT} attempts, stop ==="
exit 2
