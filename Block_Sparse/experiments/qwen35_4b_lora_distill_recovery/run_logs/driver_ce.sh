#!/usr/bin/env bash
# M2 (pure CE control arm) training, single GPU. No teacher -> plain CE SFT.
# Direct python call (not run_mlp_lora_sft.sh) because the shell wrapper's
# `TEACHER_MODEL_DIR="${TEACHER_MODEL_DIR:-Qwen/Qwen3.5-4B}"` treats empty string
# as unset and would force distillation. We omit --teacher_model_dir entirely.
#
# CE-only peak << distill: no teacher weights, no KL graph, no LAFD graph.
# Single GPU ~30GB @32k. Retries only on preemption OOM (other process >5GiB on
# the card); self-OOM or any non-OOM failure stops immediately.
set -o pipefail
# NOTE: no `set -u`: hif4's activate.d/gcc script references unbound SYS_SYSROOT.

source /home/shaoyuantian/anaconda3/etc/profile.d/conda.sh
conda activate hif4
cd /home/shaoyuantian/program/HiF4_Sp

export PRUNED_MODEL_DIR=Block_Sparse/outputs/qwen35_4b_fisher_budget_wanda_s0.20_b64x32_s1k_permwanda_shared_rpermnone
M2_OUT=Block_Sparse/outputs/qwen35_4b_rpermnone_mlplora16_ce500
MAX_ATT=10

pick_gpu1() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | sort -t, -k2 -nb | head -1 | cut -d, -f1 | tr -d ' '
}

other_big_process() {
  grep -oE "Process [0-9]+ has [0-9.]+ (MiB|GiB)" "$1" \
    | awk '{v=$4; if ($5=="MiB") v=v/1024; if (v>5) found=1} END {exit !found}'
}

attempt=1
while [ "${attempt}" -le "${MAX_ATT}" ]; do
  gpu=$(pick_gpu1)
  attlog=/tmp/ce500_attempt_${attempt}.log
  echo "=== ce500 attempt ${attempt} $(date '+%F %T') gpu=${gpu} ==="
  if CUDA_VISIBLE_DEVICES=${gpu} python Block_Sparse/tools/train_mlp_lora_sft.py \
       --pruned_model_dir "${PRUNED_MODEL_DIR}" \
       --output_dir "${M2_OUT}" \
       --lora_r 16 --lora_alpha 32 --lora_dropout 0.0 \
       --max_steps 500 --learning_rate 1e-4 \
       --gradient_accumulation_steps 8 \
       --model_max_length 32768 --allow_truncate false \
       --logit_chunk_size 512 --parallel_mode layer \
       > "${attlog}" 2>&1; then
    cat "${attlog}"
    echo "=== ce500 SUCCESS attempt ${attempt} ==="
    exit 0
  fi
  rc=$?
  cat "${attlog}"
  if grep -q "OutOfMemoryError" "${attlog}"; then
    if other_big_process "${attlog}"; then
      echo "=== ce500 attempt ${attempt} rc=${rc} preemption OOM, retry in 180s ==="
      sleep 180
    else
      echo "=== ce500 attempt ${attempt} rc=${rc} SELF OOM (systematic), stop ==="
      exit 3
    fi
  else
    echo "=== ce500 attempt ${attempt} rc=${rc} NON-OOM failure, stop ==="
    exit 1
  fi
  attempt=$((attempt + 1))
done
echo "=== ce500 exhausted ${MAX_ATT} attempts, stop ==="
exit 2
