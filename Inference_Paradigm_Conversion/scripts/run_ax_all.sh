#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/shaoyuantian/program/HiF4_Sp"
PY="/home/shaoyuantian/anaconda3/envs/hif4/bin/python"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 TORCH_NUM_THREADS=4 MKL_NUM_THREADS=4

CONFIG="${CONFIG:-Inference_Paradigm_Conversion/configs/qwen3_8b_activation_incremental.yaml}"
OUT="${OUT_DIR:-Inference_Paradigm_Conversion/results}"
SPLIT="${SPLIT:-discovery}"
PHASES="${PHASES:-prefill}"
SAMPLES="${SAMPLES_PER_FAMILY:-8}"
A2_DIR="${A2_RUN_DIR:-Inference_Paradigm_Conversion/results/20260811T032247Z_a2}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_ax}"

SKIP_FLAGS=()
[[ "${SKIP_AX1:-0}" == "1" ]] && SKIP_FLAGS+=(--skip-ax1)
[[ "${SKIP_AX2:-0}" == "1" ]] && SKIP_FLAGS+=(--skip-ax2)
[[ "${SKIP_AX3:-0}" == "1" ]] && SKIP_FLAGS+=(--skip-ax3)
[[ "${SKIP_AX4:-0}" == "1" ]] && SKIP_FLAGS+=(--skip-ax4)

if [[ -n "${GPU_LIST:-}" ]]; then
  IFS=',' read -r -a GPUS <<< "$GPU_LIST"
else
  mapfile -t GPUS < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F',' '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($2+0 < 2048) print $1}')
fi
NUM_GPUS=${#GPUS[@]}
if [[ "$NUM_GPUS" -lt 1 ]]; then
  echo "No free GPUs (<2GB used). Set GPU_LIST or free a GPU."; exit 1
fi

mkdir -p "$OUT/$RUN_ID/logs"
echo "AX RUN_ID=$RUN_ID GPUS=${GPUS[*]} SPLIT=$SPLIT PHASES=$PHASES"

for ((i=0; i<NUM_GPUS; i++)); do
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "$PY" -m Inference_Paradigm_Conversion.run_analysis \
    --config "$CONFIG" --out-dir "$OUT" ax \
    --device cuda:0 --run-id "$RUN_ID" \
    --shard-id "$i" --num-shards "$NUM_GPUS" \
    --split "$SPLIT" --phases "$PHASES" \
    --samples-per-family "$SAMPLES" \
    --a2-run-dir "$A2_DIR" \
    "${SKIP_FLAGS[@]}" \
    >"$OUT/$RUN_ID/logs/shard_${i}.log" 2>&1 &
  echo "[AX] shard $i on GPU ${GPUS[$i]} pid $!"
done
wait

"$PY" -m Inference_Paradigm_Conversion.run_analysis \
  --out-dir "$OUT" ax-merge \
  --run-id "$RUN_ID" --a2-run-dir "$A2_DIR"
echo "$RUN_ID" > "$OUT/latest_ax_run_id.txt"
echo "AX DONE $RUN_ID"
