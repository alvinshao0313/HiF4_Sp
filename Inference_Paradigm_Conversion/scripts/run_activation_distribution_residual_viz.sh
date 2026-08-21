#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/shaoyuantian/program/HiF4_Sp"
PY="/home/shaoyuantian/anaconda3/envs/hif4/bin/python"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 TORCH_NUM_THREADS=4 MKL_NUM_THREADS=4

CONFIG="${CONFIG:-Inference_Paradigm_Conversion/configs/qwen3_8b_nvfp4_qat_formal.yaml}"
OUT="${OUT_DIR:-Inference_Paradigm_Conversion/results}"
SPLIT="${SPLIT:-discovery}"
SAMPLES="${SAMPLES_PER_FAMILY:-8}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-256}"
DECODE_STEPS="${DECODE_STEPS:-8}"
MAX_POINTS="${MAX_POINT_SAMPLES_PER_CAPTURE:-1024}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_activation_viz_${SPLIT}}"

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
echo "ACTIVATION-VIZ RUN_ID=$RUN_ID GPUS=${GPUS[*]} SPLIT=$SPLIT SAMPLES=$SAMPLES"

for ((i=0; i<NUM_GPUS; i++)); do
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "$PY" -m Inference_Paradigm_Conversion.run_analysis \
    --config "$CONFIG" --out-dir "$OUT" activation-viz \
    --device cuda:0 --run-id "$RUN_ID" \
    --shard-id "$i" --num-shards "$NUM_GPUS" \
    --split "$SPLIT" \
    --samples-per-family "$SAMPLES" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --decode-steps "$DECODE_STEPS" \
    --max-point-samples-per-capture "$MAX_POINTS" \
    >"$OUT/$RUN_ID/logs/shard_${i}.log" 2>&1 &
  echo "[activation-viz] shard $i on GPU ${GPUS[$i]} pid $!"
done
wait

"$PY" -m Inference_Paradigm_Conversion.run_analysis \
  --out-dir "$OUT" activation-viz-merge --run-id "$RUN_ID"

"$PY" -m Inference_Paradigm_Conversion.run_analysis \
  --out-dir "$OUT" activation-viz-report --run-id "$RUN_ID"

echo "$RUN_ID" > "$OUT/latest_activation_viz_run_id.txt"
echo "ACTIVATION-VIZ DONE $RUN_ID"
