#!/usr/bin/env bash
# Representative-layer activation + linear decomposition across selected GPUs.
set -euo pipefail
ROOT="/home/shaoyuantian/program/HiF4_Sp"
PY="/home/shaoyuantian/anaconda3/envs/hif4/bin/python"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Prefer free GPUs: override with GPU_LIST="0,1,6,7"
if [[ -n "${GPU_LIST:-}" ]]; then
  IFS=',' read -r -a GPUS <<< "$GPU_LIST"
else
  # Auto-pick GPUs with < 2GiB used
  mapfile -t GPUS < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F',' '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($2+0 < 2048) print $1}')
fi
NUM_GPUS="${#GPUS[@]}"
if [[ "$NUM_GPUS" -lt 1 ]]; then
  echo "[AL] no free GPUs found" >&2
  exit 1
fi

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_repr_al}"
OUT_DIR="${OUT_DIR:-Inference_Paradigm_Conversion/results}"
SAMPLES_PER_FAMILY="${SAMPLES_PER_FAMILY:-32}"
LOG_DIR="$OUT_DIR/$RUN_ID/logs"
mkdir -p "$LOG_DIR"

echo "[AL] run_id=$RUN_ID gpus=${GPUS[*]} samples_per_family=$SAMPLES_PER_FAMILY"
# Avoid CPU oversubscription when launching many shards.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-4}"
PIDS=()
for ((i=0; i<NUM_GPUS; i++)); do
  gpu="${GPUS[$i]}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m Inference_Paradigm_Conversion.run_analysis \
    --config Inference_Paradigm_Conversion/configs/qwen3_8b_nvfp4_qat_formal.yaml \
    --out-dir "$OUT_DIR" \
    repr-al \
    --device cuda:0 \
    --run-id "$RUN_ID" \
    --shard-id "$i" \
    --num-shards "$NUM_GPUS" \
    --samples-per-family "$SAMPLES_PER_FAMILY" \
    --max-seq-len 256 \
    --decode-steps 8 \
    >"$LOG_DIR/shard_${i}.log" 2>&1 &
  PIDS+=("$!")
  echo "[AL] launched shard $i pid=${PIDS[$i]} on GPU $gpu"
done

FAIL=0
for i in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$i]}"; then
    echo "[AL] shard $i FAILED — see $LOG_DIR/shard_${i}.log"
    FAIL=1
  else
    echo "[AL] shard $i OK"
  fi
done

"$PY" - <<PY
import csv
from pathlib import Path
from Inference_Paradigm_Conversion.ipc_analysis.io_utils import atomic_write_json, write_csv

run = Path("$OUT_DIR") / "$RUN_ID"
act, lin = [], []
for p in sorted(run.glob("activation_summary_shard*.csv")):
    with p.open() as f:
        act.extend(csv.DictReader(f))
for p in sorted(run.glob("linear_decomp_shard*.csv")):
    with p.open() as f:
        lin.extend(csv.DictReader(f))
write_csv(run / "activation_summary.csv", act)
write_csv(run / "linear_decomp.csv", lin)
def mean(key, rows):
    xs=[float(r[key]) for r in rows if r.get(key) not in (None,'')]
    return sum(xs)/len(xs) if xs else 0.0
summary={
    "run_id": "$RUN_ID",
    "num_activation_rows": len(act),
    "num_linear_rows": len(lin),
    "mean_nmse_hif4_vs_nvfp4": mean("nmse_hif4_vs_nvfp4", act),
    "mean_nmse_nvfp4": mean("nmse_nvfp4", act),
    "mean_nmse_hif4": mean("nmse_hif4", act),
    "mean_nmse_mxfp8": mean("nmse_mxfp8", act),
}
atomic_write_json(run / "repr_al_summary.json", summary)
print("[AL] merged", summary)
PY

echo "$RUN_ID" > "$OUT_DIR/latest_repr_al_run_id.txt"
if [[ "$FAIL" -ne 0 ]]; then
  exit 1
fi
echo "[AL] ALL SHARDS PASSED → $OUT_DIR/$RUN_ID"
