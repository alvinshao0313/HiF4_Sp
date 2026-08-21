#!/usr/bin/env bash
# Launch W0–W2 full-model weight analysis across all visible GPUs in parallel.
set -euo pipefail
ROOT="/home/shaoyuantian/program/HiF4_Sp"
PY="/home/shaoyuantian/anaconda3/envs/hif4/bin/python"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_weight_full}"
OUT_DIR="${OUT_DIR:-Inference_Paradigm_Conversion/results}"
LOG_DIR="$OUT_DIR/$RUN_ID/logs"
mkdir -p "$LOG_DIR"

echo "[W] run_id=$RUN_ID num_gpus=$NUM_GPUS"
PIDS=()
for ((i=0; i<NUM_GPUS; i++)); do
  CUDA_VISIBLE_DEVICES="$i" "$PY" -m Inference_Paradigm_Conversion.run_analysis \
    --config Inference_Paradigm_Conversion/configs/qwen3_8b_nvfp4_qat_formal.yaml \
    --out-dir "$OUT_DIR" \
    weight \
    --device cuda:0 \
    --run-id "$RUN_ID" \
    --shard-id "$i" \
    --num-shards "$NUM_GPUS" \
    --max-groups-per-tensor 4096 \
    >"$LOG_DIR/shard_${i}.log" 2>&1 &
  PIDS+=("$!")
  echo "[W] launched shard $i pid=${PIDS[$i]} on physical GPU $i"
done

FAIL=0
for i in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$i]}"; then
    echo "[W] shard $i FAILED — see $LOG_DIR/shard_${i}.log"
    FAIL=1
  else
    echo "[W] shard $i OK"
  fi
done

"$PY" - <<PY
import json
from pathlib import Path
from Inference_Paradigm_Conversion.ipc_analysis.io_utils import atomic_write_json, write_csv, read_json

run = Path("$OUT_DIR") / "$RUN_ID"
rows = []
ref_e = err_e = numel = 0.0
summaries = []
for p in sorted(run.glob("weight_tensor_summary_shard*.csv")):
    import csv
    with p.open() as f:
        rows.extend(csv.DictReader(f))
for p in sorted(run.glob("weight_summary_shard*.json")):
    s = read_json(p)
    summaries.append(s)
    g = s["global_format_metrics"]
    ref_e += g["reference_energy"]
    err_e += g["error_energy"]
    numel += g["numel"]
merged = {
    "run_id": "$RUN_ID",
    "num_shards": len(summaries),
    "num_tensors": len(rows),
    "global_nmse": (err_e / ref_e) if ref_e > 0 else 0.0,
    "global_reference_energy": ref_e,
    "global_error_energy": err_e,
    "global_numel": numel,
    "evidence_class": "observational_correlation",
}
write_csv(run / "weight_tensor_summary.csv", rows)
atomic_write_json(run / "weight_summary.json", merged)
print("[W] merged", run / "weight_summary.json", "global_nmse=", merged["global_nmse"])
PY

echo "$RUN_ID" > "$OUT_DIR/latest_weight_run_id.txt"
if [[ "$FAIL" -ne 0 ]]; then
  echo "[W] completed with failures"
  exit 1
fi
echo "[W] ALL SHARDS PASSED → $OUT_DIR/$RUN_ID"
