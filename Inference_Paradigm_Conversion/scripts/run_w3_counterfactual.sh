#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/shaoyuantian/program/HiF4_Sp"
PY="/home/shaoyuantian/anaconda3/envs/hif4/bin/python"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_w3}"
OUT_DIR="${OUT_DIR:-Inference_Paradigm_Conversion/results}"
LOG_DIR="$OUT_DIR/$RUN_ID/logs"
mkdir -p "$LOG_DIR"

PIDS=()
for ((i=0; i<NUM_GPUS; i++)); do
  CUDA_VISIBLE_DEVICES="$i" "$PY" -m Inference_Paradigm_Conversion.run_analysis \
    --config Inference_Paradigm_Conversion/configs/qwen3_8b_nvfp4_qat_formal.yaml \
    --out-dir "$OUT_DIR" \
    w3 --device cuda:0 --run-id "$RUN_ID" --shard-id "$i" --num-shards "$NUM_GPUS" \
    >"$LOG_DIR/shard_${i}.log" 2>&1 &
  PIDS+=("$!")
done

FAIL=0
for i in "${!PIDS[@]}"; do
  wait "${PIDS[$i]}" || FAIL=1
done

"$PY" - <<PY
import csv
from pathlib import Path
from collections import defaultdict
from Inference_Paradigm_Conversion.ipc_analysis.io_utils import atomic_write_json, write_csv
run = Path("$OUT_DIR") / "$RUN_ID"
rows=[]
for p in sorted(run.glob("w3_variants_shard*.csv")):
    rows.extend(csv.DictReader(p.open()))
write_csv(run/"w3_variants.csv", rows)
by_v=defaultdict(list)
for r in rows:
    if r.get("R_cf_output"):
        by_v[r["variant"]].append(float(r["R_cf_output"]))
ranking=sorted(({"variant":v,"mean_R_cf_output":sum(xs)/len(xs),"n":len(xs)} for v,xs in by_v.items()), key=lambda d:d["mean_R_cf_output"], reverse=True)
atomic_write_json(run/"w3_summary.json", {"run_id":"$RUN_ID","ranking_by_mean_R_cf_output":ranking,"num_rows":len(rows)})
print(ranking[:8])
PY
echo "$RUN_ID" > "$OUT_DIR/latest_w3_run_id.txt"
exit "$FAIL"
