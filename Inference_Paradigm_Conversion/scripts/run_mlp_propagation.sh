#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/shaoyuantian/program/HiF4_Sp"
PY="/home/shaoyuantian/anaconda3/envs/hif4/bin/python"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 TORCH_NUM_THREADS=4

if [[ -n "${GPU_LIST:-}" ]]; then
  IFS=',' read -r -a GPUS <<< "$GPU_LIST"
else
  mapfile -t GPUS < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F',' '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($2+0 < 2048) print $1}')
fi
NUM_GPUS=${#GPUS[@]}
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_mlp}"
OUT="${OUT_DIR:-Inference_Paradigm_Conversion/results}"
mkdir -p "$OUT/$RUN_ID/logs"
SAMPLES="${SAMPLES_PER_FAMILY:-8}"

for ((i=0; i<NUM_GPUS; i++)); do
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "$PY" - <<PY >"$OUT/$RUN_ID/logs/shard_${i}.log" 2>&1 &
from pathlib import Path
from Inference_Paradigm_Conversion.ipc_analysis.analysis.mlp_repr_pipeline import run_mlp_repr_shard
from Inference_Paradigm_Conversion.ipc_analysis.config import load_experiment_config
cfg = load_experiment_config()
run_mlp_repr_shard(
    cfg.source_checkpoint_path(),
    Path("$OUT") / "$RUN_ID",
    device="cuda:0",
    shard_id=$i,
    num_shards=$NUM_GPUS,
    samples_per_family=$SAMPLES,
)
PY
  echo "[M] shard $i on GPU ${GPUS[$i]}"
done
wait
"$PY" - <<PY
import csv, json
from pathlib import Path
from Inference_Paradigm_Conversion.ipc_analysis.io_utils import atomic_write_json, write_csv
run = Path("$OUT") / "$RUN_ID"
rows=[]
for p in sorted(run.glob("mlp_propagation_shard*.csv")):
    rows.extend(csv.DictReader(p.open()))
write_csv(run/"mlp_propagation.csv", rows)
xs=[float(r["product_cross_share"]) for r in rows]
summary={"run_id":"$RUN_ID","num_rows":len(rows),"mean_product_cross_share": (sum(xs)/len(xs) if xs else 0.0)}
atomic_write_json(run/"mlp_summary.json", summary)
print(summary)
PY
echo "$RUN_ID" > "$OUT/latest_mlp_run_id.txt"
