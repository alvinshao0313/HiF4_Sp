#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/shaoyuantian/program/HiF4_Sp"
PY="/home/shaoyuantian/anaconda3/envs/hif4/bin/python"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 TORCH_NUM_THREADS=4 MKL_NUM_THREADS=4

MODE="${MODE:-n1_n2}"  # n1_n2 | prefix_suffix | oracle
if [[ -n "${GPU_LIST:-}" ]]; then
  IFS=',' read -r -a GPUS <<< "$GPU_LIST"
else
  mapfile -t GPUS < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F',' '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($2+0 < 2048) print $1}')
fi
NUM_GPUS=${#GPUS[@]}
if [[ "$NUM_GPUS" -lt 1 ]]; then
  echo "No free GPUs (<2GiB used)"; exit 1
fi
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_inject_${MODE}}"
OUT="${OUT_DIR:-Inference_Paradigm_Conversion/results}"
mkdir -p "$OUT/$RUN_ID/logs"
echo "INJECT MODE=$MODE RUN_ID=$RUN_ID GPUS=${GPUS[*]}"

for ((i=0; i<NUM_GPUS; i++)); do
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "$PY" -m Inference_Paradigm_Conversion.run_analysis \
    --out-dir "$OUT" inject --device cuda:0 --run-id "$RUN_ID" \
    --shard-id "$i" --num-shards "$NUM_GPUS" --mode "$MODE" \
    >"$OUT/$RUN_ID/logs/shard_${i}.log" 2>&1 &
  echo "[N] shard $i on GPU ${GPUS[$i]} pid $!"
done
wait

"$PY" - <<PY
import csv, json
from pathlib import Path
from collections import defaultdict
from Inference_Paradigm_Conversion.ipc_analysis.io_utils import atomic_write_json, write_csv
run = Path("$OUT") / "$RUN_ID"
mode = "$MODE"
rows=[]
for p in sorted(run.glob(f"injection_{mode}_shard*.csv")):
    rows.extend(csv.DictReader(p.open()))
write_csv(run/f"injection_{mode}.csv", rows)
summary={"run_id":"$RUN_ID","mode":mode,"num_rows":len(rows)}
if mode=="n1_n2" and rows:
    by=defaultdict(list)
    for r in rows:
        key=(r.get("mask_kind"), r.get("projection") or "layer")
        if r.get("kl_last") not in (None,""):
            by[key].append(float(r["kl_last"]))
    summary["mean_kl_by_mask"]={
        f"{k[0]}:{k[1]}": sum(v)/len(v) for k,v in by.items()
    }
    summary["mean_kl_last"]=sum(float(r["kl_last"]) for r in rows)/len(rows)
elif mode=="oracle" and rows:
    by_frac=defaultdict(list)
    by_rand=defaultdict(list)
    for r in rows:
        f=r.get("top_frac")
        if r.get("recoverable_kl") not in (None,""):
            by_frac[f].append(float(r["recoverable_kl"]))
        if r.get("random_recoverable_kl") not in (None,""):
            by_rand[f].append(float(r["random_recoverable_kl"]))
    summary["mean_recoverable_kl_by_frac"]={k:sum(v)/len(v) for k,v in by_frac.items()}
    summary["mean_random_recoverable_kl_by_frac"]={k:sum(v)/len(v) for k,v in by_rand.items()}
elif mode=="prefix_suffix" and rows:
    by=defaultdict(list)
    for r in rows:
        key=(r.get("mask_kind"), r.get("prefix_k"))
        if r.get("kl_last") not in (None,""):
            by[key].append(float(r["kl_last"]))
    summary["mean_kl_by_boundary"]={
        f"{k[0]}:k={k[1]}": sum(v)/len(v) for k,v in by.items()
    }
atomic_write_json(run/f"injection_{mode}_summary.json", summary)
print(json.dumps(summary, indent=2)[:1200])
PY
echo "$RUN_ID" > "$OUT/latest_inject_${MODE}_run_id.txt"
echo "INJECT DONE $RUN_ID"
