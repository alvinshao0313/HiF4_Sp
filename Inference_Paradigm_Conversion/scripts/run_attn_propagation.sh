#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/shaoyuantian/program/HiF4_Sp"
PY="/home/shaoyuantian/anaconda3/envs/hif4/bin/python"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 TORCH_NUM_THREADS=4 MKL_NUM_THREADS=4

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
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_attn}"
OUT="${OUT_DIR:-Inference_Paradigm_Conversion/results}"
SAMPLES="${SAMPLES_PER_FAMILY:-8}"
mkdir -p "$OUT/$RUN_ID/logs"
echo "ATTN RUN_ID=$RUN_ID GPUS=${GPUS[*]} samples_per_family=$SAMPLES"

for ((i=0; i<NUM_GPUS; i++)); do
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "$PY" -m Inference_Paradigm_Conversion.run_analysis \
    --out-dir "$OUT" attn --device cuda:0 --run-id "$RUN_ID" \
    --shard-id "$i" --num-shards "$NUM_GPUS" --samples-per-family "$SAMPLES" \
    >"$OUT/$RUN_ID/logs/shard_${i}.log" 2>&1 &
  echo "[T] shard $i on GPU ${GPUS[$i]} pid $!"
done
wait

"$PY" - <<PY
import csv, json
from pathlib import Path
from Inference_Paradigm_Conversion.ipc_analysis.io_utils import atomic_write_json, write_csv
run = Path("$OUT") / "$RUN_ID"
rows=[]
for p in sorted(run.glob("attention_propagation_shard*.csv")):
    rows.extend(csv.DictReader(p.open()))
write_csv(run/"attention_propagation.csv", rows)
def mean(k):
    xs=[float(r[k]) for r in rows if r.get(k) not in (None,"")]
    return sum(xs)/len(xs) if xs else 0.0
summary={
    "run_id":"$RUN_ID",
    "num_rows":len(rows),
    "mean_kl_st": mean("kl_st"),
    "mean_js": mean("js"),
    "mean_flip": mean("top_attended_flip_rate"),
    "mean_logits_gain": mean("logits_gain"),
    "mean_nmse_logits": mean("nmse_logits"),
    "mean_nmse_residual": mean("nmse_residual"),
    "linear_attn_present": False,
}
for p in sorted(run.glob("attention_summary_shard*.json")):
    s=json.loads(p.read_text())
    if s.get("linear_attn_present"):
        summary["linear_attn_present"]=True
        summary["linear_attn_modules"]=s.get("linear_attn_modules")
if len(rows)==0:
    raise SystemExit("ATTN FAILED: zero rows after merge; check shard logs")
atomic_write_json(run/"attention_summary.json", summary)
print(summary)
PY
echo "$RUN_ID" > "$OUT/latest_attn_run_id.txt"
echo "ATTN DONE $RUN_ID"
