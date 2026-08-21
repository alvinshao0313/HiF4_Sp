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
  echo "No free GPUs"; exit 1
fi
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_a2}"
OUT="${OUT_DIR:-Inference_Paradigm_Conversion/results}"
SAMPLES="${SAMPLES_PER_FAMILY:-8}"
mkdir -p "$OUT/$RUN_ID/logs"
echo "A2 RUN_ID=$RUN_ID GPUS=${GPUS[*]} samples_per_family=$SAMPLES"

for ((i=0; i<NUM_GPUS; i++)); do
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "$PY" -m Inference_Paradigm_Conversion.run_analysis \
    --out-dir "$OUT" a2 --device cuda:0 --run-id "$RUN_ID" \
    --shard-id "$i" --num-shards "$NUM_GPUS" --samples-per-family "$SAMPLES" \
    >"$OUT/$RUN_ID/logs/shard_${i}.log" 2>&1 &
  echo "[A2] shard $i on GPU ${GPUS[$i]} pid $!"
done
wait

"$PY" - <<PY
import csv, json
from pathlib import Path
from collections import defaultdict
from Inference_Paradigm_Conversion.ipc_analysis.io_utils import atomic_write_json, write_csv

run = Path("$OUT") / "$RUN_ID"
rows=[]
for p in sorted(run.glob("a2_variants_shard*.csv")):
    rows.extend(csv.DictReader(p.open()))
write_csv(run/"a2_variants.csv", rows)

def mean(xs):
    return sum(xs)/len(xs) if xs else 0.0

# ranking by mean R_cf / R_cf_output, excluding boundary-hit oracle rows
by_fmt_var = defaultdict(lambda: {"rcf": [], "rcf_out": [], "nmse": []})
for r in rows:
    if r.get("exclude_from_main_rcf") in ("True", "true", True, "1"):
        continue
    key = (r["format"], r["variant"])
    if r.get("R_cf") not in (None, ""):
        by_fmt_var[key]["rcf"].append(float(r["R_cf"]))
    if r.get("R_cf_output") not in (None, ""):
        by_fmt_var[key]["rcf_out"].append(float(r["R_cf_output"]))
    if r.get("nmse") not in (None, ""):
        by_fmt_var[key]["nmse"].append(float(r["nmse"]))

ranking=[]
for (fmt, var), d in by_fmt_var.items():
    ranking.append({
        "format": fmt,
        "variant": var,
        "mean_R_cf": mean(d["rcf"]),
        "mean_R_cf_output": mean(d["rcf_out"]),
        "mean_nmse": mean(d["nmse"]),
        "n": len(d["rcf"]),
    })
ranking_act = sorted(ranking, key=lambda x: x["mean_R_cf"], reverse=True)
ranking_out = sorted(ranking, key=lambda x: x["mean_R_cf_output"], reverse=True)

# phase / projection breakdown for top interesting variants
phase = defaultdict(list)
proj = defaultdict(list)
for r in rows:
    if r.get("format")=="nvfp4" and r.get("variant")=="nv_oracle_global_scale":
        if r.get("exclude_from_main_rcf") in ("True","true",True,"1"):
            continue
        if r.get("R_cf") not in (None,""):
            phase[r["phase"]].append(float(r["R_cf"]))
            proj[r["projection"]].append(float(r["R_cf"]))

summary={
    "run_id":"$RUN_ID",
    "num_rows": len(rows),
    "ranking_by_mean_R_cf": ranking_act,
    "ranking_by_mean_R_cf_output": ranking_out,
    "nv_oracle_global_scale_by_phase": {k: mean(v) for k,v in phase.items()},
    "nv_oracle_global_scale_by_proj": {k: mean(v) for k,v in proj.items()},
    "note": "R_cf is recoverable activation recon error under idealization; not independent shares.",
}
atomic_write_json(run/"a2_summary.json", summary)
print("top R_cf:")
for r in ranking_act[:8]:
    print(r)
print("top R_cf_output:")
for r in ranking_out[:8]:
    print(r)
PY
echo "$RUN_ID" > "$OUT/latest_a2_run_id.txt"
echo "A2 DONE $RUN_ID"
