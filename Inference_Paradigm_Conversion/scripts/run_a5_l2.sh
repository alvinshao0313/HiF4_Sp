#!/usr/bin/env bash
# Sequential A5 then L2 on free GPUs (default: auto-pick <2GiB used).
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
OUT="${OUT_DIR:-Inference_Paradigm_Conversion/results}"
SAMPLES="${SAMPLES_PER_FAMILY:-4}"

# ---- A5 ----
A5_ID="${A5_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_a5}"
mkdir -p "$OUT/$A5_ID/logs"
echo "A5 RUN_ID=$A5_ID GPUS=${GPUS[*]}"
for ((i=0; i<NUM_GPUS; i++)); do
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "$PY" -m Inference_Paradigm_Conversion.run_analysis \
    --out-dir "$OUT" a5 --device cuda:0 --run-id "$A5_ID" \
    --shard-id "$i" --num-shards "$NUM_GPUS" --samples-per-family "$SAMPLES" \
    --max-groups-per-module "${MAX_GROUPS:-64}" \
    >"$OUT/$A5_ID/logs/shard_${i}.log" 2>&1 &
  echo "[A5] shard $i on GPU ${GPUS[$i]} pid $!"
done
wait
"$PY" - <<PY
import csv, json
from pathlib import Path
from collections import defaultdict
from Inference_Paradigm_Conversion.ipc_analysis.io_utils import atomic_write_json, write_csv
run=Path("$OUT")/"$A5_ID"
rows=[]
for p in sorted(run.glob("a5_interventions_shard*.csv")):
    rows.extend(csv.DictReader(p.open()))
if not rows:
    raise SystemExit("A5 FAILED: zero rows")
write_csv(run/"a5_interventions.csv", rows)
by=defaultdict(list)
for r in rows:
    key=(r.get("intervention"), r.get("setting"))
    if r.get("nmse_h_vs_n") not in (None,""):
        by[key].append(float(r["nmse_h_vs_n"]))
def mean(xs): return sum(xs)/len(xs) if xs else 0.0
ranking=sorted(
    ({"intervention":k[0],"setting":k[1],"mean_nmse_h_vs_n":mean(v),"n":len(v)} for k,v in by.items()),
    key=lambda d:d["mean_nmse_h_vs_n"], reverse=True,
)
base=[r for r in ranking if r["intervention"]=="baseline"]
base_m=base[0]["mean_nmse_h_vs_n"] if base else 0.0
summary={"run_id":"$A5_ID","num_rows":len(rows),"baseline_mean_nmse_h_vs_n":base_m,"ranking":ranking[:40]}
atomic_write_json(run/"a5_summary.json", summary)
print(json.dumps({"num_rows":len(rows),"baseline":base_m,"top5":ranking[:5]}, indent=2))
PY
echo "$A5_ID" > "$OUT/latest_a5_run_id.txt"
echo "A5 DONE $A5_ID"

# ---- L2 ----
L2_ID="${L2_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_l2}"
mkdir -p "$OUT/$L2_ID/logs"
echo "L2 RUN_ID=$L2_ID GPUS=${GPUS[*]}"
for ((i=0; i<NUM_GPUS; i++)); do
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "$PY" -m Inference_Paradigm_Conversion.run_analysis \
    --out-dir "$OUT" l2 --device cuda:0 --run-id "$L2_ID" \
    --shard-id "$i" --num-shards "$NUM_GPUS" --samples-per-family "$SAMPLES" \
    >"$OUT/$L2_ID/logs/shard_${i}.log" 2>&1 &
  echo "[L2] shard $i on GPU ${GPUS[$i]} pid $!"
done
wait
"$PY" - <<PY
import csv, json
from pathlib import Path
from Inference_Paradigm_Conversion.ipc_analysis.io_utils import atomic_write_json, write_csv
run=Path("$OUT")/"$L2_ID"
rows=[]
for p in sorted(run.glob("l2_shapley_shard*.csv")):
    rows.extend(csv.DictReader(p.open()))
if not rows:
    raise SystemExit("L2 FAILED: zero rows")
write_csv(run/"l2_shapley.csv", rows)
def mean(k):
    xs=[float(r[k]) for r in rows if r.get(k) not in (None,"")]
    return sum(xs)/len(xs) if xs else 0.0
aud_ok=sum(1 for r in rows if str(r.get("fp64_audit_ok")).lower() in ("true","1"))
aud_fail=sum(1 for r in rows if str(r.get("fp64_audit_ok")).lower() in ("false","0"))
ew,ea=mean("energy_phi_w"),mean("energy_phi_a")
summary={
    "run_id":"$L2_ID","num_rows":len(rows),
    "mean_energy_phi_w":ew,"mean_energy_phi_a":ea,
    "ratio_phi_a_over_phi_w": (ea/ew if ew>0 else 0.0),
    "mean_shapley_residual_rel":mean("shapley_residual_rel"),
    "fp64_audit_ok":aud_ok,"fp64_audit_fail":aud_fail,
}
atomic_write_json(run/"l2_summary.json", summary)
print(json.dumps(summary, indent=2))
PY
echo "$L2_ID" > "$OUT/latest_l2_run_id.txt"
echo "L2 DONE $L2_ID"
