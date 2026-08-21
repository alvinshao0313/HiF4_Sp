#!/usr/bin/env bash
# Semantic E2E (ARC via lm_eval). Runtime E2E marked unsupported unless kernels exist.
set -euo pipefail
ROOT="/home/shaoyuantian/program/HiF4_Sp"
PY="/home/shaoyuantian/anaconda3/envs/hif4/bin/python"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 TORCH_NUM_THREADS=4 MKL_NUM_THREADS=4

OUT="${OUT_DIR:-Inference_Paradigm_Conversion/results}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_e2e}"
PATH_ID="${PATH_ID:-P1_semantic}"
TASKS="${TASKS:-arc_easy,arc_challenge}"
BATCH="${BATCH_SIZE:-4}"
mkdir -p "$OUT/$RUN_ID/logs"

# pick two free GPUs if possible
if [[ -n "${GPU_LIST:-}" ]]; then
  IFS=',' read -r -a GPUS <<< "$GPU_LIST"
else
  mapfile -t GPUS < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F',' '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($2+0 < 2048) print $1}')
fi
if [[ ${#GPUS[@]} -lt 1 ]]; then
  echo "No free GPUs"; exit 1
fi
G0=${GPUS[0]}
G1=${GPUS[1]:-${GPUS[0]}}
echo "E2E RUN_ID=$RUN_ID PATH_ID=$PATH_ID GPUS=$G0,$G1"

# source + target in parallel if two GPUs
CUDA_VISIBLE_DEVICES=$G0 "$PY" -m Inference_Paradigm_Conversion.ipc_analysis.eval.semantic_e2e \
  --path-id "$PATH_ID" --role source --device cuda:0 --batch-size "$BATCH" --tasks "$TASKS" \
  --out-dir "$OUT/$RUN_ID" >"$OUT/$RUN_ID/logs/source.log" 2>&1 &
PID0=$!
if [[ "$G1" != "$G0" ]]; then
  CUDA_VISIBLE_DEVICES=$G1 "$PY" -m Inference_Paradigm_Conversion.ipc_analysis.eval.semantic_e2e \
    --path-id "$PATH_ID" --role target --device cuda:0 --batch-size "$BATCH" --tasks "$TASKS" \
    --out-dir "$OUT/$RUN_ID" >"$OUT/$RUN_ID/logs/target.log" 2>&1 &
  PID1=$!
  wait $PID0 $PID1
else
  wait $PID0
  CUDA_VISIBLE_DEVICES=$G0 "$PY" -m Inference_Paradigm_Conversion.ipc_analysis.eval.semantic_e2e \
    --path-id "$PATH_ID" --role target --device cuda:0 --batch-size "$BATCH" --tasks "$TASKS" \
    --out-dir "$OUT/$RUN_ID" >"$OUT/$RUN_ID/logs/target.log" 2>&1
fi

"$PY" - <<PY
import json, csv
from pathlib import Path
from Inference_Paradigm_Conversion.ipc_analysis.io_utils import atomic_write_json, write_csv, write_text
run = Path("$OUT") / "$RUN_ID"
src = json.loads((run / f"semantic_${PATH_ID}_source.json").read_text())
tgt = json.loads((run / f"semantic_${PATH_ID}_target.json").read_text())
rows=[]
for task in sorted(set(src["scores"]) | set(tgt["scores"])):
    s=src["scores"].get(task); t=tgt["scores"].get(task)
    rows.append({
        "e2e_kind":"semantic_e2e",
        "path_id":"$PATH_ID",
        "task":task,
        "source_score":s,
        "target_score":t,
        "delta_target_minus_source": (None if s is None or t is None else t-s),
        "runtime_e2e":"unsupported_by_hardware_or_not_run",
    })
write_csv(run/"e2e_summary.csv", rows)
summary={"run_id":"$RUN_ID","path_id":"$PATH_ID","rows":rows,
         "predicted_sensitive":["gate_proj","single_layer","prefix_heavy"],
         "note":"Do not back-propagate e2e into root-cause ranking."}
atomic_write_json(run/"e2e_summary.json", summary)
write_text(Path("$OUT")/"latest_e2e_run_id.txt", "$RUN_ID")
print(json.dumps(summary, indent=2))
PY
echo "E2E DONE $RUN_ID"
