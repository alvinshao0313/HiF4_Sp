#!/usr/bin/env bash
# P2_matched_semantic MMLU-Pro via main.py (vLLM + lighteval).
# P1_semantic is unsupported: vLLM fake_act_quant has no MXFP8.
set -euo pipefail
ROOT="/home/shaoyuantian/program/HiF4_Sp"
PY="/home/shaoyuantian/anaconda3/envs/hif4/bin/python"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 TORCH_NUM_THREADS=4 MKL_NUM_THREADS=4

OUT="${OUT_DIR:-Inference_Paradigm_Conversion/results}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_e2e_mmlu_pro}"
GPU="${GPU:-7}"
MAX_SAMPLES="${MAX_SAMPLES:-300}"
SRC_CKPT="${SRC_CKPT:-Qmodel/Qwen3-8B-FPQuant-QAT-NVFP4-Dequant-BF16-NoHadamard}"
HIF4_CKPT="${HIF4_CKPT:-Inference_Paradigm_Conversion/artifacts/Qwen3-8B-NVFP4QAT-HiF4QDQ-BF16}"
mkdir -p "$OUT/$RUN_ID/logs" "$(dirname "$HIF4_CKPT")"

if [[ ! -f "$HIF4_CKPT/ipc_hif4_materialize.json" ]]; then
  echo "[mmlu_pro] materialize HiF4 weights → $HIF4_CKPT"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m Inference_Paradigm_Conversion.ipc_analysis.eval.materialize_hif4_checkpoint \
    --src "$SRC_CKPT" --dst "$HIF4_CKPT" --device cuda:0 \
    2>&1 | tee "$OUT/$RUN_ID/logs/materialize.log"
fi

run_one() {
  local role="$1"
  local model_path="$2"
  local fake_act="$3"
  local odir="$OUT/$RUN_ID/p2_${role}"
  mkdir -p "$odir"
  echo "[mmlu_pro] P2 ${role}: model=${model_path} fake_act=${fake_act} gpu=${GPU}"
  # temp=0 → deterministic single run (plan allows once if deterministic)
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" main.py \
    --model_path "$model_path" \
    --datasets "mmlu_pro|0" \
    --max_samples "$MAX_SAMPLES" \
    --tensor_parallel_size 1 \
    --max_model_length 8192 \
    --max_new_tokens 2048 \
    --temperature 0 \
    --top_p 1.0 \
    --top_k 1 \
    --gpu_memory_utilization "${GPU_MEM_UTIL:-0.55}" \
    --fake_act_quant "$fake_act" \
    --fake_act_quant_exclude "lm_head" \
    --disable_thinking \
    --output_dir "$odir" \
    2>&1 | tee "$OUT/$RUN_ID/logs/p2_${role}.log"
}

run_one source "$SRC_CKPT" nvfp4
run_one target "$HIF4_CKPT" hif4

"$PY" - <<PY
import json, glob
from pathlib import Path
from Inference_Paradigm_Conversion.ipc_analysis.io_utils import atomic_write_json, write_csv, write_text

run = Path("$OUT") / "$RUN_ID"

def pick_score(role: str) -> float:
    files = sorted(glob.glob(str(run / f"p2_{role}" / "*" / "results" / "results_*.json")))
    if not files:
        # CustomEvaluationTracker nests under short model name
        files = sorted(run.glob(f"p2_{role}/**/results/results_*.json"))
        files = [str(p) for p in files]
    if not files:
        raise FileNotFoundError(f"no mmlu_pro results for {role} under {run}/p2_{role}")
    data = json.loads(Path(files[-1]).read_text())
    block = data["results"]["mmlu_pro|0"]
    for k in ("extractive_match", "exact_match", "acc"):
        if k in block:
            return float(block[k]), files[-1], k
    raise KeyError(f"no score key in {files[-1]}: {list(block)}")

rows = []
# P1 unsupported
rows.append({
    "e2e_kind": "semantic_e2e",
    "path_id": "P1_semantic",
    "task": "mmlu_pro",
    "source_score": None,
    "target_score": None,
    "delta_target_minus_source": None,
    "runtime_e2e": "unsupported_by_hardware_or_not_run",
    "status": "unsupported_by_vllm_fake_act",
    "reason": "vLLM fake_act_quant has no MXFP8; P1 requires MXFP8 on both source/target",
})
s, sf, sk = pick_score("source")
t, tf, tk = pick_score("target")
rows.append({
    "e2e_kind": "semantic_e2e",
    "path_id": "P2_matched_semantic",
    "task": "mmlu_pro",
    "source_score": s,
    "target_score": t,
    "delta_target_minus_source": t - s,
    "runtime_e2e": "unsupported_by_hardware_or_not_run",
    "status": "ok",
    "source_file": sf,
    "target_file": tf,
    "metric_key_source": sk,
    "metric_key_target": tk,
    "max_samples": int("$MAX_SAMPLES"),
    "note": "main.py vLLM+lighteval; temp=0 disable_thinking; W HiF4 via materialized QDQ ckpt",
})
write_csv(run / "e2e_mmlu_pro_summary.csv", rows)
summary = {"run_id": "$RUN_ID", "rows": rows}
atomic_write_json(run / "e2e_mmlu_pro_summary.json", summary)
write_text(Path("$OUT") / "latest_e2e_mmlu_pro_run_id.txt", "$RUN_ID")
print(json.dumps(summary, indent=2))
PY
echo "MMLU-PRO E2E DONE $RUN_ID"
