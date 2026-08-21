#!/usr/bin/env bash
# Task 12: Qwen3-30B-A3B-NVFP4 emulation MMLU-Pro 300 (two KV modes).
# Only variable between runs: kv_cache_dtype=bfloat16 vs auto.
# Formal runs use enforce_eager (emulation is not torch.compile safe on this backport).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

CKPT="${CKPT:-/home/shaoyuantian/.cache/huggingface/hub/models--nvidia--Qwen3-30B-A3B-NVFP4/snapshots/2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3}"
REPORT_DIR="${REPORT_DIR:-${REPO_ROOT}/NVFP4/reports/vllm_v027_nvfp4_backport}"
REPORT="${REPORT:-${REPORT_DIR}/qwen3_30b_accuracy.md}"
OUT_ROOT="${OUT_ROOT:-${REPORT_DIR}/mmlu_pro_runs}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_nvfp4_emulation_mmlu_pro}"
MAX_SAMPLES="${MAX_SAMPLES:-300}"
TP="${TP:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LENGTH="${MAX_MODEL_LENGTH:-8192}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "${REPORT_DIR}" "${OUT_ROOT}/${RUN_ID}/logs"

run_one() {
  local kv_mode="$1"
  local tag="kv_${kv_mode}"
  local odir="${OUT_ROOT}/${RUN_ID}/${tag}"
  local log="${OUT_ROOT}/${RUN_ID}/logs/${tag}.log"
  mkdir -p "${odir}"
  echo "=== mmlu_pro ${tag}: TP=${TP} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ==="
  set +e
  conda run -n hif4 --no-capture-output python main.py \
    --model_path "${CKPT}" \
    --datasets "mmlu_pro|0" \
    --max_samples "${MAX_SAMPLES}" \
    --tensor_parallel_size "${TP}" \
    --max_model_length "${MAX_MODEL_LENGTH}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature 0 \
    --top_p 1.0 \
    --top_k 1 \
    --gpu_memory_utilization "${GPU_MEM_UTIL}" \
    --disable_thinking \
    --enforce_eager \
    --linear_backend emulation \
    --moe_backend emulation \
    --kv_cache_dtype "${kv_mode}" \
    --output_dir "${odir}" \
    >"${log}" 2>&1
  local rc=$?
  set -e
  echo "exit=${rc} log=${log}"
  return "${rc}"
}

BF16_OK=0
AUTO_OK=0
if run_one "bfloat16"; then BF16_OK=1; fi
if run_one "auto"; then AUTO_OK=1; fi

conda run -n hif4 python - <<PY
import glob
import json
from datetime import datetime, timezone
from pathlib import Path

run = Path("${OUT_ROOT}") / "${RUN_ID}"
report = Path("${REPORT}")
ckpt = "${CKPT}"
tp = int("${TP}")
bf16_ok = bool(int("${BF16_OK}"))
auto_ok = bool(int("${AUTO_OK}"))


def pick_score(tag: str):
    files = sorted(glob.glob(str(run / tag / "**" / "results" / "results_*.json"), recursive=True))
    if not files:
        return None, None, None
    data = json.loads(Path(files[-1]).read_text())
    block = data["results"]["mmlu_pro|0"]
    for k in ("extractive_match", "exact_match", "acc"):
        if k in block:
            return float(block[k]), files[-1], k
    return None, files[-1], None


bf16_score, bf16_file, bf16_key = pick_score("kv_bfloat16")
auto_score, auto_file, auto_key = pick_score("kv_auto")
delta = None
if bf16_score is not None and auto_score is not None:
    delta = auto_score - bf16_score

lines = [
    "# Qwen3-30B-A3B-NVFP4 MMLU-Pro 300 (Task 12)",
    "",
    f"- Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
    f"- Model snapshot: \`{ckpt}\`",
    f"- RUN_ID: \`${RUN_ID}\`",
    f"- GPU CUDA_VISIBLE_DEVICES: \`${Path('/dev/null').write_text('') or __import__('os').environ.get('CUDA_VISIBLE_DEVICES')}\`",
    f"- TP: {tp}",
    "- linear_backend: emulation",
    "- moe_backend: emulation",
    "- enforce_eager: true",
    f"- max_samples: {int('${MAX_SAMPLES}')}",
    "- Upstream backport baseline: vLLM v0.27.0",
    "",
    "## Results",
    "",
    "| KV mode | run_ok | score_key | accuracy | results_file |",
    "|---|---|---|---|---|",
    f"| bfloat16 | {bf16_ok} | {bf16_key} | {bf16_score} | {bf16_file} |",
    f"| auto (checkpoint) | {auto_ok} | {auto_key} | {auto_score} | {auto_file} |",
    "",
    f"- Absolute difference (auto - bf16): **{delta}**",
    "",
    "## Notes",
    "",
    "- Observation-only; no accuracy pass threshold.",
    "- Formal emulation runs require enforce_eager (graph/compile unsupported).",
    f"- Logs: \`{run / 'logs'}\`",
    "",
]
report.write_text("\n".join(lines))
print(f"wrote {report}")
if not (bf16_ok and auto_ok and bf16_score is not None and auto_score is not None):
    raise SystemExit(1)
PY
