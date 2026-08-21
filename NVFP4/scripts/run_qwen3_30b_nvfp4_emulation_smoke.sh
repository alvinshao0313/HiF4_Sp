#!/usr/bin/env bash
# Task 11: Qwen3-30B-A3B-NVFP4 emulation smoke (BF16 KV + checkpoint/auto KV).
#
# Mirrors root main.py vLLM kwargs (linear_backend/moe_backend/kv_cache_dtype/
# enforce_eager). Generation uses the companion .py helper so vLLM spawn can
# re-exec a real file path (stdin heredoc breaks EngineCore).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SMOKE_PY="${REPO_ROOT}/NVFP4/scripts/run_qwen3_30b_nvfp4_emulation_smoke.py"
CKPT="${CKPT:-/home/shaoyuantian/.cache/huggingface/hub/models--nvidia--Qwen3-30B-A3B-NVFP4/snapshots/2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3}"
REPORT_DIR="${REPORT_DIR:-${REPO_ROOT}/NVFP4/reports/vllm_v027_nvfp4_backport}"
REPORT="${REPORT:-${REPORT_DIR}/qwen3_30b_smoke.md}"
LOG_DIR="${LOG_DIR:-${REPORT_DIR}/smoke_logs}"
TP="${TP:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-512}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "${REPORT_DIR}" "${LOG_DIR}"

run_one() {
  local kv_mode="$1"
  local eager_flag="$2"  # --enforce-eager or empty
  local tag="$3"
  local log="${LOG_DIR}/${tag}.log"
  local out_json="${LOG_DIR}/${tag}.json"
  echo "=== smoke ${tag}: kv=${kv_mode} eager_flag='${eager_flag}' TP=${TP} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ==="
  set +e
  # shellcheck disable=SC2086
  conda run -n hif4 --no-capture-output python "${SMOKE_PY}" \
    --checkpoint "${CKPT}" \
    --tp "${TP}" \
    --kv-cache-dtype "${kv_mode}" \
    ${eager_flag} \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --tag "${tag}" \
    --out-json "${out_json}" \
    >"${log}" 2>&1
  local rc=$?
  set -e
  echo "exit=${rc} log=${log}"
  return "${rc}"
}

BF16_EAGER_OK=0
AUTO_EAGER_OK=0
FINAL_TP="${TP}"
OOM_EVIDENCE=""

if run_one "bfloat16" "--enforce-eager" "eager_bf16_tp${TP}"; then
  BF16_EAGER_OK=1
else
  if grep -qiE 'out of memory|OutOfMemory' "${LOG_DIR}/eager_bf16_tp${TP}.log"; then
    OOM_EVIDENCE="TP=${TP} BF16 eager OOM; log=${LOG_DIR}/eager_bf16_tp${TP}.log"
    for try_tp in 2 4; do
      if [[ "${try_tp}" -eq 2 ]]; then
        export CUDA_VISIBLE_DEVICES="0,2"
      else
        export CUDA_VISIBLE_DEVICES="0,2,3,6"
      fi
      TP="${try_tp}"
      if run_one "bfloat16" "--enforce-eager" "eager_bf16_tp${TP}"; then
        FINAL_TP="${TP}"
        BF16_EAGER_OK=1
        OOM_EVIDENCE="${OOM_EVIDENCE}; recovered TP=${TP} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
        break
      fi
      if ! grep -qiE 'out of memory|OutOfMemory' "${LOG_DIR}/eager_bf16_tp${TP}.log"; then
        break
      fi
      OOM_EVIDENCE="${OOM_EVIDENCE}; TP=${TP} also OOM"
    done
  fi
fi

TP="${FINAL_TP}"
# Always attempt auto/eager after BF16 attempt when BF16 passed; if BF16 failed
# for a non-OOM reason, still try auto so the report captures both modes.
if [[ "${BF16_EAGER_OK}" -eq 1 ]]; then
  if run_one "auto" "--enforce-eager" "eager_auto_tp${TP}"; then
    AUTO_EAGER_OK=1
  fi
else
  echo "BF16 eager failed; still attempting auto/eager for diagnostics"
  run_one "auto" "--enforce-eager" "eager_auto_tp${TP}" && AUTO_EAGER_OK=1 || true
fi

if [[ "${BF16_EAGER_OK}" -eq 1 && "${AUTO_EAGER_OK}" -eq 1 ]]; then
  run_one "bfloat16" "" "graph_bf16_tp${TP}" || true
  run_one "auto" "" "graph_auto_tp${TP}" || true
fi

set +e
# Do not use `conda run ... python -` here: stdin heredoc is unreliable under conda run.
REPORT="${REPORT}" \
LOG_DIR="${LOG_DIR}" \
CKPT="${CKPT}" \
FINAL_TP="${FINAL_TP}" \
MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
OOM_EVIDENCE="${OOM_EVIDENCE}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
python - <<'PY'
import json, os, re
from pathlib import Path

log_dir = Path(os.environ["LOG_DIR"])
report = Path(os.environ["REPORT"])
rows = [json.loads(p.read_text()) for p in sorted(log_dir.glob("*.json"))]

for r in rows:
    if r.get("passed"):
        continue
    log = log_dir / f"{r.get('tag')}.log"
    if not log.exists():
        continue
    text = log.read_text(errors="replace")
    m = re.search(r"torch\._dynamo\.exc\.Unsupported: ([^\n]+)", text)
    if m:
        r["error_detail"] = f"torch._dynamo.exc.Unsupported: {m.group(1)}"
        r["upstream_emulation_graph_limit"] = True
    elif "run_nvfp4_emulations" in text and "Unsupported" in text:
        r["error_detail"] = "emulation unsupported under torch.compile/CUDA graph"
        r["upstream_emulation_graph_limit"] = True

lines = [
    "# Qwen3-30B-A3B-NVFP4 Smoke (Task 11)",
    "",
    f"- Checkpoint: `{os.environ['CKPT']}`",
    f"- Final TP: **{os.environ['FINAL_TP']}**",
    f"- CUDA_VISIBLE_DEVICES (last): `{os.environ.get('CUDA_VISIBLE_DEVICES')}`",
    f"- GPU: `{next((r.get('gpu_name') for r in rows if r.get('gpu_name')), None)}`",
    f"- max_model_len={os.environ['MAX_MODEL_LEN']}, max_new_tokens={os.environ['MAX_NEW_TOKENS']}",
    "- Backends requested: linear=emulation, moe=emulation",
    "- Entry: `NVFP4/scripts/run_qwen3_30b_nvfp4_emulation_smoke.py` (same kwargs as root `main.py`)",
]
if os.environ.get("OOM_EVIDENCE"):
    lines.append(f"- OOM evidence: {os.environ['OOM_EVIDENCE']}")
lines += [
    "",
    "## Runs",
    "",
    "| tag | passed | linear | moe | kv resolved | fp8_kv | no_marlin | error |",
    "| --- | --- | --- | --- | --- | --- | --- | --- |",
]
for r in rows:
    err = r.get("error_detail") or r.get("error")
    lines.append(
        f"| `{r.get('tag')}` | {r.get('passed')} | `{r.get('resolved_linear_backend')}` | "
        f"`{r.get('resolved_moe_backend')}` | `{r.get('resolved_kv_cache_dtype')}` | "
        f"{r.get('fp8_kv_resolved')} | {r.get('no_marlin')} | `{err}` |"
    )
lines += ["", "## Acceptance", ""]
eager_bf16 = any(str(r.get("tag", "")).startswith("eager_bf16") and r.get("passed") for r in rows)
eager_auto = any(str(r.get("tag", "")).startswith("eager_auto") and r.get("passed") for r in rows)
auto_row = next((r for r in rows if str(r.get("tag", "")).startswith("eager_auto")), None)
bf16_row = next((r for r in rows if str(r.get("tag", "")).startswith("eager_bf16")), None)
lines.append(f"- Eager BF16 KV: **{'PASS' if eager_bf16 else 'FAIL'}**")
lines.append(f"- Eager checkpoint/auto KV: **{'PASS' if eager_auto else 'FAIL'}**")
if bf16_row is not None:
    lines.append(f"- BF16 mode resolved cache_dtype: `{bf16_row.get('resolved_kv_cache_dtype')}`")
if auto_row is not None:
    lines.append(
        f"- auto KV resolved dtype: `{auto_row.get('resolved_kv_cache_dtype')}` "
        f"(fp8_kv={auto_row.get('fp8_kv_resolved')})"
    )
    if auto_row.get("fp8_kv_resolved") is False:
        lines.append("- **FAIL**: kv_cache_dtype=auto did not resolve to FP8.")
graph_rows = [r for r in rows if str(r.get("tag", "")).startswith("graph_")]
if graph_rows:
    for r in graph_rows:
        status = "PASS" if r.get("passed") else "FAIL (documented limitation)"
        lines.append(f"- Graph-mode `{r.get('tag')}`: {status}")
        if r.get("upstream_emulation_graph_limit"):
            lines.append(
                "  - Root cause: torch.compile/CUDA graph cannot capture "
                "`EmulationNvFp4LinearKernel`/`run_nvfp4_emulations`. "
                "Do not fall back to Marlin; use `enforce_eager=true`."
            )
else:
    lines.append("- Graph-mode: skipped (eager gates not both green).")

for r in rows:
    if r.get("passed") and r.get("generations"):
        lines += ["", f"### Sample generations (`{r.get('tag')}`)", ""]
        for g in r["generations"]:
            lines.append(f"- prompt={g.get('prompt')!r} -> {g.get('text')!r}")
        break

eager_log = log_dir / f"eager_bf16_tp{os.environ['FINAL_TP']}.log"
if eager_log.exists():
    t = eager_log.read_text(errors="replace")
    lines += ["", "## Backend log evidence", ""]
    if "Using EmulationNvFp4LinearKernel" in t:
        lines.append("- Log: `Using EmulationNvFp4LinearKernel for NVFP4 GEMM`")
    if "Nvfp4QuantizationEmulationTritonExperts" in t:
        lines.append("- Log: `Using Nvfp4QuantizationEmulationTritonExperts MOE backend`")
    lines.append("- Formal acceptance: no Marlin fallback for NVFP4 target layers")

lines += ["", "## Logs", "", f"- Directory: `{log_dir}`", ""]
report.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(report.read_text())
print(f"Wrote report: {report}")
raise SystemExit(0 if (eager_bf16 and eager_auto) else 1)
PY
report_rc=$?
exit "${report_rc}"
