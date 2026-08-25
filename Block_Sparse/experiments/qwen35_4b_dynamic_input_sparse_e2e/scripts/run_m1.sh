#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
ensure_run_dir
write_manifest
KEEP="${1:-}"
LOG="${RUN_DIR}/logs/m1_${KEEP:-all}.log"
exec > >(tee -a "${LOG}") 2>&1

# Runtime gate: short timing probe before full matrix when KEEP empty.
if [[ -z "${KEEP}" ]]; then
  TIMING="${RUN_DIR}/m1_runtime.json"
  if [[ ! -f "${TIMING}" ]]; then
    echo "=== M1 runtime probe ==="
    GPU="$(pick_gpu)"
    CUDA_VISIBLE_DEVICES="${GPU}" run_py python - <<'PY' "${TIMING}"
import json, time, sys
from pathlib import Path
import torch
REPO = Path("/home/shaoyuantian/program/HiF4_Sp")
import sys as _s
_s.path.insert(0, str(REPO))
_s.path.insert(0, str(REPO / "Block_Sparse"))
from block_pruning.config import GradientBlockPruningConfig
from block_pruning.model_loader import load_model_and_tokenizer
from Block_Sparse.dynamic_input_sparse.config import DynamicInputMaskMethod, DynamicInputSparseConfig
from Block_Sparse.dynamic_input_sparse.hf_reference import DynamicInputSparseMLPReference

out = Path(sys.argv[1])
cfg = GradientBlockPruningConfig(
    model_path="Qwen/Qwen3.5-4B", output_dir="/tmp/m1_time",
    score_type="magnitude", dtype="bfloat16", device="cuda",
    gradient_checkpointing=False, trust_remote_code=True,
)
model, _ = load_model_and_tokenizer(cfg)
mlp = None
for name, mod in model.named_modules():
    if name.endswith(".mlp") and hasattr(mod, "gate_proj"):
        mlp = mod
        break
wrap = DynamicInputSparseMLPReference(
    mlp, DynamicInputSparseConfig(method=DynamicInputMaskMethod.M1_ORACLE, keep_ratio=0.5)
)
x = torch.randn(8, mlp.gate_proj.in_features, device=next(mlp.parameters()).device, dtype=torch.bfloat16)
torch.cuda.synchronize(); t0=time.perf_counter()
_ = wrap(x)
torch.cuda.synchronize(); dt=time.perf_counter()-t0
ms_tok = dt * 1000 / 8
peak = torch.cuda.max_memory_allocated() / (1024**3)
# Rough full-run projection: 7 configs * (ARC tokens ~ few M + MMLU300 + AIME) — use smoke scale * 400
# Conservative: 32 layers * 2 linears * tokens_est; report probe only and a coarse bound.
est_gpu_hours = ms_tok / 1000 / 3600 * 32 * 2 * 50000  # 50k tokens proxy through all layers
payload = {
    "ms_per_token_mlp_forward_includes_gate_up_and_down": ms_tok,
    "peak_alloc_gb": peak,
    "est_gpu_hours_proxy_50k_tokens": est_gpu_hours,
    "stop_rule_hours": 48,
}
out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
if est_gpu_hours > 48:
    print("M1_FULL_RUN_BLOCKED by 48 GPU-hour stop rule")
    raise SystemExit(42)
PY
    status=$?
    if [[ "${status}" -eq 42 ]]; then
      cat > "${RUN_DIR}/M1_FULL_RUN_BLOCKED.md" <<EOF
# M1 Full Run Blocked

Measured probe exceeded the 48 GPU-hour stop rule. See \`m1_runtime.json\`.

Pending commands (do not approximate M1):
\`\`\`bash
RUN_DIR=${RUN_DIR} bash ${SCRIPT_DIR}/run_m1.sh 0.75
RUN_DIR=${RUN_DIR} bash ${SCRIPT_DIR}/run_m1.sh 0.50
RUN_DIR=${RUN_DIR} bash ${SCRIPT_DIR}/run_m1.sh 0.25
\`\`\`
EOF
      echo "Wrote M1_FULL_RUN_BLOCKED.md"
      exit 0
    fi
  fi
fi

PIN_GPU="${PIN_GPU:-}"
if [[ -n "${KEEP}" ]]; then
  case "${KEEP}" in
    0.75|0.750) tag=m1_keep075 ;;
    0.5|0.50|0.500) tag=m1_keep050 ;;
    0.25|0.250) tag=m1_keep025 ;;
    *) tag="m1_keep${KEEP}" ;;
  esac
  run_method_full "${tag}" m1_oracle "${KEEP}" "${PIN_GPU}"
else
  # After probe, sweep keep ratios in parallel on free GPUs.
  IFS=',' read -r -a gpus <<< "${GPU_POOL}"
  pids=()
  i=0
  for keep in 0.75 0.50 0.25; do
    gpu="${gpus[$((i % ${#gpus[@]}))]}"
    (
      export RUN_DIR PIN_GPU="${gpu}"
      bash "${SCRIPT_DIR}/run_m1.sh" "${keep}"
    ) > "${RUN_DIR}/logs/m1_${keep}_worker.log" 2>&1 &
    pids+=($!)
    i=$((i + 1))
    sleep 10
  done
  ec=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || ec=1
  done
  [[ "${ec}" -eq 0 ]]
fi
echo "m1 DONE"
