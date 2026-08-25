#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

ensure_run_dir
write_manifest
LOG="${RUN_DIR}/logs/unit_and_smoke.log"
exec > >(tee -a "${LOG}") 2>&1

echo "=== unit tests ==="
cd "${REPO_ROOT}"
# importlib mode avoids basename collisions (test_config.py in multiple dirs)
CUDA_VISIBLE_DEVICES="$(pick_gpu)" run_py pytest -q --import-mode=importlib \
  Block_Sparse/tests/dynamic_input_sparse \
  Block_Sparse/tests/input_mask_proxy_study

echo "=== HF MLP semantic smoke ==="
GPU="$(pick_gpu)"
CUDA_VISIBLE_DEVICES="${GPU}" run_py python - <<'PY'
import torch
from pathlib import Path
import sys
REPO = Path("/home/shaoyuantian/program/HiF4_Sp")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "Block_Sparse"))
from block_pruning.config import GradientBlockPruningConfig
from block_pruning.model_loader import load_model_and_tokenizer
from Block_Sparse.dynamic_input_sparse.config import DynamicInputMaskMethod, DynamicInputSparseConfig
from Block_Sparse.dynamic_input_sparse.hf_reference import (
    DynamicInputSparseMLPReference,
    install_dynamic_input_sparse_on_hf_model,
    expected_keep_counts,
)

cfg = GradientBlockPruningConfig(
    model_path="Qwen/Qwen3.5-4B",
    output_dir="/tmp/dyn_smoke",
    score_type="magnitude",
    dtype="bfloat16",
    device="cuda",
    gradient_checkpointing=False,
    trust_remote_code=True,
)
model, tok = load_model_and_tokenizer(cfg)
# locate first MLP
mlp = None
for name, mod in model.named_modules():
    if name.endswith(".mlp") and hasattr(mod, "gate_proj"):
        mlp = mod
        break
assert mlp is not None
x = torch.randn(4, mlp.gate_proj.in_features, device=next(mlp.parameters()).device, dtype=torch.bfloat16)
dense = mlp(x)
for method in (DynamicInputMaskMethod.M8_ENERGY, DynamicInputMaskMethod.M1_ORACLE):
    wrap = DynamicInputSparseMLPReference(
        mlp, DynamicInputSparseConfig(method=method, keep_ratio=1.0), capture_masks=True
    )
    out = wrap(x)
    err = (out.float() - dense.float()).abs().max().item()
    assert err <= 1e-3, (method, err)
for ratio in (0.75, 0.5, 0.25):
    cfg_d = DynamicInputSparseConfig(method=DynamicInputMaskMethod.M8_ENERGY, keep_ratio=ratio)
    wrap = DynamicInputSparseMLPReference(mlp, cfg_d, capture_masks=True)
    y = wrap(x)
    assert torch.isfinite(y).all()
    exp = expected_keep_counts(cfg_d)
    assert wrap.last_mx_gate_up.shape == (4, exp["gate_up_kb"])
    assert wrap.last_mx_down.shape == (4, exp["down_kb"])
    assert int(wrap.last_mx_gate_up.sum(dim=-1).unique().item()) == exp["gate_up_keep"]
    assert int(wrap.last_mx_down.sum(dim=-1).unique().item()) == exp["down_keep"]
print("HF MLP semantic smoke OK")
PY

echo "=== vLLM smoke (dense / m8 / m1 tiny) ==="
SMOKE_DIR="${RUN_DIR}/smoke"
mkdir -p "${SMOKE_DIR}"
GPU="$(pick_gpu)"
run_vllm_task none 1.0 mmlu_pro "${SMOKE_DIR}/dense" "${GPU}" 2 1
GPU="$(pick_gpu)"
run_vllm_task m8_energy 0.5 mmlu_pro "${SMOKE_DIR}/m8" "${GPU}" 8 1
GPU="$(pick_gpu)"
run_vllm_task m8_energy 0.5 aime25 "${SMOKE_DIR}/m8" "${GPU}" 1 0
GPU="$(pick_gpu)"
run_vllm_task m1_oracle 0.5 mmlu_pro "${SMOKE_DIR}/m1" "${GPU}" 8 1
GPU="$(pick_gpu)"
run_vllm_task m1_oracle 0.5 aime25 "${SMOKE_DIR}/m1" "${GPU}" 1 0

# ARC smoke via lm_eval limit
GPU="$(pick_gpu)"
run_arc none 1.0 "${SMOKE_DIR}/dense" "${GPU}" 4
GPU="$(pick_gpu)"
run_arc m8_energy 0.5 "${SMOKE_DIR}/m8" "${GPU}" 4
GPU="$(pick_gpu)"
run_arc m1_oracle 0.5 "${SMOKE_DIR}/m1" "${GPU}" 4

date -u +%Y-%m-%dT%H:%M:%SZ > "${RUN_DIR}/logs/unit_and_smoke.DONE"
echo "unit_and_smoke DONE -> ${RUN_DIR}"
