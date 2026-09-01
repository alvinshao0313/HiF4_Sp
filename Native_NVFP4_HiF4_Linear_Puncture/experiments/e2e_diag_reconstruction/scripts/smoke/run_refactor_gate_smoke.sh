#!/usr/bin/env bash
# Task 14.2 real-model mini smoke + resume parity on available GPUs.
# Hyperparams locked to plan: train=2 val=1 epoch=1; formal Fusable policy explicit.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"

STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
BASE="${RESULTS_ROOT}/smoke_refactor_${STAMP}"
CALIB="${BASE}/calib_t2_v1"
mkdir -p "${BASE}" "${CALIB}"

GPU_POOL="${GPU_POOL:-6,7}"
export GPU_POOL
init_stage_gpu_pool
gpu0="${AVAILABLE_GPUS[0]}"
gpu1="${AVAILABLE_GPUS[1]:-${AVAILABLE_GPUS[0]}}"

COMMON_SMOKE=(
  --calib_source s1k_original
  --calib_cache_dir "${CALIB}"
  --calib_nsamples 2
  --calib_val_nsamples 1
  --diag_epochs 1
  --diag_batch_size 1
  --optimizer AdamW
  --weight_decay 0.0
  --diag_scheduler cosine
  --loss_rollback on
  --router_rollback on
  --diag_mode fusable
)

echo "=== prepare mini smoke calib train=2 val=1 ==="
e2e_prepare_calib \
  --calib_source s1k_original \
  --calib_nsamples 2 \
  --calib_val_nsamples 1 \
  --calib_seed 42 \
  --calib_cache_dir "${CALIB}"

run_a="${BASE}/fusable_lambda0_layer0"
run_b="${BASE}/fusable_lambda05_layer0"
echo "=== A/B Fusable λ smoke on gpu=${gpu0}/${gpu1} ==="
CUDA_VISIBLE_DEVICES="${gpu0}" e2e_train \
  "${COMMON_SMOKE[@]}" \
  --output_dir "${run_a}" \
  --router_align_loss_weight 0.0 \
  --start_layer 0 --end_layer 0 &
pid_a=$!
CUDA_VISIBLE_DEVICES="${gpu1}" e2e_train \
  "${COMMON_SMOKE[@]}" \
  --output_dir "${run_b}" \
  --router_align_loss_weight 0.5 \
  --start_layer 0 --end_layer 0 &
pid_b=$!
wait_gpu_wave "${pid_a}" "${pid_b}"

python - <<PY
from pathlib import Path
import json
import math
import torch

def check_fusable_smoke(run: Path, *, expect_kl_weight: float):
    layer = run / "layers" / "layer_00"
    assert (layer / "candidate_best_diag.pt").is_file(), run
    assert (layer / "best_diag.pt").is_file(), run
    assert (layer / "candidate_metrics.json").is_file(), run
    cand_metrics = json.loads((layer / "candidate_metrics.json").read_text(encoding="utf-8"))
    metrics = json.loads((layer / "metrics.json").read_text(encoding="utf-8"))
    art = run / "checkpoint" / "final_model" / "conversion_state.pt"
    state = torch.load(art, map_location="cpu", weights_only=False)
    assert int(state["schema_version"]) == 3
    assert float(state["router_align_loss_weight"]) == expect_kl_weight
    assert state["optimizer"] == "AdamW"
    assert float(state["weight_decay"]) == 0.0
    assert state["diag_scheduler"] == "cosine"
    assert bool(state["resolved_loss_rollback_enabled"]) is True
    assert bool(state["resolved_router_rollback_enabled"]) is True
    rec = state["layers"]["0"]
    assert "candidate_z" in rec and "adopted_z" in rec
    cand = rec["candidate_z"]
    adopted = rec["adopted_z"]
    if rec.get("router_rollback_applied"):
        for k in ("z_qkv", "z_vo", "z_ud"):
            assert torch.equal(cand[k], adopted[k]), k
        assert not torch.equal(cand["z_gu"], adopted["z_gu"]) or torch.count_nonzero(cand["z_gu"]) == 0
    if expect_kl_weight > 0:
        kl = float(cand_metrics["candidate_best_router_kl"])
        assert math.isfinite(kl), kl
        obj = float(cand_metrics["candidate_best_objective"])
        recon = float(cand_metrics["candidate_best_val_loss"])
        assert abs(obj - (recon + expect_kl_weight * kl)) < 1e-5
    print(f"ok smoke {run.name}")

check_fusable_smoke(Path("${run_a}"), expect_kl_weight=0.0)
check_fusable_smoke(Path("${run_b}"), expect_kl_weight=0.5)
PY

# D: continuous 0→2, then resume layer2 from an exact copy of that prefix.
cont="${BASE}/resume_continuous"
part="${BASE}/resume_partial"
echo "=== resume parity continuous on gpu=${gpu0} ==="
CUDA_VISIBLE_DEVICES="${gpu0}" e2e_train \
  "${COMMON_SMOKE[@]}" \
  --output_dir "${cont}" \
  --router_align_loss_weight 0.0 \
  --start_layer 0 --end_layer 2

echo "=== prepare exact prefix copy for resume ==="
rm -rf "${part}"
mkdir -p "${part}/layers"
cp "${cont}/config.json" "${part}/"
cp -a "${cont}/layers/layer_00" "${cont}/layers/layer_01" "${part}/layers/"

echo "=== resume parity suffix start_layer=2 on gpu=${gpu0} ==="
CUDA_VISIBLE_DEVICES="${gpu0}" e2e_train \
  "${COMMON_SMOKE[@]}" \
  --output_dir "${part}" \
  --router_align_loss_weight 0.0 \
  --start_layer 2 --end_layer 2

conda run --no-capture-output -n hif4 python - <<PY
from pathlib import Path
import json
import torch

cont = Path("${cont}")
part = Path("${part}")

def load_layer(run: Path, lid: int):
    d = run / "layers" / f"layer_{lid:02d}"
    cand = torch.load(d / "candidate_best_diag.pt", map_location="cpu", weights_only=False)
    adopted = torch.load(d / "best_diag.pt", map_location="cpu", weights_only=False)
    metrics = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    return cand, adopted, metrics

c_cand, c_adopted, c_m = load_layer(cont, 2)
p_cand, p_adopted, p_m = load_layer(part, 2)

# Exact prefix replay must reproduce the continuous layer-2 identity input/loss.
assert abs(float(c_m["identity_val_loss"]) - float(p_m["identity_val_loss"])) < 1e-7
assert c_m["candidate_best_epoch"] == p_m["candidate_best_epoch"]
assert bool(c_m["loss_rollback_applied"]) == bool(p_m["loss_rollback_applied"])
assert bool(c_m["router_rollback_applied"]) == bool(p_m["router_rollback_applied"])

# Layer training under BF16/CUDA is not bit-exact; allow small numerical drift.
atol = 5e-3
for name in c_cand:
    torch.testing.assert_close(c_cand[name], p_cand[name], atol=atol, rtol=0.0, msg=f"candidate {name}")
    torch.testing.assert_close(c_adopted[name], p_adopted[name], atol=atol, rtol=0.0, msg=f"adopted {name}")
assert abs(float(c_m["candidate_best_val_loss"]) - float(p_m["candidate_best_val_loss"])) < atol
print("ok resume parity layer2")
PY

# C: adopted/candidate materialize + short TP2 generate
# Must be a real .py file: VLLM spawn cannot re-import `python -` / stdin as __main__.
artifact_a="${run_a}/checkpoint/final_model/conversion_state.pt"
mkdir -p "${BASE}/diagnostics"
echo "=== candidate/adopted materialize + short TP2 generate ==="
smoke_c_py="${BASE}/diagnostics/smoke_c_tp2_generate.py"
cat > "${smoke_c_py}" <<PY
from pathlib import Path
import json
import os
import shutil

# Avoid CUDA-fork hangs in in-process vLLM TP workers.
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.config import DEFAULT_MODEL_PATH
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.evaluation.vllm_runner import (
    resolve_vllm_eval_spec,
    cleanup_materialized_eval_spec,
)

base = Path("${BASE}/diagnostics")
artifact = Path("${artifact_a}")
for variant in ("adopted", "candidate"):
    out = base / f"{variant}_replay"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    cfg = {
        "source_run": str(artifact.resolve().parents[2]),
        "artifact_diag_variant": variant,
        "source_artifact": str(artifact.resolve()),
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\\n", encoding="utf-8")
    print(f"materialize variant={variant}", flush=True)
    spec = resolve_vllm_eval_spec(
        variant="artifact",
        model_path=DEFAULT_MODEL_PATH,
        artifact_path=str(artifact),
        artifact_diag_variant=variant,
        output_dir=out,
        device="cuda",
    )
    try:
        os.environ["HIF4_RUNTIME_SPEC_PATH"] = str(spec.hif4_runtime_spec_path.resolve())
        from vllm import LLM, SamplingParams
        llm = LLM(
            model=str(spec.model_path),
            trust_remote_code=True,
            tensor_parallel_size=2,
            kv_cache_dtype="bfloat16",
            enforce_eager=True,
            max_model_len=256,
            seed=42,
            additional_config={"hif4_runtime_spec_path": str(spec.hif4_runtime_spec_path)},
        )
        outs = llm.generate(["你好"], SamplingParams(max_tokens=8, temperature=0.0))
        text = outs[0].outputs[0].text
        assert isinstance(text, str)
        print(f"ok tp2 generate variant={variant} text={text!r}", flush=True)
        del llm
    finally:
        cleanup_materialized_eval_spec(spec)
print("SMOKE_C_OK", flush=True)
PY
CUDA_VISIBLE_DEVICES="${gpu0},${gpu1}" conda run --no-capture-output -n hif4 python "${smoke_c_py}"

echo "smoke_refactor_dir=${BASE}"
