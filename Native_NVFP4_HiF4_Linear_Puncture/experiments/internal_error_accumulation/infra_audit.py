"""S0 infrastructure audit against real_vllm_hooks primitives."""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from .config import PACKAGE_ROOT, RESULTS_ROOT, RUNS_ROOT, STAGE_ORDER
from .run_state import atomic_write_json


REQUIRED_HOOK_EXPORTS = {
    "Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.worker_hooks": [
        "InstallHooksOp",
        "BeginSampleOp",
        "flush_sample",
        "remove_hooks",
    ],
    "Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.build_llm": [
        "build_real_vllm",
    ],
    "Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.forced_trajectory": [
        "make_forced_sampling_params",
        "ForcedTrajectoryLogitsProcessor",
    ],
    "Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.format_conversion.production_puncture": [
        "ProductionPunctureOp",
        "frozen_router",
        "error_decomposition",
    ],
}

REQUIRED_PACKAGE_MODULES = [
    "config",
    "run_state",
    "bf16_residual",
    "math_utils",
    "residual_ledger",
    "build_state_cohort",
    "capture_states",
    "structural_accumulation",
    "moe_decomposition",
    "router_contribution",
    "one_step_intervention",
    "qkv_slices",
    "lm_head_gate",
    "layer_protection",
    "objective_study",
    "variant_validation",
    "report",
    "run_pipeline",
    "infra_audit",
    "substructure_causal",
    "s3_full48_causal",
    "protection_objective",
    "practical_protection",
    "objective_train",
    "objective_train_exec",
    "router_objective",
    "router_teacher_cache",
    "selected_layer_objective_trainer",
    "topk_validate",
    "e2e_validate",
]

REQUIRED_SCRIPTS = [
    "scripts/run_formal.sh",
    "scripts/launch_detached.sh",
    "scripts/show_status.sh",
]


def audit_infra() -> dict[str, Any]:
    failures: list[str] = []
    for mod_name, attrs in REQUIRED_HOOK_EXPORTS.items():
        try:
            mod = importlib.import_module(mod_name)
        except Exception as exc:  # noqa: BLE001 - surface import failures as gate failures
            failures.append(f"import {mod_name}: {exc}")
            continue
        for attr in attrs:
            if not hasattr(mod, attr):
                failures.append(f"missing {mod_name}.{attr}")
    pkg = "Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation"
    for name in REQUIRED_PACKAGE_MODULES:
        try:
            importlib.import_module(f"{pkg}.{name}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"package module {name}: {exc}")
    for rel in REQUIRED_SCRIPTS:
        path = PACKAGE_ROOT / rel
        if not path.is_file():
            failures.append(f"missing script {path}")
        elif rel.endswith(".sh") and not path.stat().st_mode & 0o111:
            # launcher scripts should be executable; non-fatal for audit content but flagged
            failures.append(f"script not executable: {path}")
    for path in (RESULTS_ROOT, RUNS_ROOT):
        path.mkdir(parents=True, exist_ok=True)
    # Isolation: must not write into long_trajectory_stability by default
    if "long_trajectory_stability" in str(RUNS_ROOT):
        failures.append("RUNS_ROOT incorrectly points at long_trajectory_stability")
    if STAGE_ORDER[0] != "S0_INFRA" or STAGE_ORDER[2] != "S2_LAYER_CAUSAL":
        failures.append(f"STAGE_ORDER unexpected: {STAGE_ORDER}")
    result = {
        "pass": not failures,
        "failures": failures,
        "n_required_modules": len(REQUIRED_PACKAGE_MODULES),
        "n_required_hook_modules": len(REQUIRED_HOOK_EXPORTS),
        "runs_root": str(RUNS_ROOT),
        "package_root": str(PACKAGE_ROOT),
    }
    return result


def write_s0_artifacts(protocol_dir: Path, audit: dict[str, Any]) -> None:
    protocol_dir = Path(protocol_dir)
    protocol_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(protocol_dir / "S0_INFRA_AUDIT.json", audit)
    (protocol_dir / "S0_INFRA_GATE.md").write_text(
        "# S0_INFRA Gate\n\n"
        f"- pass: {audit['pass']}\n"
        f"- failures: {audit['failures']}\n"
        f"- runs_root: {audit['runs_root']}\n",
        encoding="utf-8",
    )
    if not audit["pass"]:
        raise RuntimeError(f"S0_INFRA audit failed: {audit['failures']}")
