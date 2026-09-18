#!/usr/bin/env python3
"""Unique experiment state machine for internal_error_accumulation."""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.build_state_cohort import (
    build_and_write_cohort,
    ensure_wikitext2_shared_calibration,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.capture_states import (
    capture_variant_states,
    load_rank_records,
    load_raw_logits,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.config import (
    DEFAULT_MODEL_PATH,
    DEFAULT_PHASEA_ROOT,
    STAGE_ORDER,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_RUNNING,
    STATUS_WAITING_REVIEW,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.infra_audit import (
    audit_infra,
    write_s0_artifacts,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.layer_protection import (
    build_layer_selection_manifest,
    run_discovery_whole_layer_scan,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.objective_study import (
    build_objective_split_manifest,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.report import (
    write_final_report,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.residual_ledger import (
    check_tp_replica,
    extract_layer_tensors,
    index_capture_records,
    runtime_residual_closure_for_variant,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.run_state import (
    atomic_write_json,
    default_run_state,
    ensure_run_dirs,
    load_run_state,
    read_jsonl,
    save_run_state,
    stage_index,
    stages_inclusive,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.structural_accumulation import (
    analyze_state_pair,
    write_structural_reports,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.variant_validation import (
    write_variant_validation_stub,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--from-stage", default=None, choices=STAGE_ORDER)
    p.add_argument("--through-stage", required=True, choices=STAGE_ORDER)
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--phasea-root", default=str(DEFAULT_PHASEA_ROOT))
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    return p.parse_args()


def _mark_failed(run_id: str, state: dict, stage: str, reason: str, exit_code: int = 1) -> None:
    state["status"] = STATUS_FAILED
    state["current_stage"] = stage
    state["failure_or_gate_reason"] = reason
    state["exit_code"] = int(exit_code)
    state["next_allowed_stage"] = stage
    save_run_state(run_id, state)


def _complete_stage(run_id: str, state: dict, stage: str) -> None:
    completed = list(state.get("completed_stages") or [])
    if stage not in completed:
        completed.append(stage)
    state["completed_stages"] = completed
    state["current_stage"] = stage
    idx = stage_index(stage)
    state["next_allowed_stage"] = STAGE_ORDER[idx + 1] if idx + 1 < len(STAGE_ORDER) else None
    state["status"] = STATUS_RUNNING
    state["failure_or_gate_reason"] = None
    save_run_state(run_id, state)


def run_s0(run_id: str, root: Path, state: dict, args: argparse.Namespace) -> None:
    state["current_stage"] = "S0_INFRA"
    save_run_state(run_id, state)
    audit = audit_infra()
    write_s0_artifacts(root / "00_protocol", audit)
    ensure_wikitext2_shared_calibration(model_path=args.model_path)
    # Freeze protocol knobs into manifest
    manifest = {
        "run_id": run_id,
        "model_path": args.model_path,
        "phasea_root": str(Path(args.phasea_root).resolve()),
        "tp": 2,
        "kv_dtype": "bfloat16",
        "max_num_seqs": 1,
        "enforce_eager": True,
        "prefix_cache": False,
        "speculative_decoding": False,
        "calibration_sources": ["wikitext2", "s1k_original"],
        "prefix_lengths_j": [8, 32, 64, 128],
        "discovery": "4 wiki + 4 s1k",
        "holdout": "4 wiki + 4 s1k",
        "objective_source_ratio": "50:50",
        "ranking_metric": "P_layer_abs",
        "forbidden": [
            "semantic_replay",
            "mmlu_pro_for_discovery",
            "livecodebench_for_discovery",
            "e0_free_continuation",
            "lm_eval_mmlu_pro",
        ],
    }
    atomic_write_json(root / "run_manifest.json", manifest)
    write_variant_validation_stub(root / "70_variant_validation")
    _complete_stage(run_id, state, "S0_INFRA")


def run_s1(run_id: str, root: Path, state: dict, args: argparse.Namespace) -> None:
    state["current_stage"] = "S1_CAPTURE_STRUCTURAL"
    save_run_state(run_id, state)
    cohort = build_and_write_cohort(root / "00_protocol", model_path=args.model_path)
    cohort_path = Path(cohort["path"])
    cohort_rows = read_jsonl(cohort_path)
    causal_sample_ids = sorted({str(x["calibration_sample_id"]) for x in cohort_rows})
    build_objective_split_manifest(
        output_path=root / "00_protocol" / "objective_split_manifest.json",
        excluded_sample_ids=causal_sample_ids,
    )
    capture_root = root / "10_capture"
    for variant in ("E0", "E1"):
        capture_variant_states(
            variant=variant,
            cohort_path=cohort_path,
            output_root=capture_root / variant,
            model_path=args.model_path,
            phasea_root=Path(args.phasea_root),
            capture_level="core_qkv",
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    states = cohort_rows
    structural_rows = []
    for meta in states:
        key = meta["sample_key"]
        di = int(meta["decode_index"])
        e0_recs = load_rank_records(capture_root / "E0" / "hooks", "E0", key, 0)
        e1_recs = load_rank_records(capture_root / "E1" / "hooks", "E1", key, 0)
        # TP replica gate on both ranks
        e0_all = e0_recs + load_rank_records(capture_root / "E0" / "hooks", "E0", key, 1)
        e1_all = e1_recs + load_rank_records(capture_root / "E1" / "hooks", "E1", key, 1)
        for label, recs in (("E0", e0_all), ("E1", e1_all)):
            idx = index_capture_records(recs)
            replica = check_tp_replica(
                idx,
                sample_key=key,
                decode_index=di,
                boundaries=[
                    ("post_attn_norm", "updated_residual"),
                    ("o_proj", "tp_reduced"),
                    ("router_logits", "logits"),
                    ("moe_out", "tp_reduced"),
                    ("final_norm", "updated_residual"),
                    ("final_norm", "normalized"),
                ],
                layers=list(range(48)) + [None],
            )
            if not replica["pass"]:
                raise RuntimeError(f"TP replica gate failed for {label} {key}: {replica['failures'][:3]}")
            tensors = extract_layer_tensors(idx, sample_key=key, decode_index=di, rank=0)
            # Repeatability envelope: exact match required for offline BF16 reference vs capture
            # when no measured repeats yet — use 0/0 and let gate raise if kernels disagree.
            runtime_residual_closure_for_variant(
                tensors,
                repeatability_max_abs=0.0,
                repeatability_l2=0.0,
            )
        e0_logits = load_raw_logits(capture_root / "E0" / "raw_logits", "E0", key, di, 0)
        e1_logits = load_raw_logits(capture_root / "E1" / "raw_logits", "E1", key, di, 0)
        structural_rows.append(
            analyze_state_pair(
                sample_meta=meta,
                e0_records=e0_recs,
                e1_records=e1_recs,
                e0_logits=e0_logits,
                e1_logits=e1_logits,
            )
        )
    write_structural_reports(root / "20_structural", structural_rows)
    _complete_stage(run_id, state, "S1_CAPTURE_STRUCTURAL")


def run_s2(run_id: str, root: Path, state: dict, args: argparse.Namespace) -> None:
    state["current_stage"] = "S2_LAYER_CAUSAL"
    save_run_state(run_id, state)
    cohort_path = root / "00_protocol" / "internal_error_state_cohort.jsonl"
    result = run_discovery_whole_layer_scan(
        cohort_path=cohort_path,
        capture_root=root / "10_capture",
        output_dir=root / "50_protection",
        model_path=args.model_path,
        phasea_root=Path(args.phasea_root),
    )
    build_layer_selection_manifest(
        result["ranking"],
        root / "50_protection" / "layer_selection_manifest.json",
    )
    # Scientific gate: stop for review; do not auto-enter S3.
    state["completed_stages"] = list(state.get("completed_stages") or []) + (
        [] if "S2_LAYER_CAUSAL" in state.get("completed_stages", []) else ["S2_LAYER_CAUSAL"]
    )
    # de-dup
    seen = []
    for s in state["completed_stages"]:
        if s not in seen:
            seen.append(s)
    state["completed_stages"] = seen
    state["current_stage"] = "S2_LAYER_CAUSAL"
    state["status"] = STATUS_WAITING_REVIEW
    state["waiting_review_gate"] = "S2_LAYER_SELECTION_GATE"
    state["next_allowed_stage"] = "S3_SUBSTRUCTURE_CAUSAL"
    state["exit_code"] = 0
    state["failure_or_gate_reason"] = (
        "WAITING_REVIEW: freeze layer_selection_manifest.json after reviewing "
        "discovery P_layer_abs ranking before S3"
    )
    save_run_state(run_id, state)


def run_s3(run_id: str, root: Path, state: dict, args: argparse.Namespace) -> None:
    """Legacy partial-S3 (historical). Fresh science continues at S3_FULL48_CAUSAL."""
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.substructure_causal import (
        run_substructure_causal,
    )

    state["current_stage"] = "S3_SUBSTRUCTURE_CAUSAL"
    save_run_state(run_id, state)
    # If prior partial-S3 artifacts already exist, do not overwrite; treat as historical node.
    legacy_rows = root / "50_protection" / "s3_substructure_rows.jsonl"
    if legacy_rows.is_file() and legacy_rows.stat().st_size > 0:
        state["completed_stages"] = list(
            dict.fromkeys(list(state.get("completed_stages") or []) + ["S3_SUBSTRUCTURE_CAUSAL"])
        )
        state["status"] = STATUS_RUNNING
        state["waiting_review_gate"] = None
        state["next_allowed_stage"] = "S3_FULL48_CAUSAL"
        state["exit_code"] = None
        state["failure_or_gate_reason"] = (
            "legacy partial-S3 retained; proceeding to mandatory S3_FULL48_CAUSAL"
        )
        save_run_state(run_id, state)
        return

    run_substructure_causal(
        run_root=root,
        model_path=args.model_path,
        phasea_root=Path(args.phasea_root),
    )
    state["completed_stages"] = list(dict.fromkeys(list(state.get("completed_stages") or []) + ["S3_SUBSTRUCTURE_CAUSAL"]))
    state["status"] = STATUS_RUNNING
    state["waiting_review_gate"] = None
    state["next_allowed_stage"] = "S3_FULL48_CAUSAL"
    state["exit_code"] = None
    state["failure_or_gate_reason"] = None
    save_run_state(run_id, state)


def run_s3_full48(run_id: str, root: Path, state: dict, args: argparse.Namespace) -> None:
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.s3_full48_causal import (
        run_s3_full48_causal,
    )

    state["current_stage"] = "S3_FULL48_CAUSAL"
    save_run_state(run_id, state)
    run_s3_full48_causal(
        run_root=root,
        model_path=args.model_path,
        phasea_root=Path(args.phasea_root),
    )
    state["completed_stages"] = list(
        dict.fromkeys(list(state.get("completed_stages") or []) + ["S3_FULL48_CAUSAL"])
    )
    state["status"] = STATUS_WAITING_REVIEW
    state["waiting_review_gate"] = "S3_PROTECTION_OBJECTIVE_SCOPE_GATE"
    state["next_allowed_stage"] = "S4_PROTECTION_OBJECTIVE"
    state["exit_code"] = 0
    state["failure_or_gate_reason"] = (
        "WAITING_REVIEW: freeze practical_protection_scope.json and objective_scope.json "
        "(o3_layers, not enable_o3) before S4"
    )
    save_run_state(run_id, state)


def run_s4(run_id: str, root: Path, state: dict, args: argparse.Namespace) -> None:
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.protection_objective import (
        run_protection_and_objectives,
    )

    state["current_stage"] = "S4_PROTECTION_OBJECTIVE"
    save_run_state(run_id, state)
    run_protection_and_objectives(run_root=root, model_path=args.model_path, phasea_root=Path(args.phasea_root))
    state["completed_stages"] = list(dict.fromkeys(list(state.get("completed_stages") or []) + ["S4_PROTECTION_OBJECTIVE"]))
    state["status"] = STATUS_WAITING_REVIEW
    state["waiting_review_gate"] = "S4_OBJECTIVE_EXPANSION_GATE"
    state["next_allowed_stage"] = "S4_HOLDOUT_ACTUAL_VALIDATE"
    state["exit_code"] = 0
    state["failure_or_gate_reason"] = (
        "WAITING_REVIEW: decide whether O2/O3 warrant Top-K expansion before S5"
    )
    save_run_state(run_id, state)


def run_s4_holdout(run_id: str, root: Path, state: dict, args: argparse.Namespace) -> None:
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.objective_holdout import run_objective_holdout
    if 'S4_PROTECTION_OBJECTIVE' not in state['completed_stages']:
        raise RuntimeError('actual holdout requires completed S4 training')
    state.update(current_stage='S4_HOLDOUT_ACTUAL_VALIDATE', exit_code=None,
                 waiting_review_gate=None, failure_or_gate_reason=None,
                 next_allowed_stage='S4_HOLDOUT_ACTUAL_VALIDATE')
    save_run_state(run_id, state)
    run_objective_holdout(run_root=root, model_path=args.model_path, phasea_root=Path(args.phasea_root))
    _complete_stage(run_id, state, 'S4_HOLDOUT_ACTUAL_VALIDATE')
    state.update(status=STATUS_WAITING_REVIEW, waiting_review_gate='S4_OBJECTIVE_EXPANSION_GATE',
                 next_allowed_stage='S5_TOPK_STRUCTURAL_VALIDATE', exit_code=0,
                 failure_or_gate_reason='WAITING_REVIEW: independent actual-path objective comparison complete')
    save_run_state(run_id, state)


def run_s4_path_audit(run_id: str, root: Path, state: dict, args: argparse.Namespace) -> None:
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.path_audit import run_path_audit
    if 'S4_HOLDOUT_ACTUAL_VALIDATE' not in state['completed_stages']:
        raise RuntimeError('path audit requires completed actual holdout captures')
    state.update(current_stage='S4_PATH_MECHANISM_AUDIT', exit_code=None,
                 waiting_review_gate=None, failure_or_gate_reason=None,
                 next_allowed_stage='S4_PATH_MECHANISM_AUDIT')
    save_run_state(run_id, state)
    run_path_audit(run_root=root, model_path=args.model_path)
    _complete_stage(run_id, state, 'S4_PATH_MECHANISM_AUDIT')
    state.update(status=STATUS_WAITING_REVIEW, waiting_review_gate='S4_PATH_AUDIT_REVIEW',
                 next_allowed_stage=None, exit_code=0,
                 failure_or_gate_reason='WAITING_REVIEW: path diagnostics and exploratory mechanism review complete; legacy checkpoint semantics require review')
    save_run_state(run_id, state)


def run_corrected_alignment(run_id: str, root: Path, state: dict, args: argparse.Namespace) -> None:
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.phase_alignment import run_alignment
    if 'S4_PATH_MECHANISM_AUDIT' not in state['completed_stages']:
        raise RuntimeError('corrected alignment requires completed path audit')
    state.update(current_stage='S4_CORRECTED_PATH_ALIGNMENT', exit_code=None,
                 waiting_review_gate=None, failure_or_gate_reason=None,
                 next_allowed_stage='S4_CORRECTED_PATH_ALIGNMENT')
    save_run_state(run_id, state)
    gate = run_alignment(run_root=root, model_path=args.model_path, phasea_root=Path(args.phasea_root))
    _complete_stage(run_id, state, 'S4_CORRECTED_PATH_ALIGNMENT')
    state.update(status=STATUS_WAITING_REVIEW, waiting_review_gate='S4_CORRECTED_PATH_ALIGNMENT_GATE',
                 next_allowed_stage=None, exit_code=0,
                 failure_or_gate_reason=f'WAITING_REVIEW: corrected path alignment {gate["status"]}; no training or S5 started')
    save_run_state(run_id, state)


def run_arithmetic_alignment_stage(run_id: str, root: Path, state: dict, args: argparse.Namespace) -> None:
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.arithmetic_alignment import run_arithmetic_alignment
    if 'S4_CORRECTED_PATH_ALIGNMENT' not in state['completed_stages']:
        raise RuntimeError('arithmetic alignment requires completed production captures')
    stage = 'S4_ARITHMETIC_ALIGNMENT'
    state.update(current_stage=stage, exit_code=None, waiting_review_gate=None,
                 failure_or_gate_reason=None, next_allowed_stage=stage)
    save_run_state(run_id, state)
    run_arithmetic_alignment(run_root=root, model_path=args.model_path)
    _complete_stage(run_id, state, stage)
    state.update(status=STATUS_WAITING_REVIEW, waiting_review_gate=stage + '_GATE',
                 next_allowed_stage=None, exit_code=0,
                 failure_or_gate_reason='WAITING_REVIEW: arithmetic/history diagnostics complete; training gate remains BLOCKED')
    save_run_state(run_id, state)


def run_kernel_boundary_alignment_stage(run_id: str, root: Path, state: dict, args: argparse.Namespace) -> None:
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.kernel_boundary_alignment import run_kernel_alignment
    if 'S4_ARITHMETIC_ALIGNMENT' not in state['completed_stages']:
        raise RuntimeError('kernel boundary checks require arithmetic diagnostics')
    stage = 'S4_KERNEL_BOUNDARY_ALIGNMENT'
    state.update(current_stage=stage, exit_code=None, waiting_review_gate=None,
                 failure_or_gate_reason=None, next_allowed_stage=stage)
    save_run_state(run_id, state)
    run_kernel_alignment(run_root=root, model_path=args.model_path)
    _complete_stage(run_id, state, stage)
    state.update(status=STATUS_WAITING_REVIEW, waiting_review_gate=stage + '_GATE',
                 next_allowed_stage=None, exit_code=0,
                 failure_or_gate_reason='WAITING_REVIEW: production kernel boundary diagnostics complete; training gate remains BLOCKED')
    save_run_state(run_id, state)


def run_mechanism_prepare_stage(run_id: str, root: Path, state: dict, args: argparse.Namespace) -> None:
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.mechanism_prepare import run_prepare
    stage = 'S4_MECHANISM_PREPARE'
    state.update(current_stage=stage, exit_code=None, waiting_review_gate=None,
                 failure_or_gate_reason=None, next_allowed_stage=stage)
    save_run_state(run_id, state)
    run_prepare(run_root=root, model_path=args.model_path, phasea_root=Path(args.phasea_root))
    _complete_stage(run_id, state, stage)
    state.update(status=STATUS_WAITING_REVIEW, waiting_review_gate=stage + '_REVIEW',
                 next_allowed_stage=None, exit_code=0,
                 failure_or_gate_reason='WAITING_REVIEW: actual teacher captured; objective correctness checks remain; no S5')
    save_run_state(run_id, state)


def run_mechanism_validate_stage(run_id: str, root: Path, state: dict, args: argparse.Namespace) -> None:
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.mechanism_validation import run_validation
    stage='S4_MECHANISM_VALIDATE'
    state.update(current_stage=stage,exit_code=None,waiting_review_gate=None,
                 failure_or_gate_reason=None,next_allowed_stage=stage)
    save_run_state(run_id,state)
    run_validation(run_root=root,model_path=args.model_path,phasea_root=Path(args.phasea_root))
    _complete_stage(run_id,state,stage)
    state.update(status=STATUS_WAITING_REVIEW,waiting_review_gate=stage+'_RESULTS',
        next_allowed_stage=None,exit_code=0,
        failure_or_gate_reason='WAITING_REVIEW: inspect objective/folding evidence and proceed within authorized S4; no S5')
    save_run_state(run_id,state)


def run_s5(run_id: str, root: Path, state: dict, args: argparse.Namespace) -> None:
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.topk_validate import (
        run_topk_and_variant_validate,
    )

    if 'S4_HOLDOUT_ACTUAL_VALIDATE' not in state['completed_stages']:
        raise RuntimeError('S5 requires completed independent actual-path objective validation')
    state["current_stage"] = "S5_TOPK_STRUCTURAL_VALIDATE"
    save_run_state(run_id, state)
    run_topk_and_variant_validate(run_root=root, model_path=args.model_path, phasea_root=Path(args.phasea_root))
    _complete_stage(run_id, state, "S5_TOPK_STRUCTURAL_VALIDATE")


def run_s6(run_id: str, root: Path, state: dict, args: argparse.Namespace) -> None:
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.e2e_validate import (
        run_e2e_mmlu_pro,
    )

    state["current_stage"] = "S6_E2E_VALIDATE"
    save_run_state(run_id, state)
    run_e2e_mmlu_pro(run_root=root, model_path=args.model_path, phasea_root=Path(args.phasea_root))
    _complete_stage(run_id, state, "S6_E2E_VALIDATE")


def run_s7(run_id: str, root: Path, state: dict, args: argparse.Namespace) -> None:
    state["current_stage"] = "S7_REPORT"
    save_run_state(run_id, state)
    write_final_report(root, root / "analysis")
    state["status"] = STATUS_COMPLETED
    state["exit_code"] = 0
    _complete_stage(run_id, state, "S7_REPORT")
    state["status"] = STATUS_COMPLETED
    save_run_state(run_id, state)


STAGE_RUNNERS = {
    "S0_INFRA": run_s0,
    "S1_CAPTURE_STRUCTURAL": run_s1,
    "S2_LAYER_CAUSAL": run_s2,
    "S3_SUBSTRUCTURE_CAUSAL": run_s3,
    "S3_FULL48_CAUSAL": run_s3_full48,
    "S4_PROTECTION_OBJECTIVE": run_s4,
    "S4_HOLDOUT_ACTUAL_VALIDATE": run_s4_holdout,
    "S4_PATH_MECHANISM_AUDIT": run_s4_path_audit,
    "S4_CORRECTED_PATH_ALIGNMENT": run_corrected_alignment,
    "S4_ARITHMETIC_ALIGNMENT": run_arithmetic_alignment_stage,
    "S4_KERNEL_BOUNDARY_ALIGNMENT": run_kernel_boundary_alignment_stage,
    "S4_MECHANISM_PREPARE": run_mechanism_prepare_stage,
    "S4_MECHANISM_VALIDATE": run_mechanism_validate_stage,
    "S5_TOPK_STRUCTURAL_VALIDATE": run_s5,
    "S6_E2E_VALIDATE": run_s6,
    "S7_REPORT": run_s7,
}


def main() -> int:
    args = parse_args()
    run_id = str(args.run_id)
    root = ensure_run_dirs(run_id)
    state_path = root / "run_state.json"
    if state_path.is_file():
        state = load_run_state(run_id)
        if state.get("status") == STATUS_RUNNING and state.get("pid") and int(state["pid"]) != os.getpid():
            # Launcher should have blocked live duplicates; still refuse clobbering.
            from Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.run_state import (
                pid_alive,
            )

            if pid_alive(int(state["pid"])) and int(state["pid"]) != os.getpid():
                raise RuntimeError(f"run {run_id} still has live pid={state['pid']}")
    else:
        state = default_run_state(run_id, through_stage=args.through_stage, pid=os.getpid())
        save_run_state(run_id, state)

    pending_review = state.get('status') == STATUS_WAITING_REVIEW
    if pending_review and not args.from_stage:
        print('WAITING_REVIEW: explicit --from-stage required; no stage started')
        return 0
    state["pid"] = os.getpid()
    state["through_stage"] = args.through_stage
    state["status"] = STATUS_RUNNING
    save_run_state(run_id, state)

    completed = set(state.get("completed_stages") or [])
    if args.from_stage:
        start = args.from_stage
    elif completed:
        # resume after last completed
        last = max(completed, key=stage_index)
        nxt = stage_index(last) + 1
        if nxt >= len(STAGE_ORDER):
            state["status"] = STATUS_COMPLETED
            state["exit_code"] = 0
            save_run_state(run_id, state)
            return 0
        start = STAGE_ORDER[nxt]
    else:
        start = STAGE_ORDER[0]

    planned = stages_inclusive(start, args.through_stage)
    try:
        for stage in planned:
            if stage in completed:
                continue
            # Do not auto-cross scientific review gates unless --from-stage is explicit.
            review_gated = {
                "S3_SUBSTRUCTURE_CAUSAL": "S2_LAYER_SELECTION_GATE",
                "S4_PROTECTION_OBJECTIVE": "S3_PROTECTION_OBJECTIVE_SCOPE_GATE",
                "S5_TOPK_STRUCTURAL_VALIDATE": "S4_OBJECTIVE_EXPANSION_GATE",
            }
            if (
                stage in review_gated
                and state.get("status") == STATUS_WAITING_REVIEW
                and not args.from_stage
            ):
                break
            STAGE_RUNNERS[stage](run_id, root, state, args)
            state = load_run_state(run_id)
            if state.get("status") == STATUS_WAITING_REVIEW:
                return 0
            if state.get("status") == STATUS_FAILED:
                return int(state.get("exit_code") or 1)
            completed = set(state.get("completed_stages") or [])
        return 0
    except Exception as exc:  # noqa: BLE001 - persist failure then re-raise for non-zero exit
        reason = f"{type(exc).__name__}: {exc}"
        tb = traceback.format_exc()
        (root / "logs" / "pipeline_failure.txt").write_text(tb, encoding="utf-8")
        _mark_failed(run_id, state, state.get("current_stage") or start, reason)
        print(tb, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
