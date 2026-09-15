"""S3 substructure causal stage: Attention/MoE, conditional QKVO, MoE q/p, Router."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from vllm.inputs import TokensPrompt

from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.build_llm import (
    build_real_vllm,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.format_conversion.production_puncture import (
    ProductionPunctureOp,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.forced_trajectory import (
    make_forced_sampling_params,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.hook_spec import (
    build_probe_map,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.worker_hooks import (
    BeginSampleOp,
    InstallHooksOp,
    flush_sample,
    remove_hooks,
)

from .capture_states import load_rank_records, load_raw_logits
from .config import DEFAULT_MODEL_PATH, DEFAULT_PHASEA_ROOT
from .layer_protection import _generate_with_intervention, _tensor_from_records
from .math_utils import exact_kl
from .moe_decomposition import moe_three_way_decompose
from .one_step_intervention import FlushCausalInterventionOp, InstallCausalInterventionOp
from .residual_ledger import extract_layer_tensors, index_capture_records
from .router_contribution import conditional_enable_o3, router_basic_metrics, write_router_report
from .run_state import atomic_write_json, read_json, read_jsonl, write_jsonl


def _require_frozen_selection(path: Path) -> dict:
    payload = read_json(path)
    if payload.get("status") != "FROZEN":
        raise RuntimeError(
            f"layer_selection_manifest must be FROZEN before S3, got status={payload.get('status')}"
        )
    return payload


def run_substructure_causal(
    *,
    run_root: Path,
    model_path: str = DEFAULT_MODEL_PATH,
    phasea_root: Path = DEFAULT_PHASEA_ROOT,
) -> dict[str, Any]:
    run_root = Path(run_root)
    selection = _require_frozen_selection(run_root / "50_protection" / "layer_selection_manifest.json")
    layers = sorted(
        set(selection["top_sensitive"])
        | set(selection.get("neutral_controls") or [])
        | set(selection.get("cancellation_controls") or [])
    )
    cohort = read_jsonl(run_root / "00_protocol" / "internal_error_state_cohort.jsonl")
    holdout = [s for s in cohort if s["split"] == "holdout"]
    discovery = [s for s in cohort if s["split"] == "discovery"]
    capture_root = run_root / "10_capture"
    out_dir = run_root / "50_protection"
    moe_dir = run_root / "30_moe_qp"
    router_dir = run_root / "40_router"
    moe_dir.mkdir(parents=True, exist_ok=True)
    router_dir.mkdir(parents=True, exist_ok=True)

    llm, _ = build_real_vllm(
        "E1",
        model_path=model_path,
        phasea_root=Path(phasea_root),
        max_num_seqs=1,
        gpu_memory_utilization=0.90,
    )
    llm.apply_model(InstallHooksOp("E1", str(out_dir / "s3_hooks"), "core_qkv"))
    rows = []
    attn_sensitive = []
    router_interv = []
    causal_rows_path = out_dir / "s3_substructure_rows.jsonl"
    router_partial_path = router_dir / "router_intervention_rows.partial.jsonl"
    try:
        for state in holdout + discovery:
            # holdout confirmation on selected layers; discovery used for MoE qp on selected
            key = state["sample_key"]
            di = int(state["decode_index"])
            e0_recs = load_rank_records(capture_root / "E0" / "hooks", "E0", key, 0)
            e1_recs = load_rank_records(capture_root / "E1" / "hooks", "E1", key, 0)
            e0_logits = load_raw_logits(capture_root / "E0" / "raw_logits", "E0", key, di, 0)
            e1_logits = load_raw_logits(capture_root / "E1" / "raw_logits", "E1", key, di, 0)
            kl_base = exact_kl(e0_logits, e1_logits)
            for layer in layers:
                if state["split"] != "holdout" and state["split"] != "discovery":
                    continue
                # Level-2 Attention / MoE
                for kind, attn, moe in (
                    ("attn_only", True, False),
                    ("moe_only", False, True),
                    ("whole_layer", True, True),
                ):
                    payload = {
                        "sample_key": key,
                        "layer": layer,
                        "decode_index": di,
                        "abs_position": int(state["abs_position"]),
                        "repair_attn": _tensor_from_records(
                            e0_recs, sample_key=key, decode_index=di, layer=layer, boundary="o_proj", role="tp_reduced"
                        )
                        if attn
                        else None,
                        "repair_moe": _tensor_from_records(
                            e0_recs, sample_key=key, decode_index=di, layer=layer, boundary="moe_out", role="tp_reduced"
                        )
                        if moe
                        else None,
                        "mode": "repair",
                    }
                    logits = _generate_with_intervention(
                        llm, state, "E1", out_dir / "s3_logits", InstallCausalInterventionOp(kind, payload)
                    )
                    p = kl_base - exact_kl(e0_logits, logits)
                    rows.append(
                        {
                            "sample_key": key,
                            "split": state["split"],
                            "layer": layer,
                            "kind": kind,
                            "P": p,
                            "kl_base": kl_base,
                        }
                    )
                p_a = next(r["P"] for r in rows if r["sample_key"] == key and r["layer"] == layer and r["kind"] == "attn_only")
                p_m = next(r["P"] for r in rows if r["sample_key"] == key and r["layer"] == layer and r["kind"] == "moe_only")
                p_l = next(r["P"] for r in rows if r["sample_key"] == key and r["layer"] == layer and r["kind"] == "whole_layer")
                rows.append(
                    {
                        "sample_key": key,
                        "split": state["split"],
                        "layer": layer,
                        "kind": "interaction",
                        "P": p_l - p_a - p_m,
                        "kl_base": kl_base,
                    }
                )
                if p_a > 0 and p_a >= p_m:
                    attn_sensitive.append(layer)
                    for which in ("Q", "K", "V"):
                        e0_qkv = _tensor_from_records(
                            e0_recs, sample_key=key, decode_index=di, layer=layer, boundary="qkv_proj", role="rank_local"
                        )
                        logits = _generate_with_intervention(
                            llm,
                            state,
                            "E1",
                            out_dir / "s3_logits",
                            InstallCausalInterventionOp(
                                "qkv_slice",
                                {
                                    "sample_key": key,
                                    "layer": layer,
                                    "abs_position": int(state["abs_position"]),
                                    "which": which,
                                    "source_qkv": e0_qkv,
                                    "mode": "repair",
                                },
                            ),
                        )
                        rows.append(
                            {
                                "sample_key": key,
                                "split": state["split"],
                                "layer": layer,
                                "kind": f"{which}_only",
                                "P": kl_base - exact_kl(e0_logits, logits),
                                "kl_base": kl_base,
                            }
                        )
                    # O-only
                    logits = _generate_with_intervention(
                        llm,
                        state,
                        "E1",
                        out_dir / "s3_logits",
                        InstallCausalInterventionOp(
                            "attn_only",
                            {
                                "sample_key": key,
                                "layer": layer,
                                "decode_index": di,
                                "abs_position": int(state["abs_position"]),
                                "repair_attn": _tensor_from_records(
                                    e0_recs, sample_key=key, decode_index=di, layer=layer, boundary="o_proj", role="tp_reduced"
                                ),
                                "repair_moe": None,
                                "mode": "repair",
                            },
                        ),
                    )
                    rows.append(
                        {
                            "sample_key": key,
                            "split": state["split"],
                            "layer": layer,
                            "kind": "O_only",
                            "P": kl_base - exact_kl(e0_logits, logits),
                            "kl_base": kl_base,
                        }
                    )
                # Router freeze causal on selected layers (discovery+holdout)
                r0 = _tensor_from_records(
                    e0_recs, sample_key=key, decode_index=di, layer=layer, boundary="router_logits", role="logits"
                )
                r1 = _tensor_from_records(
                    e1_recs, sample_key=key, decode_index=di, layer=layer, boundary="router_logits", role="logits"
                )
                basic = router_basic_metrics(r0, r1)
                logits = _generate_with_intervention(
                    llm,
                    state,
                    "E1",
                    out_dir / "s3_logits",
                    InstallCausalInterventionOp(
                        "router_freeze",
                        {
                            "sample_key": key,
                            "layer": layer,
                            "abs_position": int(state["abs_position"]),
                            "router_logits": r0,
                        },
                    ),
                )
                c_router = kl_base - exact_kl(e0_logits, logits)
                router_interv.append(
                    {
                        "sample_key": key,
                        "layer": layer,
                        "C_router": c_router,
                        **basic,
                    }
                )
                # Persist after each layer so puncture/engine failures do not erase causal work.
                write_jsonl(causal_rows_path, rows)
                write_jsonl(router_partial_path, router_interv)
    finally:
        try:
            llm.apply_model(remove_hooks)
        except Exception:
            pass
        del llm

    # MoE production puncture MUST use the matching loaded variant.
    # Running E0 identity on an E1 engine is scientifically invalid and always fails.
    requests = []
    for state in discovery:
        for layer in layers:
            requests.append(
                {
                    "sample_key": state["sample_key"],
                    "decode_index": int(state["decode_index"]),
                    "layer": int(layer),
                    "operator": "moe",
                }
            )
    puncture_summary: dict[str, Any] = {"requests": len(requests), "e0_pass_keys": [], "e0_blocked_keys": [], "e1": None}
    if requests:
        e0_out = moe_dir / "puncture_E0"
        e1_out = moe_dir / "puncture_E1"
        # Fresh E0 engine for baseline identity + frozen-router identity.
        e0_llm, _ = build_real_vllm(
            "E0",
            model_path=model_path,
            phasea_root=Path(phasea_root),
            max_num_seqs=1,
            gpu_memory_utilization=0.90,
        )
        try:
            e0_replies = e0_llm.apply_model(
                ProductionPunctureOp(
                    requests=requests,
                    reference_root=str(capture_root / "E0" / "hooks" / "E0"),
                    actual_root=str(capture_root / "E0" / "hooks" / "E0"),
                    output_root=str(e0_out),
                    variant="E0",
                    canonical_history_verified=True,
                )
            )
        finally:
            del e0_llm
        # Soft-gate per plan Gate 2: failed layer/state is excluded from q/p, not a hard crash.
        passed_requests = []
        blocked_keys = []
        for sample_dir in sorted(e0_out.glob("*")):
            if not sample_dir.is_dir():
                continue
            payload = torch.load(sample_dir / "rank0.pt", map_location="cpu", weights_only=False)
            for rec in payload["records"]:
                key = tuple(rec["key"]) if isinstance(rec.get("key"), (list, tuple)) else rec.get("key")
                if rec.get("status") == "PASS" and rec.get("frozen_router_status") != "FROZEN_ROUTER_BLOCKED":
                    # Prefer explicit frozen_router_identity pass when present.
                    fri = rec.get("frozen_router_identity") or {}
                    if fri and not fri.get("both_tp_ranks_pass", False):
                        blocked_keys.append(key)
                        continue
                    passed_requests.append(
                        {
                            "sample_key": rec["sample_key"],
                            "decode_index": int(rec["decode_index"]),
                            "layer": int(rec["layer"]),
                            "operator": "moe",
                        }
                    )
                    puncture_summary["e0_pass_keys"].append(key)
                else:
                    blocked_keys.append(key)
        puncture_summary["e0_blocked_keys"] = blocked_keys
        puncture_summary["e0_replies"] = e0_replies
        atomic_write_json(moe_dir / "puncture_E0_summary.json", puncture_summary)

        if passed_requests:
            e1_llm, _ = build_real_vllm(
                "E1",
                model_path=model_path,
                phasea_root=Path(phasea_root),
                max_num_seqs=1,
                gpu_memory_utilization=0.90,
            )
            try:
                e1_replies = e1_llm.apply_model(
                    ProductionPunctureOp(
                        requests=passed_requests,
                        reference_root=str(capture_root / "E0" / "hooks" / "E0"),
                        actual_root=str(capture_root / "E1" / "hooks" / "E1"),
                        output_root=str(e1_out),
                        variant="E1",
                        canonical_history_verified=True,
                        baseline_puncture_root=str(e0_out),
                    )
                )
            finally:
                del e1_llm
            puncture_summary["e1"] = e1_replies
            atomic_write_json(moe_dir / "puncture_replies.json", puncture_summary)
        else:
            atomic_write_json(
                moe_dir / "puncture_replies.json",
                {
                    **puncture_summary,
                    "note": "No E0 MoE puncture identities passed; q/p conclusions omitted (Gate 2).",
                },
            )

    write_jsonl(out_dir / "s3_substructure_rows.jsonl", rows)
    write_router_report(router_dir, router_interv, router_interv)
    enable_o3 = conditional_enable_o3(router_interv)
    # Aggregate Attention vs MoE dominance per layer
    causal_source = {}
    for layer in layers:
        pa = [r["P"] for r in rows if r["layer"] == layer and r["kind"] == "attn_only"]
        pm = [r["P"] for r in rows if r["layer"] == layer and r["kind"] == "moe_only"]
        if not pa or not pm:
            continue
        ma, mm = sum(pa) / len(pa), sum(pm) / len(pm)
        if ma > 0 and ma >= mm:
            causal_source[layer] = "attention" if ma > mm * 1.05 else "both"
        elif mm > 0 and mm > ma:
            causal_source[layer] = "moe" if mm > ma * 1.05 else "both"
        else:
            causal_source[layer] = "both" if (ma > 0 or mm > 0) else "moe"
    objective_scope = {
        "status": "DRAFT_PENDING_REVIEW",
        "causal_source_by_layer": {str(k): v for k, v in causal_source.items()},
        "enable_o3": enable_o3,
        "o4_layers": selection.get("top_sensitive", [])[:4],
    }
    practical_scope = {
        "status": "DRAFT_PENDING_REVIEW",
        "layers": selection.get("top_sensitive", [])[:4],
        "scopes": ["attention_only", "moe_only", "whole_layer"],
        "attn_sensitive_layers": sorted(set(attn_sensitive)),
    }
    atomic_write_json(run_root / "60_objective" / "objective_scope.json", objective_scope)
    atomic_write_json(run_root / "50_protection" / "practical_protection_scope.json", practical_scope)
    (out_dir / "LAYER_PROTECTION_CAUSAL_REPORT.md").write_text(
        "# Layer Protection Causal Report (S3)\n\n"
        f"- rows: {len(rows)}\n"
        f"- causal_source_by_layer: {causal_source}\n"
        f"- enable_o3: {enable_o3}\n"
        f"- moe_puncture_e0_pass: {len(puncture_summary.get('e0_pass_keys', []))}\n"
        f"- moe_puncture_e0_blocked: {len(puncture_summary.get('e0_blocked_keys', []))}\n",
        encoding="utf-8",
    )
    (moe_dir / "MOE_ERROR_DECOMPOSITION_REPORT.md").write_text(
        "# MoE Error Decomposition\n\n"
        "E0/E1 production puncture run on matching loaded variants. "
        "Blocked identities are excluded from q/p conclusions (Gate 2).\n"
        f"- pass_keys: {len(puncture_summary.get('e0_pass_keys', []))}\n"
        f"- blocked_keys: {len(puncture_summary.get('e0_blocked_keys', []))}\n",
        encoding="utf-8",
    )
    return {
        "n_rows": len(rows),
        "enable_o3": enable_o3,
        "causal_source_by_layer": causal_source,
        "puncture": puncture_summary,
    }
