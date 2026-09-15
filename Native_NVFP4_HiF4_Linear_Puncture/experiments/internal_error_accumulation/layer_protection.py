"""Layer causal protection screening and practical protection scaffolding."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
from vllm.inputs import TokensPrompt

from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.build_llm import (
    build_real_vllm,
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
from .config import DEFAULT_MODEL_PATH, DEFAULT_PHASEA_ROOT, NUM_LAYERS
from .math_utils import exact_kl
from .one_step_intervention import (
    FlushCausalInterventionOp,
    ForceClearSampleOp,
    InstallCausalInterventionOp,
)
from .residual_ledger import extract_layer_tensors, index_capture_records
from .run_state import atomic_write_json, read_jsonl, write_jsonl


def _tensor_from_records(records: list[dict], *, sample_key: str, decode_index: int, layer: int, boundary: str, role: str, rank: int = 0) -> torch.Tensor:
    idx = index_capture_records(records)
    return idx[(sample_key, decode_index, layer, boundary, role, rank)]


def _generate_with_intervention(llm, state: dict, variant: str, logits_root: Path, intervention: InstallCausalInterventionOp | None):
    prompt = [int(x) for x in state["prompt_token_ids"]]
    forced = [int(x) for x in state["forced_token_ids"]]
    decode_index = int(state["decode_index"])
    probe_map = build_probe_map(len(prompt), [{"decode_index": decode_index}])
    try:
        llm.apply_model(BeginSampleOp(state["sample_key"], len(prompt), probe_map))
        if intervention is not None:
            llm.apply_model(intervention)
        params = make_forced_sampling_params(
            forced,
            max_tokens=len(forced),
            sample_key=state["sample_key"] + "_interv",
            variant=variant,
            probe_decode_indices=[decode_index],
            logits_root=str(logits_root),
        )
        outputs = llm.generate([TokensPrompt(prompt_token_ids=prompt)], [params], use_tqdm=False)
        generated = [int(x) for x in outputs[0].outputs[0].token_ids]
        if generated != forced:
            raise RuntimeError(f"intervention forced mismatch for {state['sample_key']}")
        if intervention is not None:
            llm.apply_model(FlushCausalInterventionOp())
        # Always release BeginSample ownership; intervention flush alone does not.
        llm.apply_model(flush_sample)
        logits = load_raw_logits(logits_root, variant, state["sample_key"] + "_interv", decode_index, rank=0)
        return logits
    except Exception:
        llm.apply_model(ForceClearSampleOp())
        raise


def run_discovery_whole_layer_scan(
    *,
    cohort_path: Path,
    capture_root: Path,
    output_dir: Path,
    model_path: str = DEFAULT_MODEL_PATH,
    phasea_root: Path = DEFAULT_PHASEA_ROOT,
) -> dict[str, Any]:
    """S2: discovery 8×4×48 whole-layer oracle repair scan. Ranking by P_layer_abs."""
    states = [s for s in read_jsonl(cohort_path) if s["split"] == "discovery"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logits_root = output_dir / "intervention_logits"
    hooks_tmp = output_dir / "interv_hooks"
    llm, _ = build_real_vllm("E1", model_path=model_path, phasea_root=Path(phasea_root), max_num_seqs=1)
    llm.apply_model(InstallHooksOp("E1", str(hooks_tmp), "core"))
    rows = []
    try:
        for state in states:
            key = state["sample_key"]
            decode_index = int(state["decode_index"])
            e0_recs = load_rank_records(capture_root / "E0" / "hooks", "E0", key, 0)
            e1_recs = load_rank_records(capture_root / "E1" / "hooks", "E1", key, 0)
            # Also need rank-local tensors? o_proj/moe_out are tp_reduced global.
            e0_logits = load_raw_logits(capture_root / "E0" / "raw_logits", "E0", key, decode_index, 0)
            e1_logits = load_raw_logits(capture_root / "E1" / "raw_logits", "E1", key, decode_index, 0)
            kl_base = exact_kl(e0_logits, e1_logits)
            # whole-layer no-op identity once per state
            noop = InstallCausalInterventionOp(
                "whole_layer",
                {
                    "sample_key": key,
                    "layer": 0,
                    "decode_index": decode_index,
                    "abs_position": int(state["abs_position"]),
                    "repair_attn": _tensor_from_records(e1_recs, sample_key=key, decode_index=decode_index, layer=0, boundary="o_proj", role="tp_reduced"),
                    "repair_moe": _tensor_from_records(e1_recs, sample_key=key, decode_index=decode_index, layer=0, boundary="moe_out", role="tp_reduced"),
                    "mode": "noop",
                },
            )
            noop_logits = _generate_with_intervention(llm, state, "E1", logits_root, noop)
            # No-op must fall in exact equality or we'll compare later against measured envelope in formal gate.
            noop_delta = (noop_logits.double().reshape(-1) - e1_logits.double().reshape(-1)).abs().max().item()
            for layer in range(NUM_LAYERS):
                repair = InstallCausalInterventionOp(
                    "whole_layer",
                    {
                        "sample_key": key,
                        "layer": layer,
                        "decode_index": decode_index,
                        "abs_position": int(state["abs_position"]),
                        "repair_attn": _tensor_from_records(e0_recs, sample_key=key, decode_index=decode_index, layer=layer, boundary="o_proj", role="tp_reduced"),
                        "repair_moe": _tensor_from_records(e0_recs, sample_key=key, decode_index=decode_index, layer=layer, boundary="moe_out", role="tp_reduced"),
                        "mode": "repair",
                    },
                )
                repaired_logits = _generate_with_intervention(llm, state, "E1", logits_root, repair)
                kl_rep = exact_kl(e0_logits, repaired_logits)
                p_abs = kl_base - kl_rep
                p_frac = (p_abs / kl_base) if kl_base > 0 else None
                rows.append(
                    {
                        "sample_key": key,
                        "calibration_sample_id": state["calibration_sample_id"],
                        "source": state["source"],
                        "prefix_length_j": state["prefix_length_j"],
                        "layer": layer,
                        "kl_base": kl_base,
                        "kl_repaired": kl_rep,
                        "P_layer_abs": p_abs,
                        "P_layer_frac": p_frac,
                        "noop_max_abs": noop_delta,
                    }
                )
                write_jsonl(output_dir / "single_layer_protection_rows.jsonl", rows)
    finally:
        try:
            llm.apply_model(ForceClearSampleOp())
        except Exception:
            pass
        llm.apply_model(remove_hooks)

    # Aggregate: within sample mean over 4 states, then across samples mean -> ranking by P_layer_abs
    layer_scores = {l: [] for l in range(NUM_LAYERS)}
    samples = sorted({r["calibration_sample_id"] for r in rows})
    for sample_id in samples:
        for layer in range(NUM_LAYERS):
            vals = [r["P_layer_abs"] for r in rows if r["calibration_sample_id"] == sample_id and r["layer"] == layer]
            layer_scores[layer].append(sum(vals) / len(vals))
    ranking = []
    for layer, vals in layer_scores.items():
        ranking.append({"layer": layer, "P_layer_abs": sum(vals) / len(vals), "n_samples": len(vals)})
    ranking.sort(key=lambda x: x["P_layer_abs"], reverse=True)
    table_path = output_dir / "layer_importance_table.csv"
    with table_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["rank", "layer", "P_layer_abs", "n_samples"])
        writer.writeheader()
        for i, row in enumerate(ranking, start=1):
            writer.writerow({"rank": i, **row})
    atomic_write_json(
        output_dir / "discovery_layer_ranking.json",
        {"ranking_metric": "P_layer_abs", "ranking": ranking, "n_rows": len(rows)},
    )
    (output_dir / "LAYER_PROTECTION_CAUSAL_REPORT.md").write_text(
        "# Layer Protection Causal Report (Discovery whole-layer scan)\n\n"
        f"- interventions: {len(rows)}\n"
        f"- top4: {ranking[:4]}\n"
        "- primary sort: P_layer_abs\n"
        "- P_layer_frac is recorded per-state but not used for ranking\n",
        encoding="utf-8",
    )
    return {"n_rows": len(rows), "ranking": ranking}


def build_layer_selection_manifest(ranking: list[dict], output_path: Path) -> dict:
    """Draft selection for review after S2; frozen only after WAITING_REVIEW."""
    top = [r["layer"] for r in ranking[:8]]
    bottom = [r["layer"] for r in ranking[-4:]]
    mid = ranking[len(ranking) // 2]["layer"]
    payload = {
        "status": "DRAFT_PENDING_REVIEW",
        "ranking_metric": "P_layer_abs",
        "top_sensitive": top[:4],
        "top8": top,
        "neutral_controls": [mid],
        "cancellation_controls": bottom,
        "note": "Freeze only after human/Cursor review of S2 artifacts.",
    }
    atomic_write_json(output_path, payload)
    return payload


def mark_practical_protection_blocked(output_dir: Path, reason: str) -> None:
    atomic_write_json(
        Path(output_dir) / "PRACTICAL_PROTECTION_BLOCKED.json",
        {"status": "PRACTICAL_PROTECTION_BLOCKED", "reason": reason},
    )
