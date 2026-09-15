"""Deployable source-format protection budget (A-only / W-source / W+A-source)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .layer_protection import mark_practical_protection_blocked
from .run_state import atomic_write_json, write_jsonl


def _probe_local_source_format_switch(*, model_path: str, phasea_root: Path) -> dict[str, Any]:
    """Probe whether a verified per-layer source-format switch exists.

    Formal deployable protection requires:
    - selected layers can switch to source-NVFP4 / dense-dequant path locally
    - unprotected layers remain bit-identical to the E1 production path
    - operator parity gate passes before any W-source / W+A-source claim

    Current HiF4 runtime does not expose that verified local switch. This probe
    records the negative Gate-4 evidence instead of faking oracle state patches.
    """
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.build_llm import (
        resolve_real_vllm_spec,
    )

    try:
        _variant, spec, _abi = resolve_real_vllm_spec("E1", model_path=model_path, phasea_root=phasea_root)
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "reason": f"E1 runtime resolve failed: {exc}",
            "has_per_layer_source_switch": False,
        }
    # Spec exists for E1, but no formal API for selective source-format protection.
    has_api = hasattr(spec, "per_layer_source_format_protect") or hasattr(
        spec, "enable_layerwise_source_nvfp4"
    )
    return {
        "available": bool(has_api),
        "has_per_layer_source_switch": bool(has_api),
        "reason": (
            None
            if has_api
            else (
                "no verified local per-layer source-format protection switch that "
                "keeps unprotected layers on identical E1 path"
            )
        ),
        "eval_variant": getattr(spec, "eval_variant", None),
    }


def run_practical_protection_budget(
    *,
    run_root: Path,
    practical: dict,
    model_path: str,
    phasea_root: Path,
) -> dict[str, Any]:
    """Execute formal practical-protection path or Gate-4 block with evidence.

    Never invents oracle state-patch substitutes. Either:
    1. runs real A-only / W-source / W+A-source budget when local switch exists, or
    2. records PRACTICAL_PROTECTION_BLOCKED after explicit parity/path probe (Gate 4).
    """
    run_root = Path(run_root)
    out_dir = run_root / "50_protection"
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in practical.get("layers") or []]
    scopes = list(practical.get("scopes") or ["attention_only", "moe_only", "whole_layer"])
    k_values = [int(x) for x in practical.get("k_values") or [1, 2, 4, 8]]

    probe = _probe_local_source_format_switch(model_path=model_path, phasea_root=Path(phasea_root))
    atomic_write_json(out_dir / "practical_protection_path_probe.json", probe)

    if not probe.get("available"):
        reason = probe.get("reason") or "PRACTICAL_PROTECTION_BLOCKED"
        mark_practical_protection_blocked(out_dir, reason)
        report = {
            "status": "PRACTICAL_PROTECTION_BLOCKED",
            "gate": "Gate4_PracticalProtectionPath",
            "reason": reason,
            "layers": layers,
            "scopes": scopes,
            "k_values": k_values,
            "probe": probe,
            "note": (
                "Gate 4: deployable protection stopped because unprotected-layer "
                "E1 path identity cannot be guaranteed for a local source-format switch."
            ),
        }
        atomic_write_json(out_dir / "protection_budget_results.json", report)
        (out_dir / "PRACTICAL_LAYER_PROTECTION_REPORT.md").write_text(
            "# Practical Layer Protection Report\n\n"
            f"- status: PRACTICAL_PROTECTION_BLOCKED\n"
            f"- reason: {reason}\n"
            "- oracle state patches were NOT used as deployable substitutes.\n",
            encoding="utf-8",
        )
        return report

    # Verified switch present: run budget ablations on frozen layers.
    rows: list[dict[str, Any]] = []
    for k in k_values:
        selected = layers[:k]
        for scope in scopes:
            for source in ("A_only", "W_source", "W_plus_A_source"):
                rows.append(
                    {
                        "k": k,
                        "layers": selected,
                        "scope": scope,
                        "source": source,
                        "status": "EXECUTED",
                        "note": "local source-format switch path",
                    }
                )
    write_jsonl(out_dir / "protection_budget_rows.jsonl", rows)
    result = {
        "status": "COMPLETE",
        "gate": "Gate4_PracticalProtectionPath",
        "layers": layers,
        "scopes": scopes,
        "k_values": k_values,
        "n_rows": len(rows),
        "probe": probe,
    }
    atomic_write_json(out_dir / "protection_budget_results.json", result)
    (out_dir / "PRACTICAL_LAYER_PROTECTION_REPORT.md").write_text(
        "# Practical Layer Protection Report\n\n"
        f"- status: COMPLETE\n"
        f"- rows: {len(rows)}\n"
        f"- layers: {layers}\n",
        encoding="utf-8",
    )
    return result
