"""S3_FULL48_CAUSAL: 48-layer Attention / MoE / Router causal map + QKVO.

Isolated outputs under 50_protection/s3_full48/ and 40_router/s3_full48/.
Never overwrites legacy partial-S3 row artifacts.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

# Worker subprocesses execute this file directly; ensure repo root is importable.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.build_llm import (
    build_real_vllm,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.worker_hooks import (
    InstallHooksOp,
    remove_hooks,
)

from .capture_states import load_rank_records, load_raw_logits
from .config import DEFAULT_MODEL_PATH, DEFAULT_PHASEA_ROOT, NUM_LAYERS
from .layer_protection import _generate_with_intervention, _tensor_from_records
from .math_utils import exact_kl
from .one_step_intervention import ForceClearSampleOp, InstallCausalInterventionOp
from .router_contribution import decide_o3_layers
from .router_objective import router_proxy_metrics_from_logits
from .run_state import atomic_write_json, read_json, read_jsonl, write_jsonl


DISCOVERY_BASE_KINDS = ("attn_only", "moe_only", "router_freeze")
HOLDOUT_BASE_KINDS = ("attn_only", "moe_only", "whole_layer", "router_freeze")
QKVO_KINDS = ("Q_only", "K_only", "V_only", "O_only")
ROUTER_TOP_K = 8
EXPECTED_DISCOVERY_BASE = 32 * NUM_LAYERS * len(DISCOVERY_BASE_KINDS)  # 4608
EXPECTED_HOLDOUT_BASE = 32 * NUM_LAYERS * len(HOLDOUT_BASE_KINDS)  # 6144
EXPECTED_BASE_TOTAL = EXPECTED_DISCOVERY_BASE + EXPECTED_HOLDOUT_BASE  # 10752

ENV_WORKER_ID = "S3_FULL48_WORKER_ID"
ENV_NUM_WORKERS = "S3_FULL48_NUM_WORKERS"
ENV_GPU_PAIRS = "IEA_S3_GPU_PAIRS"
ENV_SHARD = "IEA_S3_SHARD"
DEFAULT_GPU_PAIRS = "4,5;6,7"


# ---------------------------------------------------------------------------
# Pure helpers (exported for unit tests)
# ---------------------------------------------------------------------------


def intervention_key(
    *,
    split: str,
    sample_key: str,
    decode_index: int,
    layer: int,
    kind: str,
) -> tuple[str, str, int, int, str]:
    return (str(split), str(sample_key), int(decode_index), int(layer), str(kind))


def key_from_row(row: dict[str, Any]) -> tuple[str, str, int, int, str]:
    di = row.get("decode_index", row.get("j", row.get("prefix_length_j")))
    return intervention_key(
        split=row["split"],
        sample_key=row["sample_key"],
        decode_index=int(di),
        layer=int(row["layer"]),
        kind=str(row["kind"]),
    )


def expected_base_keys(cohort: list[dict[str, Any]], *, layers: Iterable[int] | None = None) -> list[tuple]:
    layers = list(range(NUM_LAYERS) if layers is None else layers)
    keys: list[tuple] = []
    for state in cohort:
        split = state["split"]
        kinds = DISCOVERY_BASE_KINDS if split == "discovery" else HOLDOUT_BASE_KINDS
        if split not in {"discovery", "holdout"}:
            raise ValueError(f"unknown split {split}")
        for layer in layers:
            for kind in kinds:
                keys.append(
                    intervention_key(
                        split=split,
                        sample_key=state["sample_key"],
                        decode_index=int(state["decode_index"]),
                        layer=int(layer),
                        kind=kind,
                    )
                )
    return keys


def expected_discovery_count() -> int:
    return EXPECTED_DISCOVERY_BASE


def expected_holdout_count() -> int:
    return EXPECTED_HOLDOUT_BASE


def shard_owns_key(key: tuple, *, worker_id: int, num_workers: int) -> bool:
    if num_workers <= 1:
        return True
    digest = hashlib.md5(repr(key).encode("utf-8")).hexdigest()
    return int(digest, 16) % int(num_workers) == int(worker_id)


def load_completed_keys(rows: list[dict[str, Any]]) -> set[tuple]:
    done: set[tuple] = set()
    for row in rows:
        if row.get("identity_pass") is True and str(row.get("kind")) not in {"interaction"}:
            done.add(key_from_row(row))
    return done


def dedupe_rows_prefer_identity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one row per intervention key; prefer identity_pass=True."""
    best: dict[tuple, dict[str, Any]] = {}
    for row in rows:
        if str(row.get("kind")) == "interaction":
            continue
        key = key_from_row(row)
        prev = best.get(key)
        if prev is None:
            best[key] = row
            continue
        if row.get("identity_pass") and not prev.get("identity_pass"):
            best[key] = row
    return list(best.values())


def append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())


def load_jsonl_optional(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return []
    return read_jsonl(path)


def load_s2_p_layer_lookup(s2_rows_path: Path) -> dict[tuple[str, int], float]:
    rows = load_jsonl_optional(s2_rows_path)
    out: dict[tuple[str, int], float] = {}
    for row in rows:
        out[(str(row["sample_key"]), int(row["layer"]))] = float(row["P_layer_abs"])
    if not out:
        raise RuntimeError(f"empty S2 whole-layer artifact: {s2_rows_path}")
    return out


def sample_level_mean_by_layer(
    rows: list[dict[str, Any]],
    *,
    split: str,
    kind: str,
    value_field: str = "P",
) -> dict[int, list[float]]:
    """Within each sample average 4 states, return {layer: [8 sample-level values]}."""
    # (layer, calibration_sample_id) -> list of state values
    bucket: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("split") != split or row.get("kind") != kind:
            continue
        sid = str(row.get("calibration_sample_id") or row["sample_key"].rsplit("__j", 1)[0])
        bucket[(int(row["layer"]), sid)].append(float(row[value_field]))
    per_layer: dict[int, list[float]] = defaultdict(list)
    for (layer, _sid), vals in sorted(bucket.items()):
        if not vals:
            continue
        per_layer[layer].append(sum(vals) / len(vals))
    return dict(per_layer)


def classify_layer_causal_source(sample_pa: list[float], sample_pm: list[float]) -> str:
    """Holdout sample-level Attention/MoE classification (plan §16.3)."""
    if len(sample_pa) != len(sample_pm) or not sample_pa:
        raise ValueError("sample_pa/sample_pm must be non-empty and aligned")
    n = len(sample_pa)
    mean_pa = sum(sample_pa) / n
    mean_pm = sum(sample_pm) / n
    pos_a = sum(1 for v in sample_pa if v > 0)
    pos_m = sum(1 for v in sample_pm if v > 0)
    need = 5 if n >= 8 else max(1, (n + 1) // 2 + 1)  # formal n=8 → 5/8

    attn_stable = mean_pa > 0 and pos_a >= need
    moe_stable = mean_pm > 0 and pos_m >= need

    if attn_stable:
        if mean_pa > 1.05 * max(mean_pm, 0.0):
            return "attention"
        if mean_pm > 0 and max(mean_pa, mean_pm) > 0:
            rel = abs(mean_pa - mean_pm) / max(mean_pa, mean_pm)
            if rel <= 0.05:
                return "both"
    if moe_stable:
        if mean_pm > 1.05 * max(mean_pa, 0.0):
            return "moe"
        if mean_pa > 0 and max(mean_pa, mean_pm) > 0:
            rel = abs(mean_pa - mean_pm) / max(mean_pa, mean_pm)
            if rel <= 0.05:
                return "both"
    return "unstable/compensatory"


def attention_sensitive_layers_from_sources(causal_source_by_layer: dict[int, str]) -> list[int]:
    """Fixed layer set for QKVO — never per-state filtered."""
    return sorted(
        int(l) for l, src in causal_source_by_layer.items() if src in {"attention", "both"}
    )


def expected_qkvo_keys(
    cohort: list[dict[str, Any]],
    attention_layers: list[int],
) -> list[tuple]:
    keys: list[tuple] = []
    for state in cohort:
        if state["split"] not in {"discovery", "holdout"}:
            continue
        for layer in attention_layers:
            for kind in QKVO_KINDS:
                keys.append(
                    intervention_key(
                        split=state["split"],
                        sample_key=state["sample_key"],
                        decode_index=int(state["decode_index"]),
                        layer=int(layer),
                        kind=kind,
                    )
                )
    return keys


def spearman_rho(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None

    def _ranks(vals: list[float]) -> list[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(vals):
            j = i
            while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = 0.5 * (i + j) + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx, ry = _ranks(xs), _ranks(ys)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    deny = math.sqrt(sum((b - my) ** 2 for b in ry))
    if denx == 0.0 or deny == 0.0:
        return None
    return num / (denx * deny)


def topk_overlap(ranked_a: list[int], ranked_b: list[int], ks: tuple[int, ...] = (1, 2, 4, 8)) -> dict[str, float]:
    out = {}
    for k in ks:
        a = set(ranked_a[:k])
        b = set(ranked_b[:k])
        out[f"top{k}"] = len(a & b) / float(k)
    return out


def assert_base_completeness(rows: list[dict[str, Any]], cohort: list[dict[str, Any]]) -> dict[str, Any]:
    """Completeness Gate: no missing/duplicate base keys; 4608/6144; 48 layers; all kinds."""
    expected = expected_base_keys(cohort)
    expected_set = set(expected)
    seen: dict[tuple, int] = defaultdict(int)
    layers_seen: set[int] = set()
    kinds_by_split: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        kind = str(row["kind"])
        if kind in QKVO_KINDS or kind == "interaction":
            continue
        if kind not in set(DISCOVERY_BASE_KINDS) | set(HOLDOUT_BASE_KINDS):
            continue
        if row.get("identity_pass") is not True:
            continue
        key = key_from_row(row)
        seen[key] += 1
        layers_seen.add(int(row["layer"]))
        kinds_by_split[str(row["split"])].add(kind)

    missing = sorted(expected_set - set(seen))
    duplicates = sorted([k for k, c in seen.items() if c > 1])
    extra = sorted(set(seen) - expected_set)
    n_disc = sum(1 for k in seen if k[0] == "discovery")
    n_hold = sum(1 for k in seen if k[0] == "holdout")
    ok = (
        not missing
        and not duplicates
        and not extra
        and n_disc == EXPECTED_DISCOVERY_BASE
        and n_hold == EXPECTED_HOLDOUT_BASE
        and layers_seen == set(range(NUM_LAYERS))
        and kinds_by_split.get("discovery") == set(DISCOVERY_BASE_KINDS)
        and kinds_by_split.get("holdout") == set(HOLDOUT_BASE_KINDS)
    )
    report = {
        "ok": ok,
        "n_discovery": n_disc,
        "n_holdout": n_hold,
        "expected_discovery": EXPECTED_DISCOVERY_BASE,
        "expected_holdout": EXPECTED_HOLDOUT_BASE,
        "n_missing": len(missing),
        "n_duplicates": len(duplicates),
        "n_extra": len(extra),
        "layers_covered": sorted(layers_seen),
        "kinds_discovery": sorted(kinds_by_split.get("discovery") or []),
        "kinds_holdout": sorted(kinds_by_split.get("holdout") or []),
        "missing_preview": [list(k) for k in missing[:10]],
        "duplicate_preview": [list(k) for k in duplicates[:10]],
    }
    if not ok:
        raise RuntimeError(f"S3_FULL48 completeness gate failed: {json.dumps(report, ensure_ascii=False)}")
    return report


def aggregate_split_layer_summary(
    causal_rows: list[dict[str, Any]],
    router_rows: list[dict[str, Any]],
    *,
    split: str,
    s2_p_layer: dict[tuple[str, int], float] | None = None,
) -> dict[str, Any]:
    """Per-layer sample-level means for P_A/P_M/P_layer/I_A,M/C_router."""
    pa = sample_level_mean_by_layer(causal_rows, split=split, kind="attn_only")
    pm = sample_level_mean_by_layer(causal_rows, split=split, kind="moe_only")
    if split == "holdout":
        pl = sample_level_mean_by_layer(causal_rows, split=split, kind="whole_layer")
    else:
        # discovery whole-layer reused from S2 — rebuild sample-level from S2 lookup
        if s2_p_layer is None:
            raise RuntimeError("discovery aggregation requires S2 P_layer lookup")
        bucket: dict[tuple[int, str], list[float]] = defaultdict(list)
        for row in causal_rows:
            if row.get("split") != "discovery" or row.get("kind") != "attn_only":
                continue
            sid = str(row.get("calibration_sample_id") or row["sample_key"].rsplit("__j", 1)[0])
            key = (str(row["sample_key"]), int(row["layer"]))
            if key not in s2_p_layer:
                raise RuntimeError(f"missing S2 P_layer for {key}")
            bucket[(int(row["layer"]), sid)].append(float(s2_p_layer[key]))
        pl = defaultdict(list)
        for (layer, _sid), vals in sorted(bucket.items()):
            pl[layer].append(sum(vals) / len(vals))
        pl = dict(pl)

    cr = sample_level_mean_by_layer(router_rows, split=split, kind="router_freeze", value_field="C_router")
    layers = sorted(set(pa) | set(pm) | set(pl) | set(cr))
    by_layer = {}
    causal_source: dict[int, str] = {}
    for layer in layers:
        spa = pa.get(layer) or []
        spm = pm.get(layer) or []
        spl = pl.get(layer) or []
        scr = cr.get(layer) or []
        mean_pa = sum(spa) / len(spa) if spa else None
        mean_pm = sum(spm) / len(spm) if spm else None
        mean_pl = sum(spl) / len(spl) if spl else None
        mean_cr = sum(scr) / len(scr) if scr else None
        med_cr = float(sorted(scr)[len(scr) // 2]) if scr else None
        pos_cr = sum(1 for v in scr if v > 0) if scr else 0
        i_am = None
        if mean_pl is not None and mean_pa is not None and mean_pm is not None:
            i_am = mean_pl - mean_pa - mean_pm
        src = None
        if split == "holdout" and spa and spm:
            src = classify_layer_causal_source(spa, spm)
            causal_source[layer] = src
        by_layer[str(layer)] = {
            "layer": layer,
            "P_A": mean_pa,
            "P_M": mean_pm,
            "P_layer": mean_pl,
            "I_A_M": i_am,
            "C_router": mean_cr,
            "C_router_median": med_cr,
            "C_router_positive_count": pos_cr,
            "C_router_n": len(scr),
            "sample_level_P_A": spa,
            "sample_level_P_M": spm,
            "sample_level_C_router": scr,
            "causal_source": src,
        }
    return {"split": split, "by_layer": by_layer, "causal_source_by_layer": causal_source}


def select_s4_mechanism_layers(
    *,
    discovery_summary: dict[str, Any],
    holdout_summary: dict[str, Any],
    o3_layers: list[int],
) -> dict[str, Any]:
    """Deterministic cross-split selection for S4 mechanism experiments."""
    d_by = discovery_summary["by_layer"]
    h_by = holdout_summary["by_layer"]
    layers = list(range(NUM_LAYERS))

    def _rank(by: dict[str, Any]) -> dict[int, int]:
        ordered = sorted(
            layers,
            key=lambda l: (-float(by[str(l)]["P_layer"] or 0.0), l),
        )
        return {layer: i + 1 for i, layer in enumerate(ordered)}

    rd = _rank(d_by)
    rh = _rank(h_by)
    candidates: list[dict[str, Any]] = []
    for layer in layers:
        pd = float(d_by[str(layer)]["P_layer"] or 0.0)
        ph = float(h_by[str(layer)]["P_layer"] or 0.0)
        source = str(h_by[str(layer)].get("causal_source"))
        if pd <= 0 or ph <= 0 or source not in {"attention", "moe", "both"}:
            continue
        candidates.append(
            {
                "layer": layer,
                "causal_source": source,
                "P_layer_discovery": pd,
                "P_layer_holdout": ph,
                "rank_discovery": rd[layer],
                "rank_holdout": rh[layer],
                "cross_split_rank": (rd[layer] + rh[layer]) / 2.0,
                "C_router_holdout": float(h_by[str(layer)]["C_router"] or 0.0),
            }
        )

    def _top_for(source: str, n: int) -> list[int]:
        rows = [x for x in candidates if x["causal_source"] == source]
        rows.sort(
            key=lambda x: (
                float(x["cross_split_rank"]),
                int(x["rank_holdout"]),
                int(x["rank_discovery"]),
                int(x["layer"]),
            )
        )
        if len(rows) < n:
            raise RuntimeError(f"insufficient stable {source} layers for S4: need={n} got={len(rows)}")
        return [int(x["layer"]) for x in rows[:n]]

    attention_layers = _top_for("attention", 2)
    moe_layers = _top_for("moe", 2)
    main_layers = attention_layers + moe_layers

    eligible_o3 = {int(x) for x in o3_layers}
    router_rows = [
        x
        for x in candidates
        if int(x["layer"]) in eligible_o3 and x["causal_source"] in {"moe", "both"}
    ]
    router_rows.sort(
        key=lambda x: (-float(x["C_router_holdout"]), float(x["cross_split_rank"]), int(x["layer"]))
    )
    if len(router_rows) < 2:
        raise RuntimeError(f"insufficient stable Router-positive layers for O3 study: got={len(router_rows)}")
    router_objective_layers = [int(x["layer"]) for x in router_rows[:2]]

    selected_objective_layers = list(main_layers)
    for layer in router_objective_layers:
        if layer not in selected_objective_layers:
            selected_objective_layers.append(layer)

    return {
        "main_layers": main_layers,
        "attention_layers": attention_layers,
        "moe_layers": moe_layers,
        "router_objective_layers": router_objective_layers,
        "selected_objective_layers": selected_objective_layers,
        "stable_candidates": candidates,
        "rule": (
            "require P_layer>0 in discovery+holdout; rank all 48 independently in both splits; "
            "use mean rank; select Attention top2 + MoE top2. Router objective selects top2 "
            "holdout C_router within stable eligible o3_layers."
        ),
    }


def build_draft_scopes(
    *,
    discovery_summary: dict[str, Any],
    holdout_summary: dict[str, Any],
    selection: dict[str, Any] | None,
    o3_layers: list[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    causal = {int(k): v for k, v in (holdout_summary.get("causal_source_by_layer") or {}).items()}
    attn_layers = attention_sensitive_layers_from_sources(causal)
    s4 = select_s4_mechanism_layers(
        discovery_summary=discovery_summary,
        holdout_summary=holdout_summary,
        o3_layers=o3_layers,
    )
    main_layers = [int(x) for x in s4["main_layers"]]
    selected_objective_layers = [int(x) for x in s4["selected_objective_layers"]]
    practical_causal = {str(l): causal[l] for l in main_layers}
    selected_causal = {str(l): causal[l] for l in selected_objective_layers}
    prereg_top4 = [int(x) for x in list((selection or {}).get("top_sensitive") or [])[:4]]

    practical = {
        "status": "DRAFT_PENDING_REVIEW",
        "source": "S3_FULL48_CAUSAL_cross_split_review",
        "layers": main_layers,
        "scopes": ["attention_only", "moe_only", "whole_layer"],
        "attn_sensitive_layers": attn_layers,
        "causal_source_by_layer": practical_causal,
        "causal_source_by_layer_all": {str(k): v for k, v in causal.items()},
        "discovery_preregistered_top4": prereg_top4,
        "selection_rule": s4["rule"],
        "selection_detail": s4,
        "note": (
            "Causal holdout is used for S4 scope selection after discovery/holdout instability was observed; "
            "it is not reused as S4 objective validation."
        ),
    }
    objective = {
        "status": "DRAFT_PENDING_REVIEW",
        "source": "S3_FULL48_CAUSAL_cross_split_review",
        "selected_objective_layers": selected_objective_layers,
        "causal_source_by_layer": selected_causal,
        "causal_source_by_layer_all": {str(k): v for k, v in causal.items()},
        "discovery_preregistered_top4": prereg_top4,
        "o3_layers": list(o3_layers),
        "router_objective_layers": [int(x) for x in s4["router_objective_layers"]],
        "o3_objectives": ["O3_full", "O3_topk"],
        "o4_layers": main_layers,
        "forbidden": ["enable_o3", "O3_conditional"],
        "selection_rule": s4["rule"],
        "selection_detail": s4,
        "note": (
            "Only selected_objective_layers may emit recipes. o3_layers is the full eligibility gate; "
            "router_objective_layers is the focused O3 study subset."
        ),
    }
    return practical, objective


def archive_pre_full48_scope(path: Path) -> Path | None:
    """If path looks like pre-full48 DRAFT, copy to *.pre_full48.json once."""
    path = Path(path)
    if not path.is_file():
        return None
    archive = path.with_name(path.stem + ".pre_full48.json")
    if archive.is_file():
        return archive
    payload = read_json(path)
    accepted_sources = {"S3_FULL48_CAUSAL_holdout", "S3_FULL48_CAUSAL_cross_split_review"}
    if path.name.startswith("practical"):
        is_old = payload.get("source") not in accepted_sources
    else:
        is_old = (
            "enable_o3" in payload
            or "o3_layers" not in payload
            or payload.get("source") not in accepted_sources
        )
    if is_old:
        shutil.copy2(path, archive)
        return archive
    return None


# ---------------------------------------------------------------------------
# Intervention execution
# ---------------------------------------------------------------------------


def _noop_max_abs(llm, state: dict, e1_logits: torch.Tensor, logits_root: Path, intervention: InstallCausalInterventionOp) -> float:
    noop_logits = _generate_with_intervention(llm, state, "E1", logits_root, intervention)
    return float((noop_logits.double().reshape(-1) - e1_logits.double().reshape(-1)).abs().max().item())


def _run_one_base_intervention(
    *,
    llm,
    state: dict,
    layer: int,
    kind: str,
    e0_recs: list,
    e1_recs: list,
    e0_logits: torch.Tensor,
    e1_logits: torch.Tensor,
    kl_base: float,
    logits_root: Path,
    top_k: int = ROUTER_TOP_K,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    key = state["sample_key"]
    di = int(state["decode_index"])
    abs_pos = int(state["abs_position"])
    router_row = None

    if kind == "attn_only":
        e1_attn = _tensor_from_records(e1_recs, sample_key=key, decode_index=di, layer=layer, boundary="o_proj", role="tp_reduced")
        e0_attn = _tensor_from_records(e0_recs, sample_key=key, decode_index=di, layer=layer, boundary="o_proj", role="tp_reduced")
        noop = InstallCausalInterventionOp(
            "attn_only",
            {
                "sample_key": key,
                "layer": layer,
                "decode_index": di,
                "abs_position": abs_pos,
                "repair_attn": e1_attn,
                "repair_moe": None,
                "mode": "noop",
            },
        )
        noop_max = _noop_max_abs(llm, state, e1_logits, logits_root, noop)
        repair = InstallCausalInterventionOp(
            "attn_only",
            {
                "sample_key": key,
                "layer": layer,
                "decode_index": di,
                "abs_position": abs_pos,
                "repair_attn": e0_attn,
                "repair_moe": None,
                "mode": "repair",
            },
        )
    elif kind == "moe_only":
        e1_moe = _tensor_from_records(e1_recs, sample_key=key, decode_index=di, layer=layer, boundary="moe_out", role="tp_reduced")
        e0_moe = _tensor_from_records(e0_recs, sample_key=key, decode_index=di, layer=layer, boundary="moe_out", role="tp_reduced")
        noop = InstallCausalInterventionOp(
            "moe_only",
            {
                "sample_key": key,
                "layer": layer,
                "decode_index": di,
                "abs_position": abs_pos,
                "repair_attn": None,
                "repair_moe": e1_moe,
                "mode": "noop",
            },
        )
        noop_max = _noop_max_abs(llm, state, e1_logits, logits_root, noop)
        repair = InstallCausalInterventionOp(
            "moe_only",
            {
                "sample_key": key,
                "layer": layer,
                "decode_index": di,
                "abs_position": abs_pos,
                "repair_attn": None,
                "repair_moe": e0_moe,
                "mode": "repair",
            },
        )
    elif kind == "whole_layer":
        e1_attn = _tensor_from_records(e1_recs, sample_key=key, decode_index=di, layer=layer, boundary="o_proj", role="tp_reduced")
        e1_moe = _tensor_from_records(e1_recs, sample_key=key, decode_index=di, layer=layer, boundary="moe_out", role="tp_reduced")
        e0_attn = _tensor_from_records(e0_recs, sample_key=key, decode_index=di, layer=layer, boundary="o_proj", role="tp_reduced")
        e0_moe = _tensor_from_records(e0_recs, sample_key=key, decode_index=di, layer=layer, boundary="moe_out", role="tp_reduced")
        noop = InstallCausalInterventionOp(
            "whole_layer",
            {
                "sample_key": key,
                "layer": layer,
                "decode_index": di,
                "abs_position": abs_pos,
                "repair_attn": e1_attn,
                "repair_moe": e1_moe,
                "mode": "noop",
            },
        )
        noop_max = _noop_max_abs(llm, state, e1_logits, logits_root, noop)
        repair = InstallCausalInterventionOp(
            "whole_layer",
            {
                "sample_key": key,
                "layer": layer,
                "decode_index": di,
                "abs_position": abs_pos,
                "repair_attn": e0_attn,
                "repair_moe": e0_moe,
                "mode": "repair",
            },
        )
    elif kind == "router_freeze":
        r0 = _tensor_from_records(e0_recs, sample_key=key, decode_index=di, layer=layer, boundary="router_logits", role="logits")
        r1 = _tensor_from_records(e1_recs, sample_key=key, decode_index=di, layer=layer, boundary="router_logits", role="logits")
        proxies = router_proxy_metrics_from_logits(r0, r1, top_k=top_k, norm_topk_prob=True)
        noop = InstallCausalInterventionOp(
            "router_freeze",
            {"sample_key": key, "layer": layer, "abs_position": abs_pos, "router_logits": r1},
        )
        noop_max = _noop_max_abs(llm, state, e1_logits, logits_root, noop)
        repair = InstallCausalInterventionOp(
            "router_freeze",
            {"sample_key": key, "layer": layer, "abs_position": abs_pos, "router_logits": r0},
        )
    else:
        raise ValueError(f"unknown base kind {kind}")

    repaired = _generate_with_intervention(llm, state, "E1", logits_root, repair)
    kl_rep = exact_kl(e0_logits, repaired)
    p = kl_base - kl_rep
    identity_pass = noop_max == 0.0
    row = {
        "sample_key": key,
        "calibration_sample_id": state.get("calibration_sample_id"),
        "split": state["split"],
        "decode_index": di,
        "prefix_length_j": int(state["prefix_length_j"]),
        "layer": int(layer),
        "kind": kind,
        "P": float(p),
        "kl_base": float(kl_base),
        "kl_repaired": float(kl_rep),
        "noop_max_abs": float(noop_max),
        "identity_pass": bool(identity_pass),
    }
    if kind == "router_freeze":
        router_row = {
            **row,
            "C_router": float(p),
            **proxies,
        }
    return row, router_row


def _qkv_by_rank(
    *,
    capture_root: Path,
    variant: str,
    sample_key: str,
    decode_index: int,
    layer: int,
) -> dict[int, torch.Tensor]:
    """Load rank-local qkv_proj captures for all TP ranks."""
    out: dict[int, torch.Tensor] = {}
    for rank in range(2):  # formal protocol is TP2
        recs = load_rank_records(capture_root / variant / "hooks", variant, sample_key, rank)
        out[rank] = _tensor_from_records(
            recs,
            sample_key=sample_key,
            decode_index=decode_index,
            layer=layer,
            boundary="qkv_proj",
            role="rank_local",
            rank=rank,
        )
    return out


def _run_one_qkvo(
    *,
    llm,
    state: dict,
    layer: int,
    which: str,
    e0_recs: list,
    e1_recs: list,
    e0_logits: torch.Tensor,
    e1_logits: torch.Tensor,
    kl_base: float,
    logits_root: Path,
    capture_root: Path | None = None,
) -> dict[str, Any]:
    key = state["sample_key"]
    di = int(state["decode_index"])
    abs_pos = int(state["abs_position"])
    kind = f"{which}_only"

    if which in {"Q", "K", "V"}:
        if capture_root is None:
            raise RuntimeError("Q/K/V interventions require capture_root for per-rank qkv tensors")
        e0_by_rank = _qkv_by_rank(
            capture_root=capture_root, variant="E0", sample_key=key, decode_index=di, layer=layer
        )
        e1_by_rank = _qkv_by_rank(
            capture_root=capture_root, variant="E1", sample_key=key, decode_index=di, layer=layer
        )
        # Runtime rank-local bounds come from live Attention module inside the op.
        # Captured-E1 no-op must use each TP rank's own rank-local qkv capture.
        noop = InstallCausalInterventionOp(
            "qkv_slice",
            {
                "sample_key": key,
                "layer": layer,
                "abs_position": abs_pos,
                "which": which,
                "source_qkv_by_rank": e1_by_rank,
                "mode": "noop",
            },
        )
        noop_max = _noop_max_abs(llm, state, e1_logits, logits_root, noop)
        repair = InstallCausalInterventionOp(
            "qkv_slice",
            {
                "sample_key": key,
                "layer": layer,
                "abs_position": abs_pos,
                "which": which,
                "source_qkv_by_rank": e0_by_rank,
                "mode": "repair",
            },
        )
    elif which == "O":
        e1_attn = _tensor_from_records(e1_recs, sample_key=key, decode_index=di, layer=layer, boundary="o_proj", role="tp_reduced")
        e0_attn = _tensor_from_records(e0_recs, sample_key=key, decode_index=di, layer=layer, boundary="o_proj", role="tp_reduced")
        noop = InstallCausalInterventionOp(
            "attn_only",
            {
                "sample_key": key,
                "layer": layer,
                "decode_index": di,
                "abs_position": abs_pos,
                "repair_attn": e1_attn,
                "repair_moe": None,
                "mode": "noop",
            },
        )
        noop_max = _noop_max_abs(llm, state, e1_logits, logits_root, noop)
        repair = InstallCausalInterventionOp(
            "attn_only",
            {
                "sample_key": key,
                "layer": layer,
                "decode_index": di,
                "abs_position": abs_pos,
                "repair_attn": e0_attn,
                "repair_moe": None,
                "mode": "repair",
            },
        )
    else:
        raise ValueError(which)

    repaired = _generate_with_intervention(llm, state, "E1", logits_root, repair)
    kl_rep = exact_kl(e0_logits, repaired)
    return {
        "sample_key": key,
        "calibration_sample_id": state.get("calibration_sample_id"),
        "split": state["split"],
        "decode_index": di,
        "prefix_length_j": int(state["prefix_length_j"]),
        "layer": int(layer),
        "kind": kind,
        "which": which,
        "P": float(kl_base - kl_rep),
        "kl_base": float(kl_base),
        "kl_repaired": float(kl_rep),
        "noop_max_abs": float(noop_max),
        "identity_pass": bool(noop_max == 0.0),
    }


def _parse_gpu_pairs(raw: str | None) -> list[str]:
    text = (raw or DEFAULT_GPU_PAIRS).strip()
    pairs = [p.strip() for p in text.split(";") if p.strip()]
    if not pairs:
        raise ValueError("empty GPU pairs")
    return pairs


def _worker_paths(prot_dir: Path, router_dir: Path, worker_id: int | None) -> tuple[Path, Path]:
    if worker_id is None:
        return prot_dir / "full48_causal_rows.jsonl", router_dir / "router_intervention_rows.jsonl"
    return (
        prot_dir / f"full48_causal_rows.worker{worker_id}.jsonl",
        router_dir / f"router_intervention_rows.worker{worker_id}.jsonl",
    )


def _merge_worker_jsonl(paths: list[Path], out_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(load_jsonl_optional(path))
    rows = dedupe_rows_prefer_identity(rows)
    write_jsonl(out_path, rows)
    return rows


def run_base_interventions_worker(
    *,
    run_root: Path,
    model_path: str,
    phasea_root: Path,
    worker_id: int = 0,
    num_workers: int = 1,
) -> dict[str, Any]:
    run_root = Path(run_root)
    prot_dir = run_root / "50_protection" / "s3_full48"
    router_dir = run_root / "40_router" / "s3_full48"
    prot_dir.mkdir(parents=True, exist_ok=True)
    router_dir.mkdir(parents=True, exist_ok=True)
    causal_path, router_path = _worker_paths(prot_dir, router_dir, worker_id if num_workers > 1 else None)
    logits_root = prot_dir / f"logits_w{worker_id}"
    hooks_tmp = prot_dir / f"hooks_w{worker_id}"

    cohort = read_jsonl(run_root / "00_protocol" / "internal_error_state_cohort.jsonl")
    capture_root = run_root / "10_capture"
    existing = load_jsonl_optional(causal_path) + load_jsonl_optional(
        prot_dir / "full48_causal_rows.jsonl"
    )
    existing_router = load_jsonl_optional(router_path) + load_jsonl_optional(
        router_dir / "router_intervention_rows.jsonl"
    )
    done = load_completed_keys(existing) | load_completed_keys(existing_router)

    planned = []
    for state in cohort:
        kinds = DISCOVERY_BASE_KINDS if state["split"] == "discovery" else HOLDOUT_BASE_KINDS
        for layer in range(NUM_LAYERS):
            for kind in kinds:
                key = intervention_key(
                    split=state["split"],
                    sample_key=state["sample_key"],
                    decode_index=int(state["decode_index"]),
                    layer=layer,
                    kind=kind,
                )
                if key in done:
                    continue
                if not shard_owns_key(key, worker_id=worker_id, num_workers=num_workers):
                    continue
                planned.append((state, layer, kind))

    if not planned:
        return {"worker_id": worker_id, "n_planned": 0, "n_done": 0, "causal_path": str(causal_path)}

    llm, _ = build_real_vllm(
        "E1",
        model_path=model_path,
        phasea_root=Path(phasea_root),
        max_num_seqs=1,
        gpu_memory_utilization=0.90,
    )
    llm.apply_model(InstallHooksOp("E1", str(hooks_tmp), "core_qkv"))
    n_done = 0
    try:
        # Group by state for capture reuse
        by_state: dict[str, list[tuple]] = defaultdict(list)
        for state, layer, kind in planned:
            by_state[state["sample_key"]].append((state, layer, kind))
        for _sk, items in by_state.items():
            state = items[0][0]
            key = state["sample_key"]
            di = int(state["decode_index"])
            e0_recs = load_rank_records(capture_root / "E0" / "hooks", "E0", key, 0)
            e1_recs = load_rank_records(capture_root / "E1" / "hooks", "E1", key, 0)
            e0_logits = load_raw_logits(capture_root / "E0" / "raw_logits", "E0", key, di, 0)
            e1_logits = load_raw_logits(capture_root / "E1" / "raw_logits", "E1", key, di, 0)
            kl_base = exact_kl(e0_logits, e1_logits)
            for state, layer, kind in items:
                row, router_row = _run_one_base_intervention(
                    llm=llm,
                    state=state,
                    layer=layer,
                    kind=kind,
                    e0_recs=e0_recs,
                    e1_recs=e1_recs,
                    e0_logits=e0_logits,
                    e1_logits=e1_logits,
                    kl_base=kl_base,
                    logits_root=logits_root,
                )
                row["worker_id"] = worker_id
                append_jsonl_row(causal_path, row)
                if router_row is not None:
                    router_row["worker_id"] = worker_id
                    append_jsonl_row(router_path, router_row)
                n_done += 1
    finally:
        try:
            llm.apply_model(ForceClearSampleOp())
        except Exception:
            pass
        try:
            llm.apply_model(remove_hooks)
        except Exception:
            pass
        del llm
    return {"worker_id": worker_id, "n_planned": len(planned), "n_done": n_done, "causal_path": str(causal_path)}


def run_qkvo_worker(
    *,
    run_root: Path,
    model_path: str,
    phasea_root: Path,
    attention_layers: list[int],
    worker_id: int = 0,
    num_workers: int = 1,
) -> dict[str, Any]:
    """QKVO on FIXED attention_sensitive_layers for ALL discovery+holdout states."""
    if not attention_layers:
        return {"n_done": 0, "attention_layers": []}
    run_root = Path(run_root)
    prot_dir = run_root / "50_protection" / "s3_full48"
    qkvo_path = prot_dir / (
        f"qkvo_rows.worker{worker_id}.jsonl" if num_workers > 1 else "qkvo_rows.jsonl"
    )
    logits_root = prot_dir / f"qkvo_logits_w{worker_id}"
    hooks_tmp = prot_dir / f"qkvo_hooks_w{worker_id}"
    cohort = read_jsonl(run_root / "00_protocol" / "internal_error_state_cohort.jsonl")
    capture_root = run_root / "10_capture"
    existing = load_jsonl_optional(qkvo_path) + load_jsonl_optional(prot_dir / "qkvo_rows.jsonl")
    done = load_completed_keys(existing)

    planned = []
    for state in cohort:
        for layer in attention_layers:
            for which in ("Q", "K", "V", "O"):
                key = intervention_key(
                    split=state["split"],
                    sample_key=state["sample_key"],
                    decode_index=int(state["decode_index"]),
                    layer=int(layer),
                    kind=f"{which}_only",
                )
                if key in done:
                    continue
                if not shard_owns_key(key, worker_id=worker_id, num_workers=num_workers):
                    continue
                planned.append((state, int(layer), which))

    if not planned:
        return {"n_planned": 0, "n_done": 0}

    llm, _ = build_real_vllm(
        "E1",
        model_path=model_path,
        phasea_root=Path(phasea_root),
        max_num_seqs=1,
        gpu_memory_utilization=0.90,
    )
    llm.apply_model(InstallHooksOp("E1", str(hooks_tmp), "core_qkv"))
    n_done = 0
    try:
        by_state: dict[str, list] = defaultdict(list)
        for item in planned:
            by_state[item[0]["sample_key"]].append(item)
        for _sk, items in by_state.items():
            state = items[0][0]
            key = state["sample_key"]
            di = int(state["decode_index"])
            e0_recs = load_rank_records(capture_root / "E0" / "hooks", "E0", key, 0)
            e1_recs = load_rank_records(capture_root / "E1" / "hooks", "E1", key, 0)
            e0_logits = load_raw_logits(capture_root / "E0" / "raw_logits", "E0", key, di, 0)
            e1_logits = load_raw_logits(capture_root / "E1" / "raw_logits", "E1", key, di, 0)
            kl_base = exact_kl(e0_logits, e1_logits)
            for state, layer, which in items:
                row = _run_one_qkvo(
                    llm=llm,
                    state=state,
                    layer=layer,
                    which=which,
                    e0_recs=e0_recs,
                    e1_recs=e1_recs,
                    e0_logits=e0_logits,
                    e1_logits=e1_logits,
                    kl_base=kl_base,
                    logits_root=logits_root,
                    capture_root=capture_root,
                )
                row["worker_id"] = worker_id
                append_jsonl_row(qkvo_path, row)
                n_done += 1
    finally:
        try:
            llm.apply_model(ForceClearSampleOp())
        except Exception:
            pass
        try:
            llm.apply_model(remove_hooks)
        except Exception:
            pass
        del llm
    return {"n_planned": len(planned), "n_done": n_done, "attention_layers": attention_layers}


def _spawn_workers(*, run_root: Path, model_path: str, phasea_root: Path, phase: str, extra_env: dict | None = None) -> None:
    pairs = _parse_gpu_pairs(os.environ.get(ENV_GPU_PAIRS) or os.environ.get("S3_FULL48_GPU_PAIRS"))
    num_workers = len(pairs)
    script = Path(__file__).resolve()
    repo_root = Path(__file__).resolve().parents[3]
    log_dir = Path(run_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    for wid, pair in enumerate(pairs):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = pair
        env[ENV_WORKER_ID] = str(wid)
        env[ENV_NUM_WORKERS] = str(num_workers)
        env["IEA_S3_PHASE"] = phase
        env["IEA_S3_RUN_ROOT"] = str(run_root)
        env["IEA_S3_MODEL_PATH"] = model_path
        env["IEA_S3_PHASEA_ROOT"] = str(phasea_root)
        env["PYTHONPATH"] = str(repo_root) + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        env["PYTHONUNBUFFERED"] = "1"
        if extra_env:
            env.update({k: str(v) for k, v in extra_env.items()})
        worker_log = log_dir / f"s3_full48_worker{wid}_{phase}.log"
        cmd = [
            sys.executable,
            "-m",
            "Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.s3_full48_causal",
            "--worker-entrypoint",
        ]
        fh = open(worker_log, "a", encoding="utf-8")
        procs.append(subprocess.Popen(cmd, env=env, cwd=str(repo_root), stdout=fh, stderr=subprocess.STDOUT))
        # Stagger engine init so two TP2 loads do not race the same node resources.
        if wid + 1 < num_workers:
            import time

            time.sleep(20)
    rc = 0
    for p in procs:
        rc = max(rc, int(p.wait()))
        if p.stdout is not None:
            p.stdout.close()
    if rc != 0:
        raise RuntimeError(f"S3_FULL48 worker phase={phase} failed with rc={rc}")


def _write_layer_importance_csv(path: Path, holdout: dict, discovery: dict) -> None:
    fields = [
        "layer",
        "causal_source",
        "P_layer_holdout",
        "P_A_holdout",
        "P_M_holdout",
        "I_A_M_holdout",
        "C_router_holdout",
        "P_layer_discovery",
        "P_A_discovery",
        "P_M_discovery",
        "I_A_M_discovery",
        "C_router_discovery",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for layer in range(NUM_LAYERS):
            h = holdout["by_layer"].get(str(layer), {})
            d = discovery["by_layer"].get(str(layer), {})
            w.writerow(
                {
                    "layer": layer,
                    "causal_source": h.get("causal_source"),
                    "P_layer_holdout": h.get("P_layer"),
                    "P_A_holdout": h.get("P_A"),
                    "P_M_holdout": h.get("P_M"),
                    "I_A_M_holdout": h.get("I_A_M"),
                    "C_router_holdout": h.get("C_router"),
                    "P_layer_discovery": d.get("P_layer"),
                    "P_A_discovery": d.get("P_A"),
                    "P_M_discovery": d.get("P_M"),
                    "I_A_M_discovery": d.get("I_A_M"),
                    "C_router_discovery": d.get("C_router"),
                }
            )


def _write_qkvo_importance_csv(path: Path, qkvo_rows: list[dict], attention_layers: list[int]) -> None:
    # sample-level then mean across samples, pooled discovery+holdout
    fields = ["layer", "P_Q", "P_K", "P_V", "P_O"]
    with Path(path).open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for layer in attention_layers:
            row_out = {"layer": layer}
            for which, kind in (("Q", "Q_only"), ("K", "K_only"), ("V", "V_only"), ("O", "O_only")):
                bucket: dict[str, list[float]] = defaultdict(list)
                for r in qkvo_rows:
                    if int(r["layer"]) != layer or r.get("kind") != kind:
                        continue
                    sid = str(r.get("calibration_sample_id") or r["sample_key"].rsplit("__j", 1)[0])
                    bucket[sid].append(float(r["P"]))
                sample_means = [sum(v) / len(v) for v in bucket.values() if v]
                row_out[f"P_{which}"] = (sum(sample_means) / len(sample_means)) if sample_means else None
            w.writerow(row_out)


def _finalize_artifacts(
    *,
    run_root: Path,
    causal_rows: list[dict],
    router_rows: list[dict],
    qkvo_rows: list[dict],
    s2_p_layer: dict[tuple[str, int], float],
    cohort: list[dict],
) -> dict[str, Any]:
    prot_dir = run_root / "50_protection" / "s3_full48"
    router_dir = run_root / "40_router" / "s3_full48"
    gate = assert_base_completeness(causal_rows, cohort)

    holdout_summary = aggregate_split_layer_summary(causal_rows, router_rows, split="holdout")
    discovery_summary = aggregate_split_layer_summary(
        causal_rows, router_rows, split="discovery", s2_p_layer=s2_p_layer
    )
    atomic_write_json(prot_dir / "full48_holdout_layer_summary.json", holdout_summary)
    atomic_write_json(prot_dir / "full48_discovery_layer_summary.json", discovery_summary)
    _write_layer_importance_csv(prot_dir / "full48_layer_importance_table.csv", holdout_summary, discovery_summary)

    causal_map = {int(k): v for k, v in holdout_summary["causal_source_by_layer"].items()}
    attn_layers = attention_sensitive_layers_from_sources(causal_map)
    atomic_write_json(
        prot_dir / "attention_sensitive_layers.DRAFT.json",
        {
            "status": "DRAFT_PENDING_REVIEW",
            "attention_sensitive_layers": attn_layers,
            "causal_source_by_layer": {str(k): v for k, v in causal_map.items()},
            "rule": "holdout sample-level P_A/P_M only; QKVO uses this fixed set",
        },
    )

    # Router summary + Spearman / Top-K overlap (analysis only)
    layers = list(range(NUM_LAYERS))
    c_vals, full_kl, topk_tot = [], [], []
    for layer in layers:
        h = holdout_summary["by_layer"][str(layer)]
        c_vals.append(float(h["C_router"] or 0.0))
        # mean proxy over holdout router rows for this layer (state then sample — use row mean of sample-level)
        r_bucket: dict[str, list[dict]] = defaultdict(list)
        for r in router_rows:
            if r.get("split") != "holdout" or int(r["layer"]) != layer:
                continue
            sid = str(r.get("calibration_sample_id") or r["sample_key"].rsplit("__j", 1)[0])
            r_bucket[sid].append(r)
        sample_full, sample_tot = [], []
        for _sid, rs in r_bucket.items():
            sample_full.append(sum(float(x["router_full_kl"]) for x in rs) / len(rs))
            sample_tot.append(sum(float(x["router_topk_total"]) for x in rs) / len(rs))
        full_kl.append(sum(sample_full) / len(sample_full) if sample_full else 0.0)
        topk_tot.append(sum(sample_tot) / len(sample_tot) if sample_tot else 0.0)

    ranked_c = [l for l, _ in sorted(zip(layers, c_vals), key=lambda t: t[1], reverse=True)]
    ranked_full = [l for l, _ in sorted(zip(layers, full_kl), key=lambda t: t[1], reverse=True)]
    ranked_tot = [l for l, _ in sorted(zip(layers, topk_tot), key=lambda t: t[1], reverse=True)]
    o3 = decide_o3_layers(holdout_summary["by_layer"], causal_source_by_layer=causal_map)
    router_summary = {
        "o3_layers": o3,
        "spearman_router_full_kl_vs_C_router": spearman_rho(full_kl, c_vals),
        "spearman_router_topk_total_vs_C_router": spearman_rho(topk_tot, c_vals),
        "topk_overlap_full_kl_vs_C_router": topk_overlap(ranked_full, ranked_c),
        "topk_overlap_topk_total_vs_C_router": topk_overlap(ranked_tot, ranked_c),
        "by_layer": holdout_summary["by_layer"],
        "note": "proxies are analysis-only; O3 eligibility uses C_router gate only",
    }
    atomic_write_json(router_dir / "router_layer_summary.json", router_summary)
    atomic_write_json(
        router_dir / "o3_layers.DRAFT.json",
        {
            "status": "DRAFT_PENDING_REVIEW",
            "o3_layers": o3,
            "rule": "causal_source in {moe,both} AND mean(C_router)>0 AND median>0 AND positive_count>=5/8",
        },
    )

    write_jsonl(prot_dir / "qkvo_rows.jsonl", qkvo_rows)
    _write_qkvo_importance_csv(prot_dir / "attention_linear_importance.csv", qkvo_rows, attn_layers)

    selection_path = run_root / "50_protection" / "layer_selection_manifest.json"
    selection = read_json(selection_path) if selection_path.is_file() else {}
    practical, objective = build_draft_scopes(
        discovery_summary=discovery_summary,
        holdout_summary=holdout_summary,
        selection=selection,
        o3_layers=o3,
    )
    # Write under s3_full48 always; archive then update main DRAFT paths for S4.
    atomic_write_json(prot_dir / "practical_protection_scope.DRAFT.json", practical)
    atomic_write_json(prot_dir / "objective_scope.DRAFT.json", objective)
    main_practical = run_root / "50_protection" / "practical_protection_scope.json"
    main_objective = run_root / "60_objective" / "objective_scope.json"
    archive_pre_full48_scope(main_practical)
    archive_pre_full48_scope(main_objective)
    atomic_write_json(main_practical, practical)
    atomic_write_json(main_objective, objective)

    report = (
        "# S3_FULL48_CAUSAL Report\n\n"
        f"- base_completeness: {gate}\n"
        f"- attention_sensitive_layers: {attn_layers}\n"
        f"- o3_layers: {o3}\n"
        f"- qkvo_rows: {len(qkvo_rows)} (expected {len(attn_layers) * 16 * 4 * 4} if 16 samples×4 states×4 ops)\n"
        f"- spearman full_kl vs C_router: {router_summary['spearman_router_full_kl_vs_C_router']}\n"
        f"- spearman topk_total vs C_router: {router_summary['spearman_router_topk_total_vs_C_router']}\n"
        "- Old partial-S3 artifacts retained; new rows only under s3_full48/.\n"
        "- Formal scope uses holdout sample-level aggregation only.\n"
    )
    (prot_dir / "S3_FULL48_CAUSAL_REPORT.md").write_text(report, encoding="utf-8")
    return {
        "gate": gate,
        "attention_sensitive_layers": attn_layers,
        "o3_layers": o3,
        "n_qkvo": len(qkvo_rows),
        "causal_source_by_layer": causal_map,
    }


def run_s3_full48_causal(
    *,
    run_root: Path,
    model_path: str = DEFAULT_MODEL_PATH,
    phasea_root: Path = DEFAULT_PHASEA_ROOT,
) -> dict[str, Any]:
    """Formal entry: base 10752 interventions → gate → QKVO → DRAFT scopes."""
    run_root = Path(run_root)
    prot_dir = run_root / "50_protection" / "s3_full48"
    router_dir = run_root / "40_router" / "s3_full48"
    prot_dir.mkdir(parents=True, exist_ok=True)
    router_dir.mkdir(parents=True, exist_ok=True)

    # Refuse clobbering legacy partial-S3 paths by never writing them.
    legacy_causal = run_root / "50_protection" / "s3_substructure_rows.jsonl"
    legacy_router = run_root / "40_router" / "router_intervention_rows.jsonl"

    worker_env = os.environ.get(ENV_WORKER_ID)
    num_env = os.environ.get(ENV_NUM_WORKERS)
    phase = os.environ.get("IEA_S3_PHASE", "base")

    # External shard / internal worker mode
    if worker_env is not None:
        wid = int(worker_env)
        n_w = int(num_env or 1)
        if phase == "qkvo":
            attn = json.loads(os.environ["IEA_S3_ATTN_LAYERS"])
            return run_qkvo_worker(
                run_root=run_root,
                model_path=model_path,
                phasea_root=phasea_root,
                attention_layers=[int(x) for x in attn],
                worker_id=wid,
                num_workers=n_w,
            )
        return run_base_interventions_worker(
            run_root=run_root,
            model_path=model_path,
            phasea_root=phasea_root,
            worker_id=wid,
            num_workers=n_w,
        )

    pairs_raw = os.environ.get(ENV_GPU_PAIRS) or os.environ.get("S3_FULL48_GPU_PAIRS")
    # Spawn multi-GPU workers when pairs configured and not already sharded externally
    if pairs_raw and ";" in pairs_raw and os.environ.get(ENV_SHARD) is None:
        _spawn_workers(run_root=run_root, model_path=model_path, phasea_root=phasea_root, phase="base")
        pairs = _parse_gpu_pairs(pairs_raw)
        causal_rows = _merge_worker_jsonl(
            [prot_dir / f"full48_causal_rows.worker{i}.jsonl" for i in range(len(pairs))],
            prot_dir / "full48_causal_rows.jsonl",
        )
        router_rows = _merge_worker_jsonl(
            [router_dir / f"router_intervention_rows.worker{i}.jsonl" for i in range(len(pairs))],
            router_dir / "router_intervention_rows.jsonl",
        )
    else:
        # Single-engine sequential (or external IEA_S3_SHARD workers already wrote shards)
        shard = os.environ.get(ENV_SHARD)
        if shard is not None:
            # External launcher: this process is one shard
            wid_s, n_s = shard.split("/")
            run_base_interventions_worker(
                run_root=run_root,
                model_path=model_path,
                phasea_root=phasea_root,
                worker_id=int(wid_s),
                num_workers=int(n_s),
            )
            return {"status": "shard_worker_complete", "shard": shard}

        # Merge any existing worker shards then fill gaps sequentially
        worker_causal = sorted(prot_dir.glob("full48_causal_rows.worker*.jsonl"))
        if worker_causal:
            causal_rows = _merge_worker_jsonl(worker_causal, prot_dir / "full48_causal_rows.jsonl")
            router_rows = _merge_worker_jsonl(
                sorted(router_dir.glob("router_intervention_rows.worker*.jsonl")),
                router_dir / "router_intervention_rows.jsonl",
            )
        else:
            causal_rows = load_jsonl_optional(prot_dir / "full48_causal_rows.jsonl")
            router_rows = load_jsonl_optional(router_dir / "router_intervention_rows.jsonl")

        # Resume missing keys on this process
        run_base_interventions_worker(
            run_root=run_root,
            model_path=model_path,
            phasea_root=phasea_root,
            worker_id=0,
            num_workers=1,
        )
        causal_rows = load_jsonl_optional(prot_dir / "full48_causal_rows.jsonl")
        router_rows = load_jsonl_optional(router_dir / "router_intervention_rows.jsonl")

    cohort = read_jsonl(run_root / "00_protocol" / "internal_error_state_cohort.jsonl")
    s2_path = run_root / "50_protection" / "single_layer_protection_rows.jsonl"
    s2_p_layer = load_s2_p_layer_lookup(s2_path)
    causal_rows = dedupe_rows_prefer_identity(causal_rows)
    router_rows = dedupe_rows_prefer_identity(router_rows)
    write_jsonl(prot_dir / "full48_causal_rows.jsonl", causal_rows)
    write_jsonl(router_dir / "router_intervention_rows.jsonl", router_rows)

    # Completeness gate BEFORE new scopes / QKVO freeze path
    gate = assert_base_completeness(causal_rows, cohort)
    holdout_summary = aggregate_split_layer_summary(causal_rows, router_rows, split="holdout")
    causal_map = {int(k): v for k, v in holdout_summary["causal_source_by_layer"].items()}
    attn_layers = attention_sensitive_layers_from_sources(causal_map)
    atomic_write_json(
        prot_dir / "attention_sensitive_layers.DRAFT.json",
        {
            "status": "DRAFT_PENDING_REVIEW",
            "attention_sensitive_layers": attn_layers,
            "causal_source_by_layer": {str(k): v for k, v in causal_map.items()},
        },
    )

    # QKVO on fixed layers only
    if attn_layers:
        if pairs_raw and ";" in pairs_raw:
            _spawn_workers(
                run_root=run_root,
                model_path=model_path,
                phasea_root=phasea_root,
                phase="qkvo",
                extra_env={"IEA_S3_ATTN_LAYERS": json.dumps(attn_layers)},
            )
            pairs = _parse_gpu_pairs(pairs_raw)
            qkvo_rows = _merge_worker_jsonl(
                [prot_dir / f"qkvo_rows.worker{i}.jsonl" for i in range(len(pairs))],
                prot_dir / "qkvo_rows.jsonl",
            )
        else:
            run_qkvo_worker(
                run_root=run_root,
                model_path=model_path,
                phasea_root=phasea_root,
                attention_layers=attn_layers,
                worker_id=0,
                num_workers=1,
            )
            qkvo_rows = load_jsonl_optional(prot_dir / "qkvo_rows.jsonl")
        expected_qkvo = expected_qkvo_keys(cohort, attn_layers)
        got = {key_from_row(r) for r in qkvo_rows if r.get("identity_pass")}
        missing_q = [k for k in expected_qkvo if k not in got]
        if missing_q:
            raise RuntimeError(
                f"QKVO incomplete for fixed attention layers: missing={len(missing_q)} "
                f"preview={missing_q[:5]}"
            )
    else:
        qkvo_rows = []
        write_jsonl(prot_dir / "qkvo_rows.jsonl", [])
    qkvo_rows = dedupe_rows_prefer_identity(qkvo_rows)

    # Legacy partial-S3 paths are never opened for write in this module.
    _ = (legacy_causal, legacy_router)

    result = _finalize_artifacts(
        run_root=run_root,
        causal_rows=causal_rows,
        router_rows=router_rows,
        qkvo_rows=qkvo_rows,
        s2_p_layer=s2_p_layer,
        cohort=cohort,
    )
    result["gate"] = gate
    result["legacy_s3_preserved"] = {
        "s3_substructure_rows": str(legacy_causal),
        "router_intervention_rows": str(legacy_router),
    }
    return result


def _worker_main() -> int:
    run_root = Path(os.environ["IEA_S3_RUN_ROOT"])
    model_path = os.environ.get("IEA_S3_MODEL_PATH", DEFAULT_MODEL_PATH)
    phasea_root = Path(os.environ.get("IEA_S3_PHASEA_ROOT", str(DEFAULT_PHASEA_ROOT)))
    run_s3_full48_causal(run_root=run_root, model_path=model_path, phasea_root=phasea_root)
    return 0


if __name__ == "__main__":
    if "--worker-entrypoint" in sys.argv:
        raise SystemExit(_worker_main())
    raise SystemExit("Use run_s3_full48_causal(...) from pipeline or set worker env vars")
