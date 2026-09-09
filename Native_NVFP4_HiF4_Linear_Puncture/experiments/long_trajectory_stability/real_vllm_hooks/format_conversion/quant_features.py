"""Offline features of captured production inputs; never a model/runtime replay."""
from __future__ import annotations

import json
import hashlib
from collections import defaultdict
from pathlib import Path

import torch

from Native_NVFP4_HiF4_Linear_Puncture.src.formats import qdq_hif4_direct

FEATURE_BOUNDARIES = frozenset({"input_norm", "post_attn_norm"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_verified_capture(capture_path: Path, manifest_path: Path, canonical_path: Path,
                          *, capture_level: str) -> tuple[dict, dict]:
    """Bind tensors to a forced manifest and the actual isolated E0 token file.

    A caller-provided boolean is deliberately insufficient. Old captures without
    input IDs and publication-time file hashes cannot establish a new gate.
    """
    capture_path, manifest_path, canonical_path = map(Path, (capture_path, manifest_path, canonical_path))
    manifest = json.loads(manifest_path.read_text())
    mode = {"feature_scan": "forced_feature_scan", "core_qkv": "forced_core_qkv", "core": "forced_core"}[capture_level]
    if manifest["mode"] != mode or manifest["variant"] not in {"E0", "E1", "E2", "E3"}:
        raise RuntimeError("capture manifest is not a permitted canonical forced mode")
    protocol = {"tensor_parallel_size": 2, "kv_cache_dtype": "bfloat16", "enforce_eager": True,
                "enable_prefix_caching": False, "max_num_seqs": 1, "max_model_len": 40960}
    if any(manifest["runtime"].get(k) != v for k, v in protocol.items()):
        raise RuntimeError("capture runtime does not satisfy the isolated TP2 protocol")
    plan_path = Path(manifest["probe_plan"])
    if manifest.get("probe_plan_sha256") != _sha256(plan_path):
        raise RuntimeError("capture probe plan is unbound or changed")
    canonical_meta_path = canonical_path.with_suffix(".meta.json")
    meta = json.loads(canonical_meta_path.read_text())
    if meta["variant"] != "E0" or meta["execution_shape"] != "single_request_max_num_seqs_1":
        raise RuntimeError("canonical source must be an isolated E0 run")
    if any(meta["runtime"].get(k) != v for k, v in protocol.items()):
        raise RuntimeError("canonical source runtime does not satisfy isolated TP2 protocol")
    canonical_rows = [json.loads(line) for line in canonical_path.read_text().splitlines() if line.strip()]
    canonical = {row["prompt_key"]: row for row in canonical_rows}
    if len(canonical) != len(canonical_rows):
        raise RuntimeError("duplicate isolated E0 sample keys")
    payload = torch.load(capture_path, map_location="cpu", weights_only=False)
    sample_key, rank = payload["sample_key"], payload["rank"]
    matches = [row for row in manifest["samples"] if row["sample_key"] == sample_key]
    if len(matches) != 1 or rank not in (0, 1) or payload["world_size"] != 2:
        raise RuntimeError("capture sample/rank metadata mismatch")
    sample, source = matches[0], canonical[sample_key]
    if payload["variant"] != manifest["variant"] or payload.get("capture_level") != capture_level:
        raise RuntimeError("capture payload variant/level mismatch")
    input_ids = sample.get("input_ids")
    if input_ids != source["input_ids"] or payload["prompt_len"] != len(input_ids):
        raise RuntimeError("captured prompt IDs differ from isolated E0 or are absent")
    output_hash = hashlib.sha256(json.dumps(source["output_ids"], separators=(",", ":")).encode()).hexdigest()
    if sample.get("canonical_output_ids_sha256") != output_hash:
        raise RuntimeError("capture canonical output identity differs from isolated E0")
    length = sample["required_tokens"]
    expected = source["output_ids"][:length]
    if length <= 0 or len(expected) != length or sample.get("forced_exact") is not True or sample["generated_ids"] != expected or sample["expected_forced_ids"] != expected:
        raise RuntimeError("captured generated trajectory is not exact isolated E0 history")
    published = [row for row in sample["flush"] if row["rank"] == rank]
    capture_hash = _sha256(capture_path)
    if len(published) != 1 or Path(published[0]["path"]).resolve() != capture_path.resolve() or published[0].get("sha256") != capture_hash:
        raise RuntimeError("capture tensor artifact is unbound or changed")
    probes = set(sample["probe_decode_indices"])
    for record in payload["records"]:
        if record["sample_key"] != sample_key or record["variant"] != payload["variant"] or record["tp_rank"] != rank or record["tp_world_size"] != 2:
            raise RuntimeError("capture record provenance differs from its manifest")
        decode = record["decode_index"]
        if decode not in probes or not 0 <= decode < length or record["abs_position"] != len(input_ids) + decode - 1:
            raise RuntimeError("capture record predictor position differs from its manifest")
    evidence = {"capture_sha256": capture_hash, "manifest_sha256": _sha256(manifest_path),
                "canonical_trajectory_sha256": _sha256(canonical_path), "canonical_meta_sha256": _sha256(canonical_meta_path)}
    return payload, evidence


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def distribution_features(x: torch.Tensor) -> dict:
    """One captured row. Ratios with a zero denominator are explicitly undefined.

    Group dynamic range is max/min group amax; subgroup imbalance is the
    max/min of the four subgroup16 amax values within each group64.
    Kurtosis is the centered fourth standardized moment (not excess kurtosis).
    """
    if x.ndim != 1 or x.numel() == 0 or x.numel() % 64:
        raise ValueError(f"expected one nonempty row with width divisible by 64: {tuple(x.shape)}")
    if not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError("feature input must be finite floating point")
    xf = x.detach().to(device="cpu", dtype=torch.float64)
    amax = float(xf.abs().max())
    # Scale before moments to avoid overflow/underflow from fourth powers.
    scaled = xf / amax if amax else xf
    energy = float(scaled.square().sum())
    rms_scaled = float(scaled.square().mean().sqrt())
    groups = scaled.reshape(-1, 64)
    group_amax = groups.abs().amax(dim=-1)
    group_rms = groups.square().mean(dim=-1).sqrt()
    sub_amax = groups.reshape(-1, 4, 16).abs().amax(dim=-1)
    group_ratios = [_ratio(float(a), float(r)) for a, r in zip(group_amax, group_rms, strict=True)]
    subgroup_ratios = [_ratio(float(v.max()), float(v.min())) for v in sub_amax]
    centered = scaled - scaled.mean()
    variance = float(centered.square().mean())
    valid_subgroups = [v for v in subgroup_ratios if v is not None]
    return {
        "width": x.numel(),
        "rms": amax * rms_scaled,
        "amax": amax,
        "amax_over_rms": _ratio(1.0, rms_scaled) if amax else None,
        "l2_norm": amax * energy**0.5,
        "max_element_energy_share": _ratio(1.0, energy) if amax else None,
        "group64_count": len(group_ratios),
        "group64_amax": (group_amax * amax).tolist(),
        "group64_rms": (group_rms * amax).tolist(),
        "group64_amax_over_rms": group_ratios,
        "group64_amax_dynamic_range": _ratio(float(group_amax.max()), float(group_amax.min())),
        "group64_amax_cv": _ratio(float(group_amax.std(correction=0)), float(group_amax.mean())),
        "group64_zero_count": int((group_amax == 0).sum()),
        "subgroup16_amax": (sub_amax * amax).tolist(),
        "subgroup16_imbalance_by_group64": subgroup_ratios,
        "subgroup16_imbalance_max": max(valid_subgroups) if valid_subgroups else None,
        "subgroup16_zero_count": int((sub_amax == 0).sum()),
        "kurtosis": _ratio(float(centered.pow(4).mean()), variance**2),
    }


def quant_features(x: torch.Tensor) -> dict:
    """Activation-only HiF4 susceptibility, distinct from production e/q/p."""
    result = distribution_features(x)
    x_cpu = x.detach().to(device="cpu")
    # Use the established format oracle and its BF16 reconstruction semantics.
    quantized = qdq_hif4_direct(x_cpu)
    error = quantized.to(torch.float64) - x_cpu.to(torch.float64)
    norm = result["l2_norm"]
    error_norm = float(torch.linalg.vector_norm(error))
    result.update({
        "hif4_same_state_qdq_rel_l2": error_norm / (norm + 1e-12),
        "hif4_same_state_qdq_l2": error_norm,
        "hif4_qdq_scope": "activation_only",
        "nvfp4_same_state_qdq_status": "NOT_COMPUTED_NO_UNAMBIGUOUS_SCALE_TRANSFORM_METADATA",
    })
    return result


def feature_rows_from_payload(payload: dict, *, canonical_history_verified: bool,
                              provenance: dict | None = None) -> list[dict]:
    """Unit-testable row extraction; production path must pass verified=True after load_verified_capture."""
    if not canonical_history_verified:
        raise RuntimeError("forced token history must be verified before feature extraction")
    if payload.get("variant") not in {"E0", "E1", "E2", "E3"} or int(payload.get("world_size", -1)) != 2:
        raise RuntimeError("formal feature input must be a permitted real-vLLM TP2 capture")
    rows = []
    seen = set()
    for record in payload["records"]:
        if record["boundary"] not in FEATURE_BOUNDARIES or record["role"] != "normalized":
            raise RuntimeError("feature_scan contains a non-feature boundary/role")
        key = (record["sample_key"], record["decode_index"], record["layer"], record["boundary"])
        if key in seen:
            raise RuntimeError(f"duplicate feature key {key}")
        seen.add(key)
        metadata = {k: record[k] for k in ("sample_key", "variant", "tp_rank", "layer", "boundary", "decode_index", "abs_position")}
        rows.append({**metadata, **quant_features(record["tensor"]),
                     "trajectory": "isolated_e0_canonical_forced", "provenance": provenance or {}})
    return rows


def feature_rows(capture_path: Path, manifest_path: Path, canonical_path: Path) -> list[dict]:
    """Compute features only after validating concrete trajectory artifacts."""
    payload, provenance = load_verified_capture(capture_path, manifest_path, canonical_path, capture_level="feature_scan")
    return feature_rows_from_payload(payload, canonical_history_verified=True, provenance=provenance)


def aggregate_token_features(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["variant"], row["sample_key"], row["decode_index"], row["tp_rank"])].append(row)
    result = []
    for (variant, sample, decode, rank), group in sorted(grouped.items()):
        output = {"variant": variant, "sample_key": sample, "decode_index": decode, "tp_rank": rank}
        for source, short in (("hif4_same_state_qdq_rel_l2", "hif4_qdq_rel_l2"), ("amax_over_rms", "amax_over_rms")):
            valid = [row for row in group if row[source] is not None]
            if not valid:
                output.update({f"max_{short}": None, f"median_{short}": None, f"argmax_{short}": None})
                continue
            largest = max(valid, key=lambda r: r[source])
            values = torch.tensor([row[source] for row in valid], dtype=torch.float64)
            output.update({f"max_{short}": largest[source], f"median_{short}": float(values.quantile(0.5)),
                           f"argmax_{short}": {"layer": largest["layer"], "boundary": largest["boundary"]}})
        result.append(output)
    return result


def analyze_feature_capture(capture_root: Path, output: Path, *, manifest_path: Path, canonical_path: Path) -> dict:
    """Flush features sample by sample; compare replicated rank inputs exactly."""
    output.parent.mkdir(parents=True, exist_ok=True)
    token_path = output.with_name(output.stem + "_tokens.jsonl")
    count = samples = 0
    with output.open("w") as stream, token_path.open("w") as tokens:
        for path in sorted(capture_root.glob("**/rank0.pt")):
            payload, _ = load_verified_capture(path, manifest_path, canonical_path, capture_level="feature_scan")
            other, _ = load_verified_capture(path.with_name("rank1.pt"), manifest_path, canonical_path, capture_level="feature_scan")
            def keyed(p):
                return {(r["decode_index"], r["layer"], r["boundary"], r["role"]): r["tensor"] for r in p["records"]}
            left, right = keyed(payload), keyed(other)
            if left.keys() != right.keys() or any(not torch.equal(left[k], right[k]) for k in left):
                raise RuntimeError(f"replicated feature inputs differ between ranks: {path}")
            rows = feature_rows(path, manifest_path, canonical_path)
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            for row in aggregate_token_features(rows):
                tokens.write(json.dumps(row, allow_nan=False) + "\n")
            count += len(rows)
            samples += 1
    if samples == 0:
        raise RuntimeError(f"no feature captures found under {capture_root}")
    return {"samples": samples, "rows": count, "output": str(output), "token_output": str(token_path)}
