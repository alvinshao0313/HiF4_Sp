"""E0 progressive Router teacher cache for formal O3_full / O3_topk (plan O3.1 C)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.modelopt_moe_checkpoint import (
    load_qwen3_moe_layer_state,
    release_qwen3_moe_layer_state,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_model_spec import (
    load_qwen3_moe_model_spec,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_semantic_hif4 import (
    NativeQwen3MoELayerRuntime,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.data.batching import (
    DynamicCalibrationCollator,
    build_validation_batches,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.data.calibration import (
    CalibrationSample,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.training.moe_layer_runtime import (
    build_qwen3_moe_layer_call,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.training.moe_trainer import (
    build_initial_moe_hidden_cache,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.training.layer_runtime import (
    ProgressiveHiddenCache,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.checkpoint import resolve_local_snapshot

from .config import S1K_SHARED_CALIB, WIKITEXT2_SHARED_CALIB
from .router_objective import build_production_topk_teacher
from .run_state import atomic_write_json
from .attention_semantics import causal_attention_mask, TRAINING_PATH_VERSION


STATUS_EMPTY = "EMPTY"
STATUS_COMPLETE = "COMPLETE"


def load_objective_calibration_samples(sample_ids: list[str]) -> list[CalibrationSample]:
    """Load calibration rows for objective IDs from the shared 50:50 pools.

    Formal objective split IDs are drawn from shared train.pt pools (Wiki + S1K).
    """
    pools: list[CalibrationSample] = []
    for root in (WIKITEXT2_SHARED_CALIB, S1K_SHARED_CALIB):
        train_path = Path(root) / "calibration" / "train.pt"
        val_path = Path(root) / "calibration" / "val.pt"
        for path in (train_path, val_path):
            if path.is_file():
                rows = torch.load(path, map_location="cpu", weights_only=False)
                pools.extend(list(rows))
    by_id = {s.sample_id: s for s in pools}
    missing = [sid for sid in sample_ids if sid not in by_id]
    if missing:
        raise RuntimeError(f"objective sample ids missing from shared calib: {missing[:8]}")
    # Preserve requested order (trainer / cache batching consume this order).
    return [by_id[sid] for sid in sample_ids]


def reshape_flat_router_logits(
    router_logits: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
) -> torch.Tensor:
    """Reshape flat [B*T, E] router logits to [B, T, E]. Reject ambiguous shapes."""
    if router_logits.ndim == 3:
        if int(router_logits.shape[0]) != batch_size or int(router_logits.shape[1]) != seq_len:
            raise RuntimeError(
                f"router_logits [B,T,E] shape {tuple(router_logits.shape)} "
                f"incompatible with B={batch_size} T={seq_len}"
            )
        return router_logits
    if router_logits.ndim != 2:
        raise RuntimeError(f"router_logits must be 2D or 3D, got shape={tuple(router_logits.shape)}")
    n_rows, n_experts = int(router_logits.shape[0]), int(router_logits.shape[1])
    expected = int(batch_size) * int(seq_len)
    if n_rows != expected:
        raise RuntimeError(
            f"flat router_logits rows={n_rows} != B*T={expected}; "
            "refusing to guess sample boundaries"
        )
    return router_logits.view(batch_size, seq_len, n_experts)


def slice_router_logits_by_lengths(
    router_logits_bte: torch.Tensor,
    lengths: torch.Tensor | list[int],
) -> list[torch.Tensor]:
    """Slice [B,T,E] logits by real lengths; drop padding rows."""
    if router_logits_bte.ndim != 3:
        raise ValueError(f"expected [B,T,E], got {tuple(router_logits_bte.shape)}")
    out: list[torch.Tensor] = []
    for i in range(int(router_logits_bte.shape[0])):
        n = int(lengths[i] if not torch.is_tensor(lengths) else lengths[i].item())
        if n <= 0 or n > int(router_logits_bte.shape[1]):
            raise RuntimeError(f"invalid length={n} for T={router_logits_bte.shape[1]}")
        out.append(router_logits_bte[i, :n].contiguous())
    return out


def _cache_dir(run_root: Path) -> Path:
    return Path(run_root) / "60_objective" / "router_teacher_cache"


def _layer_tag(layer: int) -> str:
    return f"L{int(layer):02d}"


def _write_split_payload(
    path: Path,
    *,
    layer: int,
    payloads: dict[str, dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payloads, path)


def build_router_teacher_cache(
    *,
    run_root: Path,
    model_path: str,
    objective_split_manifest: dict,
    layers: list[int],
    batch_size: int,
) -> dict:
    """Build E0 progressive Router teacher cache for o3_layers only (plan O3.1 C)."""
    run_root = Path(run_root)
    out_dir = _cache_dir(run_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "manifest.json").exists():
        raise RuntimeError("refusing to overwrite an existing Router teacher cache; use a separate explicit phase")

    layer_list = sorted({int(x) for x in layers})
    train_ids = [str(x) for x in objective_split_manifest["train_ids"]]
    val_ids = [str(x) for x in objective_split_manifest["val_ids"]]
    source_ratio = objective_split_manifest.get("source_ratio")

    overlap = sorted(set(train_ids) & set(val_ids))
    if overlap:
        raise RuntimeError(f"train/val objective ids overlap: {overlap[:8]}")

    if not layer_list:
        manifest = {
            "training_path_version": TRAINING_PATH_VERSION,
            "model_path": model_path,
            "layers": [],
            "num_experts": None,
            "top_k": None,
            "norm_topk_prob": None,
            "train_ids": train_ids,
            "val_ids": val_ids,
            "source_ratio": source_ratio,
            "status": STATUS_EMPTY,
        }
        atomic_write_json(out_dir / "manifest.json", manifest)
        return manifest

    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")

    snapshot = Path(resolve_local_snapshot(model_path))
    spec = load_qwen3_moe_model_spec(str(snapshot))
    if int(spec.top_k) != int(spec.top_k):  # noqa: PLR0124 — keep explicit runtime read
        raise RuntimeError("top_k self-check failed")
    top_k = int(spec.top_k)
    num_experts = int(spec.num_experts)
    norm_topk_prob = bool(spec.norm_topk_prob)
    if not norm_topk_prob:
        raise RuntimeError("norm_topk_prob must be True for router teacher cache (Gate fail)")

    # Cross-check HF config.json (must agree with loaded spec; never hardcode k).
    import json

    hf_cfg = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    hf_top_k = int(hf_cfg["num_experts_per_tok"])
    hf_experts = int(hf_cfg["num_experts"])
    hf_norm = bool(hf_cfg["norm_topk_prob"])
    if hf_top_k != top_k:
        raise RuntimeError(f"top_k mismatch: spec={top_k} hf={hf_top_k}")
    if hf_experts != num_experts:
        raise RuntimeError(f"num_experts mismatch: spec={num_experts} hf={hf_experts}")
    if hf_norm != norm_topk_prob:
        raise RuntimeError(f"norm_topk_prob mismatch: spec={norm_topk_prob} hf={hf_norm}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    collator = DynamicCalibrationCollator(int(tokenizer.pad_token_id))

    train_samples = load_objective_calibration_samples(train_ids)
    val_samples = load_objective_calibration_samples(val_ids)
    all_samples = list(train_samples) + list(val_samples)
    train_id_set = set(train_ids)
    val_id_set = set(val_ids)

    max_layer = max(layer_list)
    x_cache = build_initial_moe_hidden_cache(
        snapshot, all_samples, collator, device, int(batch_size)
    )

    # layer -> split -> sample_id -> payload
    collected: dict[int, dict[str, dict[str, dict[str, Any]]]] = {
        layer: {"train": {}, "val": {}} for layer in layer_list
    }

    for layer_idx in range(0, max_layer + 1):
        state = load_qwen3_moe_layer_state(snapshot, layer_idx, device)
        try:
            native = NativeQwen3MoELayerRuntime(state, is_causal=True, o_proj_tp_size=2).to(device).eval()
            nxt = ProgressiveHiddenCache()
            capture = layer_idx in collected
            for batch in build_validation_batches(all_samples, int(batch_size)):
                packed = collator(batch)
                sample_ids = [s.sample_id for s in batch]
                hidden, _ = x_cache.assemble(sample_ids, device)
                bsz, seq_len, _ = hidden.shape
                call = build_qwen3_moe_layer_call(str(snapshot), hidden)
                with torch.no_grad():
                    out = native(
                        hidden,
                        attention_mask=None,
                        position_embeddings=call.position_embeddings,
                    )
                lengths = packed["lengths"]
                if capture:
                    logits_bte = reshape_flat_router_logits(
                        out.router_logits.detach(),
                        batch_size=bsz,
                        seq_len=seq_len,
                    )
                    per_sample = slice_router_logits_by_lengths(logits_bte, lengths)
                    for i, sample in enumerate(batch):
                        sid = sample.sample_id
                        if sid in train_id_set:
                            split = "train"
                        elif sid in val_id_set:
                            split = "val"
                        else:
                            raise RuntimeError(f"sample {sid} not in train/val objective split")
                        logits_i = per_sample[i].to(device="cpu", dtype=torch.bfloat16).contiguous()
                        if int(logits_i.shape[0]) != int(lengths[i].item()):
                            raise RuntimeError("length mismatch after padding strip")
                        if int(logits_i.shape[1]) != num_experts:
                            raise RuntimeError(
                                f"num_experts mismatch: logits E={logits_i.shape[1]} spec E={num_experts}"
                            )
                        topk_ids, topk_weights = build_production_topk_teacher(
                            logits_i.float(),
                            top_k=top_k,
                            norm_topk_prob=norm_topk_prob,
                        )
                        payload = {
                            "sample_id": sid,
                            "layer": int(layer_idx),
                            "length": int(logits_i.shape[0]),
                            "router_logits": logits_i,
                            "topk_ids": topk_ids.to(device="cpu", dtype=torch.int64).contiguous(),
                            "topk_weights": topk_weights.to(device="cpu", dtype=torch.float32).contiguous(),
                        }
                        collected[layer_idx][split][sid] = payload
                for i, sample in enumerate(batch):
                    n = int(lengths[i].item())
                    nxt.store(sample.sample_id, out.output[i, :n], n)
            x_cache = nxt
        finally:
            release_qwen3_moe_layer_state(state)
            del native
            if device.type == "cuda":
                torch.cuda.empty_cache()

    for layer_idx in layer_list:
        for split, ids in (("train", train_ids), ("val", val_ids)):
            payloads = collected[layer_idx][split]
            missing = [sid for sid in ids if sid not in payloads]
            if missing:
                raise RuntimeError(
                    f"router teacher cache incomplete for layer={layer_idx} split={split}: {missing[:8]}"
                )
            # Ensure no cross-split leakage in the dict itself.
            foreign = [sid for sid in payloads if sid not in set(ids)]
            if foreign:
                raise RuntimeError(
                    f"train/val isolation violated at L{layer_idx} {split}: {foreign[:8]}"
                )
            _write_split_payload(
                out_dir / f"{_layer_tag(layer_idx)}_{split}.pt",
                layer=layer_idx,
                payloads=payloads,
            )

    manifest = {
        "model_path": model_path,
        "layers": layer_list,
        "num_experts": num_experts,
        "top_k": top_k,
        "norm_topk_prob": norm_topk_prob,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "source_ratio": source_ratio,
        "status": STATUS_COMPLETE,
        "training_path_version": TRAINING_PATH_VERSION,
        "execution_path": "causal_training_runtime; actual-vLLM equivalence requires independent audit",
    }
    atomic_write_json(out_dir / "manifest.json", manifest)
    return manifest


def load_router_teacher_split(
    cache_dir: Path,
    *,
    layer: int,
    split: str,
) -> dict[str, dict[str, Any]]:
    path = Path(cache_dir) / f"{_layer_tag(layer)}_{split}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"missing router teacher cache: {path}")
    manifest_path=Path(cache_dir)/'manifest.json'
    if manifest_path.exists():
        from .run_state import read_json
        manifest=read_json(manifest_path)
        if manifest.get('training_path_version')==4:
            from .candidate_runtime import sha256
            if sha256(path)!=manifest['files_sha256'][path.name]:
                raise RuntimeError('actual Router teacher content hash changed')
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid teacher cache type at {path}")
    return payload
