"""NVFP4 W4A4 online activation capture + NVFP4→HiF4 residual viz pipeline."""

from __future__ import annotations

import csv
import json
import re
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from Inference_Paradigm_Conversion.ipc_analysis.analysis.activation_distribution_residual import (
    basic_tensor_moments,
    build_token_group_residual_map,
    flatten_stats_for_csv,
    group64_residual_stats,
    residual_element_stats,
    residual_energy_concentration,
    zero_transition_stats,
)
from Inference_Paradigm_Conversion.ipc_analysis.analysis.activation_grid_occupancy import (
    build_theoretical_grid_json,
)
from Inference_Paradigm_Conversion.ipc_analysis.capture.model_adapter import (
    load_source_model_for_capture,
)
from Inference_Paradigm_Conversion.ipc_analysis.capture.prompts import (
    discovery_items,
    validation_items,
)
from Inference_Paradigm_Conversion.ipc_analysis.config import (
    IPC_ROOT,
    LINEAR_PROJECTIONS,
    resolve_representative_layers,
)
from Inference_Paradigm_Conversion.ipc_analysis.formats.hif4 import quantize_hif4_tensor
from Inference_Paradigm_Conversion.ipc_analysis.formats.nvfp4 import (
    load_nvfp4_activation_scales,
    quantize_nvfp4_activation,
    resolve_activation_scale_path,
    resolve_nvfp4_scale_for_module,
)
from Inference_Paradigm_Conversion.ipc_analysis.io_utils import (
    atomic_write_json,
    ensure_dir,
    write_csv,
)

ANALYSIS_SEED = 20260810
MAX_PREFILL_STAT_TOKENS = 128
DEFAULT_AX3_CONSOLIDATED = (
    IPC_ROOT / "results" / "20260811T_ax_final_consolidated"
)

_PROJ_RE = re.compile(
    r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
)
_PHASE_NAMES = ("prefill", "decode")
_PROJ_TO_ID = {p: i for i, p in enumerate(LINEAR_PROJECTIONS)}
_PHASE_TO_ID = {p: i for i, p in enumerate(_PHASE_NAMES)}


@dataclass
class ActivationVizContext:
    sample_id: str = ""
    prompt_family: str = ""
    split: str = ""
    phase: str = "prefill"
    decode_step: int = -1
    record_enabled: bool = False


# Module-level mutable context read by hooks.
ACTIVATION_VIZ_CONTEXT = ActivationVizContext()


@dataclass
class CaptureBuffers:
    summary_rows: list[dict[str, Any]] = field(default_factory=list)
    group_rows: list[dict[str, Any]] = field(default_factory=list)
    point_chunks: list[dict[str, torch.Tensor]] = field(default_factory=list)
    sample_id_vocab: list[str] = field(default_factory=list)
    sample_id_index: dict[str, int] = field(default_factory=dict)
    token_group_maps: list[dict[str, Any]] = field(default_factory=list)

    def sample_id_code(self, sample_id: str) -> int:
        if sample_id not in self.sample_id_index:
            self.sample_id_index[sample_id] = len(self.sample_id_vocab)
            self.sample_id_vocab.append(sample_id)
        return self.sample_id_index[sample_id]


def _proj_of(name: str) -> str | None:
    m = _PROJ_RE.search(name)
    return m.group(1) if m else None


def _layer_idx(name: str) -> int | None:
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


def _as_token_hidden(x: torch.Tensor) -> torch.Tensor:
    """Reshape activation to [T, K]."""
    t = x.detach()
    if t.ndim == 1:
        return t.unsqueeze(0)
    if t.ndim == 2:
        return t
    if t.ndim == 3:
        b, seq, k = t.shape
        return t.reshape(b * seq, k)
    return t.reshape(-1, t.shape[-1])


def _uniform_token_indices(num_tokens: int, max_tokens: int = MAX_PREFILL_STAT_TOKENS) -> torch.Tensor:
    if num_tokens <= max_tokens:
        return torch.arange(num_tokens, dtype=torch.long)
    idx = torch.linspace(0, num_tokens - 1, steps=max_tokens).round().long()
    return torch.unique(idx, sorted=True)


def _sample_seed(
    sample_id: str,
    phase: str,
    decode_step: int,
    layer_idx: int,
    projection: str,
) -> int:
    raw = f"{ANALYSIS_SEED}|{sample_id}|{phase}|{decode_step}|{layer_idx}|{projection}"
    return zlib.adler32(raw.encode()) & 0xFFFFFFFF


def _deterministic_flat_indices(numel: int, max_n: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    if numel <= max_n:
        return torch.arange(numel, dtype=torch.long)
    return torch.randperm(numel, generator=g)[:max_n]


def _expand_nvfp4_element_fields(
    meta: dict[str, Any],
    num_tokens: int,
    hidden: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand NVFP4 block metadata to per-element [T, K] tensors."""
    if "e2m1_payload" not in meta or "e4m3_local_scale" not in meta:
        raise KeyError("NVFP4 metadata missing e2m1_payload / e4m3_local_scale")
    payload = meta["e2m1_payload"].detach().to(device="cpu", dtype=torch.float32)
    local = meta["e4m3_local_scale"].detach().to(device="cpu", dtype=torch.float32)
    payload = payload.reshape(num_tokens, hidden)
    if local.shape != (num_tokens, hidden // 16):
        raise ValueError(
            f"NVFP4 e4m3_local_scale shape {tuple(local.shape)} "
            f"!= ({num_tokens}, {hidden // 16})"
        )
    local_el = local.repeat_interleave(16, dim=-1)
    if local_el.shape != (num_tokens, hidden):
        raise RuntimeError("NVFP4 local scale expand failed")
    return payload, local_el


def _expand_hif4_element_fields(
    meta: dict[str, Any],
    x_tk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """HiF4 payload/local_scale already element-aligned; sign payload for viz."""
    if "payload_magnitude" not in meta or "local_scale" not in meta:
        raise KeyError("HiF4 metadata missing payload_magnitude / local_scale")
    mag = meta["payload_magnitude"].detach().to(device="cpu", dtype=torch.float32).reshape(
        x_tk.shape
    )
    local = meta["local_scale"].detach().to(device="cpu", dtype=torch.float32).reshape(
        x_tk.shape
    )
    sign = torch.sign(x_tk.to(device="cpu", dtype=torch.float32))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    signed_payload = mag * sign
    return signed_payload, local


def _moments_prefixed(prefix: str, x: torch.Tensor) -> dict[str, float]:
    m = basic_tensor_moments(x)
    return {
        f"{prefix}_mean": m["mean"],
        f"{prefix}_std": m["std"],
        f"{prefix}_rms": m["rms"],
        f"{prefix}_zero_rate": m["zero_rate"],
    }


def _record_capture(
    *,
    buffers: CaptureBuffers,
    x_in: torch.Tensor,
    a_n: torch.Tensor,
    a_h: torch.Tensor,
    nv_meta: dict[str, Any],
    hf_meta: dict[str, Any],
    module_name: str,
    layer_idx: int,
    projection: str,
    max_point_samples: int,
) -> None:
    ctx = ACTIVATION_VIZ_CONTEXT
    # Stats/sampling on CPU so the W4A4 forward is not stalled by host syncs.
    x_full = _as_token_hidden(x_in).detach().to(device="cpu", dtype=torch.float32)
    an_full = _as_token_hidden(a_n).detach().to(device="cpu", dtype=torch.float32)
    ah_full = _as_token_hidden(a_h).detach().to(device="cpu", dtype=torch.float32)
    if x_full.shape != an_full.shape or x_full.shape != ah_full.shape:
        raise ValueError(
            f"capture shape mismatch {tuple(x_full.shape)} / "
            f"{tuple(an_full.shape)} / {tuple(ah_full.shape)}"
        )
    t0, k = x_full.shape
    if k % 64 != 0:
        raise ValueError(f"K={k} must be divisible by 64 for HiF4 residual viz")

    nv_payload_full, nv_local_full = _expand_nvfp4_element_fields(nv_meta, t0, k)
    hf_payload_full, hf_local_full = _expand_hif4_element_fields(hf_meta, x_full)

    token_pos_full = torch.arange(t0, dtype=torch.long)
    if ctx.phase == "prefill" and t0 > MAX_PREFILL_STAT_TOKENS:
        keep = _uniform_token_indices(t0, MAX_PREFILL_STAT_TOKENS)
        x_tk = x_full[keep]
        an_tk = an_full[keep]
        ah_tk = ah_full[keep]
        token_pos_full = token_pos_full[keep]
        nv_payload_full = nv_payload_full[keep]
        nv_local_full = nv_local_full[keep]
        hf_payload_full = hf_payload_full[keep]
        hf_local_full = hf_local_full[keep]
    else:
        x_tk = x_full
        an_tk = an_full
        ah_tk = ah_full
    t = int(x_tk.shape[0])

    delta = ah_tk - an_tk
    residual = residual_element_stats(an_tk, ah_tk)
    zero_tr = zero_transition_stats(an_tk, ah_tk)
    energy = residual_energy_concentration(delta)
    summary = {
        "sample_id": ctx.sample_id,
        "prompt_family": ctx.prompt_family,
        "split": ctx.split,
        "phase": ctx.phase,
        "decode_step": int(ctx.decode_step),
        "layer_idx": int(layer_idx),
        "module_name": module_name,
        "projection": projection,
        "num_tokens": int(t),
        "num_elements": int(t * k),
        **_moments_prefixed("x", x_tk),
        **_moments_prefixed("an", an_tk),
        **_moments_prefixed("ah", ah_tk),
        **flatten_stats_for_csv(residual, zero_tr, energy),
    }
    buffers.summary_rows.append(summary)

    for row in group64_residual_stats(x_tk, an_tk, ah_tk):
        buffers.group_rows.append(
            {
                "sample_id": ctx.sample_id,
                "prompt_family": ctx.prompt_family,
                "split": ctx.split,
                "phase": ctx.phase,
                "decode_step": int(ctx.decode_step),
                "layer_idx": int(layer_idx),
                "module_name": module_name,
                "projection": projection,
                **row,
            }
        )

    tg_map = build_token_group_residual_map(delta, group_size=64)
    buffers.token_group_maps.append(
        {
            "sample_id": ctx.sample_id,
            "prompt_family": ctx.prompt_family,
            "split": ctx.split,
            "phase": ctx.phase,
            "decode_step": int(ctx.decode_step),
            "layer_idx": int(layer_idx),
            "module_name": module_name,
            "projection": projection,
            "map": tg_map.detach().cpu().contiguous(),
        }
    )

    numel = int(an_tk.numel())
    seed = _sample_seed(ctx.sample_id, ctx.phase, ctx.decode_step, layer_idx, projection)
    flat_idx = _deterministic_flat_indices(numel, max_point_samples, seed)
    token_idx = torch.div(flat_idx, k, rounding_mode="floor")
    channel_idx = flat_idx % k
    sid = buffers.sample_id_code(ctx.sample_id)
    n = int(flat_idx.numel())
    chunk = {
        "x_in": x_tk.reshape(-1)[flat_idx].cpu().contiguous(),
        "a_nvfp4": an_tk.reshape(-1)[flat_idx].cpu().contiguous(),
        "a_hif4": ah_tk.reshape(-1)[flat_idx].cpu().contiguous(),
        "delta_a": delta.reshape(-1)[flat_idx].cpu().contiguous(),
        "layer_idx": torch.full((n,), int(layer_idx), dtype=torch.int32),
        "projection_id": torch.full((n,), int(_PROJ_TO_ID[projection]), dtype=torch.uint8),
        "phase_id": torch.full((n,), int(_PHASE_TO_ID[ctx.phase]), dtype=torch.uint8),
        "decode_step": torch.full((n,), int(ctx.decode_step), dtype=torch.int16),
        "token_position": token_pos_full[token_idx].to(torch.int16).cpu().contiguous(),
        "channel_idx": channel_idx.to(torch.int32).cpu().contiguous(),
        "nv_payload": nv_payload_full.reshape(-1)[flat_idx].cpu().contiguous(),
        "hf_payload": hf_payload_full.reshape(-1)[flat_idx].cpu().contiguous(),
        "nv_local_scale": nv_local_full.reshape(-1)[flat_idx].cpu().contiguous(),
        "hf_local_scale": hf_local_full.reshape(-1)[flat_idx].cpu().contiguous(),
        "sample_id": torch.full((n,), int(sid), dtype=torch.int32),
    }
    buffers.point_chunks.append(chunk)


def _make_recording_hook(
    module_name: str,
    layer_idx: int,
    projection: str,
    scale: torch.Tensor,
    *,
    is_representative: bool,
    buffers: CaptureBuffers,
    max_point_samples: int = 1024,
):
    """Single forward_pre_hook: always return A_N; optionally record A_H off-path."""
    scale_cpu = scale.detach().to(torch.float32).reshape(()).cpu()

    def hook(_mod: nn.Module, inputs: tuple[Any, ...]):
        x_in = inputs[0]
        if not torch.is_tensor(x_in):
            raise TypeError(f"expected tensor input for {module_name}, got {type(x_in)}")
        xb = x_in.to(torch.bfloat16)
        scale_dev = scale_cpu.to(device=xb.device, dtype=torch.float32)
        ctx = ACTIVATION_VIZ_CONTEXT
        do_record = bool(is_representative and ctx.record_enabled)
        nv_view = quantize_nvfp4_activation(
            xb,
            scale_dev,
            output_dtype=x_in.dtype,
            collect_metadata=do_record,
        )
        a_n = nv_view.dequantized
        if do_record:
            hf_view = quantize_hif4_tensor(xb, output_dtype=x_in.dtype)
            _record_capture(
                buffers=buffers,
                x_in=xb.detach(),
                a_n=a_n.detach(),
                a_h=hf_view.dequantized.detach(),
                nv_meta=nv_view.metadata,
                hf_meta=hf_view.metadata,
                module_name=module_name,
                layer_idx=layer_idx,
                projection=projection,
                max_point_samples=max_point_samples,
            )
        return (a_n,) + tuple(inputs[1:])

    return hook


def install_activation_viz_hooks(
    model: nn.Module,
    scales: dict[str, torch.Tensor],
    *,
    representative_layers: set[int],
    buffers: CaptureBuffers,
    max_point_samples: int = 1024,
) -> list[Any]:
    """Install NVFP4 A4 hooks on all target Linears (exclude lm_head)."""
    handles: list[Any] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if "lm_head" in name:
            continue
        proj = _proj_of(name)
        if proj is None or proj not in _PROJ_TO_ID:
            continue
        li = _layer_idx(name)
        if li is None:
            continue
        scale = resolve_nvfp4_scale_for_module(scales, name)
        handles.append(
            mod.register_forward_pre_hook(
                _make_recording_hook(
                    name,
                    li,
                    proj,
                    scale,
                    is_representative=li in representative_layers,
                    buffers=buffers,
                    max_point_samples=max_point_samples,
                )
            )
        )
    if not handles:
        raise RuntimeError("no target Linear hooks installed")
    return handles


def _cat_point_chunks(
    chunks: list[dict[str, torch.Tensor]],
    sample_ids: list[str],
) -> dict[str, Any]:
    if not chunks:
        empty = {
            "x_in": torch.empty(0, dtype=torch.float32),
            "a_nvfp4": torch.empty(0, dtype=torch.float32),
            "a_hif4": torch.empty(0, dtype=torch.float32),
            "delta_a": torch.empty(0, dtype=torch.float32),
            "layer_idx": torch.empty(0, dtype=torch.int32),
            "projection_id": torch.empty(0, dtype=torch.uint8),
            "phase_id": torch.empty(0, dtype=torch.uint8),
            "decode_step": torch.empty(0, dtype=torch.int16),
            "token_position": torch.empty(0, dtype=torch.int16),
            "channel_idx": torch.empty(0, dtype=torch.int32),
            "nv_payload": torch.empty(0, dtype=torch.float32),
            "hf_payload": torch.empty(0, dtype=torch.float32),
            "nv_local_scale": torch.empty(0, dtype=torch.float32),
            "hf_local_scale": torch.empty(0, dtype=torch.float32),
            "sample_id": torch.empty(0, dtype=torch.int32),
            "projection_names": list(LINEAR_PROJECTIONS),
            "phase_names": list(_PHASE_NAMES),
            "sample_ids": list(sample_ids),
        }
        return empty
    keys = [
        "x_in",
        "a_nvfp4",
        "a_hif4",
        "delta_a",
        "layer_idx",
        "projection_id",
        "phase_id",
        "decode_step",
        "token_position",
        "channel_idx",
        "nv_payload",
        "hf_payload",
        "nv_local_scale",
        "hf_local_scale",
        "sample_id",
    ]
    out: dict[str, Any] = {k: torch.cat([c[k] for c in chunks], dim=0) for k in keys}
    n = int(out["x_in"].numel())
    for k in keys:
        if int(out[k].shape[0]) != n:
            raise RuntimeError(f"points field length mismatch: {k}")
    out["projection_names"] = list(LINEAR_PROJECTIONS)
    out["phase_names"] = list(_PHASE_NAMES)
    out["sample_ids"] = list(sample_ids)
    return out


def _assert_finite_points(points: dict[str, Any]) -> None:
    for key in (
        "x_in",
        "a_nvfp4",
        "a_hif4",
        "delta_a",
        "nv_payload",
        "hf_payload",
        "nv_local_scale",
        "hf_local_scale",
    ):
        t = points[key]
        if not torch.is_tensor(t) or t.numel() == 0:
            continue
        if not torch.isfinite(t).all():
            bad = (~torch.isfinite(t)).sum().item()
            raise RuntimeError(f"points[{key}] contains {bad} NaN/Inf values")


@torch.no_grad()
def run_activation_viz_shard(
    checkpoint: Path,
    out_dir: Path,
    *,
    device: str,
    shard_id: int,
    num_shards: int,
    split: str,
    samples_per_family: int = 8,
    max_seq_len: int = 256,
    decode_steps: int = 8,
    max_point_samples_per_capture: int = 1024,
) -> dict[str, Any]:
    """Run one shard of W4A4 online capture on discovery or validation prompts."""
    if split not in {"discovery", "validation"}:
        raise ValueError(f"split must be discovery|validation, got {split!r}")
    if num_shards < 1 or shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"invalid shard_id={shard_id} num_shards={num_shards}")

    out_dir = ensure_dir(out_dir)
    checkpoint = Path(checkpoint)
    torch.set_num_threads(int(__import__("os").environ.get("TORCH_NUM_THREADS", "4")))
    device_t = torch.device(device)
    if device_t.type == "cuda":
        torch.cuda.set_device(device_t)

    with (checkpoint / "config.json").open("r", encoding="utf-8") as f:
        num_layers = int(json.load(f)["num_hidden_layers"])
    rep_layers = set(resolve_representative_layers(num_layers))

    items = discovery_items(samples_per_family) if split == "discovery" else validation_items(
        samples_per_family
    )
    prompts = [p for i, p in enumerate(items) if i % num_shards == shard_id]

    model, tok = load_source_model_for_capture(checkpoint, device=device_t)
    scales = load_nvfp4_activation_scales(resolve_activation_scale_path(checkpoint))
    buffers = CaptureBuffers()
    handles = install_activation_viz_hooks(
        model,
        scales,
        representative_layers=rep_layers,
        buffers=buffers,
        max_point_samples=max_point_samples_per_capture,
    )

    try:
        for item in prompts:
            enc = tok(
                item.text,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_len,
            )
            batch = {k: v.to(device_t) for k, v in enc.items()}
            if "attention_mask" not in batch:
                batch["attention_mask"] = torch.ones_like(batch["input_ids"])

            ACTIVATION_VIZ_CONTEXT.sample_id = item.sample_id
            ACTIVATION_VIZ_CONTEXT.prompt_family = item.family
            ACTIVATION_VIZ_CONTEXT.split = item.split
            ACTIVATION_VIZ_CONTEXT.phase = "prefill"
            ACTIVATION_VIZ_CONTEXT.decode_step = -1
            ACTIVATION_VIZ_CONTEXT.record_enabled = True
            print(
                f"[activation-viz] shard{shard_id} start {item.sample_id} prefill",
                flush=True,
            )

            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=True,
            )
            past = out.past_key_values
            next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([batch["input_ids"], next_id], dim=-1)
            attn = torch.cat(
                [
                    batch["attention_mask"],
                    torch.ones(
                        (batch["attention_mask"].shape[0], 1),
                        device=device_t,
                        dtype=batch["attention_mask"].dtype,
                    ),
                ],
                dim=-1,
            )

            for step in range(decode_steps):
                ACTIVATION_VIZ_CONTEXT.phase = "decode"
                ACTIVATION_VIZ_CONTEXT.decode_step = int(step)
                ACTIVATION_VIZ_CONTEXT.record_enabled = True
                step_out = model(
                    input_ids=input_ids[:, -1:],
                    attention_mask=attn,
                    past_key_values=past,
                    use_cache=True,
                )
                past = step_out.past_key_values
                next_id = step_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_id], dim=-1)
                attn = torch.cat(
                    [
                        attn,
                        torch.ones(
                            (attn.shape[0], 1),
                            device=device_t,
                            dtype=attn.dtype,
                        ),
                    ],
                    dim=-1,
                )

            ACTIVATION_VIZ_CONTEXT.record_enabled = False
            print(f"[activation-viz] shard{shard_id} {item.sample_id}", flush=True)
    finally:
        for h in handles:
            h.remove()
        ACTIVATION_VIZ_CONTEXT.record_enabled = False

    write_csv(out_dir / f"activation_capture_summary_shard{shard_id}.csv", buffers.summary_rows)
    write_csv(out_dir / f"activation_group_residual_shard{shard_id}.csv", buffers.group_rows)
    points = _cat_point_chunks(buffers.point_chunks, buffers.sample_id_vocab)
    _assert_finite_points(points)
    torch.save(points, out_dir / f"activation_viz_points_shard{shard_id}.pt")
    torch.save(
        {"entries": buffers.token_group_maps},
        out_dir / f"activation_token_group_maps_shard{shard_id}.pt",
    )
    summary = {
        "shard_id": shard_id,
        "num_shards": num_shards,
        "split": split,
        "num_prompts": len(prompts),
        "num_summary_rows": len(buffers.summary_rows),
        "num_group_rows": len(buffers.group_rows),
        "num_points": int(points["x_in"].numel()),
        "num_token_group_maps": len(buffers.token_group_maps),
        "representative_layers": sorted(rep_layers),
        "max_point_samples_per_capture": max_point_samples_per_capture,
    }
    atomic_write_json(out_dir / f"activation_viz_summary_shard{shard_id}.json", summary)
    del model
    if device_t.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def jensen_shannon(p: torch.Tensor, q: torch.Tensor) -> float:
    """Jensen-Shannon divergence for two discrete distributions (nats→bits via log2)."""
    p = p.to(torch.float64).reshape(-1)
    q = q.to(torch.float64).reshape(-1)
    if p.numel() != q.numel():
        raise ValueError("histogram length mismatch")
    if float(p.sum()) <= 0 or float(q.sum()) <= 0:
        raise ValueError("empty histogram")
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    # Skip zeros in KL; 0 log(0/m)=0 when p=0.
    def _kl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        mask = a > 0
        return torch.sum(a[mask] * (torch.log2(a[mask]) - torch.log2(b[mask])))

    return float((0.5 * (_kl(p, m) + _kl(q, m))).item())


def _hist_probs(values: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    v = values.detach().to(torch.float32).reshape(-1)
    v = v[torch.isfinite(v)]
    if v.numel() == 0:
        return torch.zeros(int(edges.numel() - 1), dtype=torch.float64)
    # torch.bucketize / histogram
    counts = torch.histogram(v.cpu(), bins=edges.cpu()).hist.to(torch.float64)
    return counts


def compute_split_stability(
    discovery_points: dict[str, Any],
    validation_points: dict[str, Any],
    *,
    num_bins: int = 64,
) -> dict[str, Any]:
    """JS stability between discovery/validation point samples (bin edges from discovery)."""
    an_d = discovery_points["a_nvfp4"].to(torch.float32)
    an_v = validation_points["a_nvfp4"].to(torch.float32)
    da_d = discovery_points["delta_a"].to(torch.float32)
    da_v = validation_points["delta_a"].to(torch.float32)

    def _signed_edges(x: torch.Tensor) -> torch.Tensor:
        finite = x[torch.isfinite(x)]
        if finite.numel() == 0:
            raise ValueError("no finite values for edges")
        lo = float(finite.min().item())
        hi = float(finite.max().item())
        if lo == hi:
            hi = lo + 1.0
        return torch.linspace(lo, hi, num_bins + 1)

    def _log_abs_edges(x: torch.Tensor, base: float) -> torch.Tensor:
        ax = x.abs()
        ax = ax[ax > 0]
        if ax.numel() == 0:
            return torch.linspace(-10.0, 0.0, num_bins + 1)
        if base == 2.0:
            lx = torch.log2(ax)
        else:
            lx = torch.log10(ax)
        return _signed_edges(lx)

    specs = {
        "an_signed": (an_d, an_v, _signed_edges(an_d), False, 0.0),
        "an_log2_abs": (
            an_d,
            an_v,
            _log_abs_edges(an_d, 2.0),
            True,
            2.0,
        ),
        "delta_signed": (da_d, da_v, _signed_edges(da_d), False, 0.0),
        "delta_log10_abs": (
            da_d,
            da_v,
            _log_abs_edges(da_d, 10.0),
            True,
            10.0,
        ),
    }

    def _transform(x: torch.Tensor, log_abs: bool, base: float) -> torch.Tensor:
        if not log_abs:
            return x
        ax = x.abs()
        ax = ax[ax > 0]
        return torch.log2(ax) if base == 2.0 else torch.log10(ax)

    global_js: dict[str, float] = {}
    for name, (xd, xv, edges, log_abs, base) in specs.items():
        pd = _hist_probs(_transform(xd, log_abs, base), edges)
        pv = _hist_probs(_transform(xv, log_abs, base), edges)
        # Avoid zero-mass histograms
        pd = pd + 1e-12
        pv = pv + 1e-12
        global_js[name] = jensen_shannon(pd, pv)

    per_proj: dict[str, dict[str, float]] = {}
    for proj in LINEAR_PROJECTIONS:
        pid = _PROJ_TO_ID[proj]
        md = discovery_points["projection_id"] == pid
        mv = validation_points["projection_id"] == pid
        if int(md.sum()) == 0 or int(mv.sum()) == 0:
            continue
        per_proj[proj] = {}
        for name, (xd, xv, edges, log_abs, base) in specs.items():
            pd = _hist_probs(_transform(xd[md], log_abs, base), edges) + 1e-12
            pv = _hist_probs(_transform(xv[mv], log_abs, base), edges) + 1e-12
            per_proj[proj][name] = jensen_shannon(pd, pv)

    max_global = max(global_js.values()) if global_js else float("nan")
    bad_proj = [
        p
        for p, js in per_proj.items()
        if any(v > 0.05 for v in js.values())
    ]
    stable = (max_global <= 0.03) and (len(bad_proj) < 2)
    return {
        "global_js": global_js,
        "per_projection_js": per_proj,
        "max_global_js": max_global,
        "projections_exceeding_0p05": bad_proj,
        "stable": stable,
        "recommend_samples_per_family_32": not stable,
        "thresholds": {"global": 0.03, "per_projection": 0.05},
    }


def attach_theoretical_grids(
    run_dir: Path,
    *,
    consolidated_dir: Path | None = None,
) -> dict[str, Any]:
    """Attach AX3 full internal grids (E4M3FN×E2M1) into run_dir."""
    run_dir = ensure_dir(run_dir)
    src = Path(consolidated_dir) if consolidated_dir is not None else DEFAULT_AX3_CONSOLIDATED
    nv_pt = src / "ax3_nvfp4_full_internal_grid.pt"
    hf_pt = src / "ax3_hif4_full_internal_grid.pt"
    meta_json = src / "ax3_theoretical_grid.json"

    payload: dict[str, Any]
    if nv_pt.is_file() and hf_pt.is_file() and meta_json.is_file():
        meta = json.loads(meta_json.read_text(encoding="utf-8"))
        if "nvfp4_full_internal_grid" not in meta or "hif4_full_internal_grid" not in meta:
            payload = build_theoretical_grid_json(run_dir)
            source = "rebuilt_missing_json_fields"
        else:
            nv = torch.load(nv_pt, map_location="cpu", weights_only=True)
            hf = torch.load(hf_pt, map_location="cpu", weights_only=True)
            torch.save(nv, run_dir / "ax3_nvfp4_full_internal_grid.pt")
            torch.save(hf, run_dir / "ax3_hif4_full_internal_grid.pt")
            # Keep JSON compact: drop huge embedded lists if present, point to local pt.
            payload = dict(meta)
            payload["nvfp4_full_internal_grid_path"] = str(
                run_dir / "ax3_nvfp4_full_internal_grid.pt"
            )
            payload["hif4_full_internal_grid_path"] = str(
                run_dir / "ax3_hif4_full_internal_grid.pt"
            )
            payload.pop("nvfp4_full_internal_grid", None)
            payload.pop("hif4_full_internal_grid", None)
            atomic_write_json(run_dir / "ax3_theoretical_grid.json", payload)
            source = "consolidated_ax3"
            return {
                "source": source,
                "nvfp4_num_unique": int(nv.numel()) if torch.is_tensor(nv) else None,
                "hif4_num_unique": int(hf.numel()) if torch.is_tensor(hf) else None,
                "scale_note": payload.get("note"),
            }
    else:
        payload = build_theoretical_grid_json(run_dir)
        source = "rebuilt"

    atomic_write_json(run_dir / "ax3_theoretical_grid.json", payload)
    return {
        "source": source,
        "nvfp4_num_unique": int(payload["nvfp4_full_stats"]["num_unique"]),
        "hif4_num_unique": int(payload["hif4_full_stats"]["num_unique"]),
        "scale_note": payload.get("note"),
    }


def merge_activation_viz_shards(run_dir: Path) -> dict[str, Any]:
    """Merge shard CSV/PT artifacts; attach theoretical grids; optional JS if both splits."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)

    def _merge_csv(pattern: str, out_name: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        paths = sorted(run_dir.glob(pattern))
        seen_shards: list[int] = []
        for p in paths:
            # activation_*_shard{N}.csv
            stem = p.stem
            if "_shard" not in stem:
                raise ValueError(f"unexpected shard filename: {p.name}")
            sid = int(stem.rsplit("_shard", 1)[1])
            if sid in seen_shards:
                raise RuntimeError(f"duplicate shard id {sid} for {pattern}")
            seen_shards.append(sid)
            rows.extend(csv.DictReader(p.open(encoding="utf-8")))
        write_csv(run_dir / out_name, rows)
        return rows

    summary_rows = _merge_csv(
        "activation_capture_summary_shard*.csv",
        "activation_capture_summary.csv",
    )
    group_rows = _merge_csv(
        "activation_group_residual_shard*.csv",
        "activation_group_residual.csv",
    )

    point_paths = sorted(run_dir.glob("activation_viz_points_shard*.pt"))
    map_paths = sorted(run_dir.glob("activation_token_group_maps_shard*.pt"))
    if not point_paths:
        raise FileNotFoundError(f"no activation_viz_points_shard*.pt under {run_dir}")

    # Remap sample_id vocab across shards.
    merged_chunks: list[dict[str, torch.Tensor]] = []
    vocab: list[str] = []
    vocab_index: dict[str, int] = {}
    for p in point_paths:
        pts = torch.load(p, map_location="cpu", weights_only=False)
        local_ids: list[str] = list(pts.get("sample_ids", []))
        if "projection_names" in pts and list(pts["projection_names"]) != list(LINEAR_PROJECTIONS):
            raise RuntimeError(f"projection_names mismatch in {p}")
        sid_local = pts["sample_id"].to(torch.int64)
        sid_global = torch.empty_like(sid_local)
        for local_i, name in enumerate(local_ids):
            if name not in vocab_index:
                vocab_index[name] = len(vocab)
                vocab.append(name)
            sid_global[sid_local == local_i] = vocab_index[name]
        chunk = {k: v for k, v in pts.items() if torch.is_tensor(v)}
        chunk["sample_id"] = sid_global.to(torch.int32)
        merged_chunks.append(chunk)

    points = _cat_point_chunks(merged_chunks, vocab)
    _assert_finite_points(points)
    torch.save(points, run_dir / "activation_viz_points.pt")

    entries: list[dict[str, Any]] = []
    for p in map_paths:
        blob = torch.load(p, map_location="cpu", weights_only=False)
        entries.extend(blob.get("entries", []))
    torch.save({"entries": entries}, run_dir / "activation_token_group_maps.pt")

    # Basic row sanity
    for r in summary_rows:
        if r.get("split") not in {"discovery", "validation"}:
            raise RuntimeError(f"bad split in summary: {r.get('split')}")
        if not r.get("sample_id"):
            raise RuntimeError("summary row missing sample_id")

    grid_info = attach_theoretical_grids(run_dir)

    stability: dict[str, Any] | None = None
    splits = {r.get("split") for r in summary_rows}
    if splits == {"discovery", "validation"}:
        # Split points by summary is hard; use sample_id vocab split via summary table.
        disc_ids = {r["sample_id"] for r in summary_rows if r["split"] == "discovery"}
        val_ids = {r["sample_id"] for r in summary_rows if r["split"] == "validation"}
        name_to_i = {n: i for i, n in enumerate(points["sample_ids"])}
        d_codes = [name_to_i[s] for s in disc_ids if s in name_to_i]
        v_codes = [name_to_i[s] for s in val_ids if s in name_to_i]
        if d_codes and v_codes:
            sid = points["sample_id"]
            d_mask = torch.zeros(sid.shape[0], dtype=torch.bool)
            v_mask = torch.zeros(sid.shape[0], dtype=torch.bool)
            for c in d_codes:
                d_mask |= sid == c
            for c in v_codes:
                v_mask |= sid == c
            if bool(d_mask.any()) and bool(v_mask.any()):
                d_pts = {k: (v[d_mask] if torch.is_tensor(v) else v) for k, v in points.items()}
                v_pts = {k: (v[v_mask] if torch.is_tensor(v) else v) for k, v in points.items()}
                stability = compute_split_stability(d_pts, v_pts)

    summary = {
        "run_id": run_dir.name,
        "num_summary_rows": len(summary_rows),
        "num_group_rows": len(group_rows),
        "num_points": int(points["x_in"].numel()),
        "num_token_group_maps": len(entries),
        "num_point_shards": len(point_paths),
        "projection_names": list(LINEAR_PROJECTIONS),
        "phase_names": list(_PHASE_NAMES),
        "theoretical_grids": grid_info,
        "stability": stability,
        "analysis_seed": ANALYSIS_SEED,
        "max_prefill_stat_tokens": MAX_PREFILL_STAT_TOKENS,
    }
    atomic_write_json(run_dir / "activation_viz_summary.json", summary)
    return summary
