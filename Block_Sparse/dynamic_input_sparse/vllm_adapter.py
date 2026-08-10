from __future__ import annotations

from typing import Any

import torch

from Block_Sparse.dynamic_input_sparse.common import (
    classify_mlp_prefix,
    flatten_tokens,
    is_target_mlp_prefix,
    ratio_to_keep_count,
)
from Block_Sparse.dynamic_input_sparse.config import (
    DynamicInputMaskMethod,
    DynamicInputSparseConfig,
    config_from_additional,
)
from Block_Sparse.dynamic_input_sparse.m1_oracle import predict_m1_full_output_mask
from Block_Sparse.dynamic_input_sparse.m8_energy import (
    compute_all_output_weight_energy,
    predict_m8_input_mask,
)
from Block_Sparse.dynamic_input_sparse.masked_linear import apply_input_block_mask
from Block_Sparse.dynamic_input_sparse.telemetry import _Timer, get_telemetry

# Re-export for callers/tests
__all__ = [
    "is_target_mlp_prefix",
    "setup_dynamic_input_sparse_for_layer",
    "mask_linear_input",
    "runtime_config_from_additional",
    "validate_parallelism_for_dynamic_input",
]


def runtime_config_from_additional(
    additional_config: dict | None,
) -> DynamicInputSparseConfig | None:
    return config_from_additional(additional_config)


def validate_parallelism_for_dynamic_input(
    *,
    tensor_parallel_size: int,
    method: DynamicInputMaskMethod | str,
) -> None:
    if isinstance(method, DynamicInputMaskMethod):
        method_e = method
    else:
        method_e = DynamicInputMaskMethod(str(method))
    if method_e == DynamicInputMaskMethod.NONE:
        return
    if int(tensor_parallel_size) > 1:
        raise RuntimeError(
            "dynamic input sparsity requires TP=1 in this experiment; "
            "TP-aware global mask coordination is not implemented "
            f"(got tensor_parallel_size={tensor_parallel_size})"
        )


def setup_dynamic_input_sparse_for_layer(
    layer: torch.nn.Module,
    runtime_config: DynamicInputSparseConfig | None,
) -> bool:
    """Initialize targeted MLP layer. Returns True if layer is a target."""
    prefix = getattr(layer, "prefix", "") or ""
    layer._dynamic_input_sparse_enabled = False
    layer._dynamic_input_sparse_config = runtime_config
    if runtime_config is None:
        return False
    if runtime_config.method == DynamicInputMaskMethod.NONE:
        return False
    kind = classify_mlp_prefix(prefix)
    if kind is None:
        if "expert" in prefix.lower() and "mlp" in prefix:
            raise RuntimeError(
                f"MoE expert path not supported for this dense-model experiment: {prefix}"
            )
        return False

    weight = layer.weight
    d_out, d_in = int(weight.shape[0]), int(weight.shape[1])
    if d_in % runtime_config.k_block_size != 0:
        raise ValueError(
            f"prefix={prefix}: D_in={d_in} not divisible by "
            f"{runtime_config.k_block_size}"
        )
    if runtime_config.method == DynamicInputMaskMethod.M8_ENERGY:
        if d_out % runtime_config.output_energy_block_size != 0:
            raise ValueError(
                f"prefix={prefix}: D_out={d_out} not divisible by "
                f"{runtime_config.output_energy_block_size}"
            )
        g_w = compute_all_output_weight_energy(
            weight,
            k_block_size=runtime_config.k_block_size,
            output_block_size=runtime_config.output_energy_block_size,
        )
        layer.register_buffer(
            "_dynamic_input_g_w",
            g_w.to(device=weight.device, dtype=torch.float32),
            persistent=False,
        )
    layer._dynamic_input_sparse_enabled = True
    layer._dynamic_input_sparse_kind = kind
    # One-time per-process confirmation (avoid per-forward spam).
    if not getattr(setup_dynamic_input_sparse_for_layer, "_logged", False):
        print(
            f"[dynamic_input_sparse] enabled on targets; first={prefix} "
            f"method={runtime_config.method.value} keep={runtime_config.keep_ratio}",
            flush=True,
        )
        setup_dynamic_input_sparse_for_layer._logged = True  # type: ignore[attr-defined]
    return True


def mask_linear_input(
    layer: torch.nn.Module,
    x: torch.Tensor,
    runtime_config: DynamicInputSparseConfig | None = None,
) -> torch.Tensor:
    """Apply dynamic input mask for a targeted layer; pass-through otherwise."""
    cfg = runtime_config or getattr(layer, "_dynamic_input_sparse_config", None)
    if cfg is None or cfg.method == DynamicInputMaskMethod.NONE:
        return x
    if not bool(getattr(layer, "_dynamic_input_sparse_enabled", False)):
        return x

    prefix = getattr(layer, "prefix", "") or ""
    x_flat, _ = flatten_tokens(x)
    t = int(x_flat.shape[0])
    d_in = int(x_flat.shape[1])
    kb = d_in // cfg.k_block_size
    keep_count = ratio_to_keep_count(cfg.keep_ratio, kb) if float(cfg.keep_ratio) < 1.0 else kb

    with _Timer() as t_pred:
        if float(cfg.keep_ratio) == 1.0:
            mx = torch.ones(t, kb, dtype=torch.bool, device=x.device)
        elif cfg.method == DynamicInputMaskMethod.M8_ENERGY:
            g_w = getattr(layer, "_dynamic_input_g_w", None)
            if g_w is None:
                raise RuntimeError(f"M8 G_W missing on layer {prefix}")
            mx = predict_m8_input_mask(
                x, g_w, cfg.keep_ratio, k_block_size=cfg.k_block_size
            )
        elif cfg.method == DynamicInputMaskMethod.M1_ORACLE:
            mx = predict_m1_full_output_mask(
                x,
                layer.weight,
                cfg.keep_ratio,
                token_chunk_size=cfg.m1_token_chunk_size,
                k_block_size=cfg.k_block_size,
            )
        else:
            raise ValueError(f"unknown dynamic method: {cfg.method}")
    pred_s = t_pred.elapsed

    with _Timer() as t_apply:
        x_masked = apply_input_block_mask(x, mx, block_size=cfg.k_block_size)
    apply_s = t_apply.elapsed

    realized = float(mx.float().mean().item())
    # Exact keep count check
    per_token = mx.sum(dim=-1)
    if not bool(torch.all(per_token == keep_count).item()):
        raise RuntimeError(
            f"realized keep count mismatch on {prefix}: "
            f"expected {keep_count}, got {per_token.tolist()[:8]}..."
        )

    tel = get_telemetry()
    if tel is not None:
        tel.record(
            prefix,
            tokens=t,
            kb=kb,
            keep_count=keep_count,
            realized_keep=realized,
            predictor_time_s=pred_s,
            mask_apply_time_s=apply_s,
            mx=mx,
        )
    return x_masked
