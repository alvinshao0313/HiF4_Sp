"""Pure tensor math for HiF4 deployment-equivalent static scaling.

This module deliberately has no model loading, hooks, plotting, or checkpoint IO.
All functions operate on explicit tensors so the deployment folding algebra can be
unit-tested independently from the experiment pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

_ALLOWED_DOMAINS = {"attn_in", "mlp_in", "down_in", "o_in"}
_ALLOWED_KINDS = {"pts_layer", "phase_g64", "equalize", "equalize_aw"}
_ALLOWED_EQUALIZATION_GRANULARITIES = {1, 4, 8, 16}


@dataclass(frozen=True)
class ScalingSpec:
    kind: str
    domain: str
    granularity: int
    alpha: float
    min_scale: float = 0.5
    max_scale: float = 2.0
    gqa_tied: bool = False

    def __post_init__(self) -> None:
        if self.kind not in _ALLOWED_KINDS:
            raise ValueError(f"unsupported scaling kind {self.kind!r}")
        if self.domain not in _ALLOWED_DOMAINS:
            raise ValueError(f"unsupported scaling domain {self.domain!r}")
        if self.min_scale <= 0 or self.max_scale <= 0 or self.min_scale > self.max_scale:
            raise ValueError("scale bounds must satisfy 0 < min_scale <= max_scale")
        if self.kind in {"equalize", "equalize_aw"}:
            if self.granularity not in _ALLOWED_EQUALIZATION_GRANULARITIES:
                raise ValueError(
                    f"equalization granularity must be one of "
                    f"{sorted(_ALLOWED_EQUALIZATION_GRANULARITIES)}, got {self.granularity}"
                )
        elif self.kind == "pts_layer" and self.granularity != 0:
            raise ValueError("pts_layer granularity must be 0")
        elif self.kind == "phase_g64" and self.granularity != 64:
            raise ValueError("phase_g64 granularity must be 64")


def _require_1d(name: str, x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape={tuple(x.shape)}")
    return x


def _validate_positive_scale(d: torch.Tensor) -> None:
    if not torch.isfinite(d).all():
        raise ValueError("scale tensor contains non-finite values")
    if bool((d <= 0).any()):
        raise ValueError("all scale values must be strictly positive")


def update_channel_stats(
    sum_sq: torch.Tensor,
    max_abs: torch.Tensor,
    count: int,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Accumulate FP64 per-channel sufficient statistics over all leading dims."""
    if x.ndim < 1:
        raise ValueError("activation tensor must have at least one dimension")
    k = int(x.shape[-1])
    _require_1d("sum_sq", sum_sq)
    _require_1d("max_abs", max_abs)
    if sum_sq.numel() != k or max_abs.numel() != k:
        raise ValueError(
            f"stat width mismatch: activation K={k}, sum_sq={sum_sq.numel()}, "
            f"max_abs={max_abs.numel()}"
        )
    flat = x.detach().to(dtype=torch.float64).reshape(-1, k)
    if flat.shape[0] == 0:
        return sum_sq.to(torch.float64), max_abs.to(torch.float64), int(count)
    ss = sum_sq.to(device=flat.device, dtype=torch.float64) + (flat * flat).sum(dim=0)
    ma = torch.maximum(
        max_abs.to(device=flat.device, dtype=torch.float64),
        flat.abs().amax(dim=0),
    )
    return ss, ma, int(count) + int(flat.shape[0])


def finalize_channel_amplitude(
    sum_sq: torch.Tensor,
    max_abs: torch.Tensor,
    count: int,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Return sqrt(max_abs * RMS), preserving exact zero channels as zero."""
    del eps  # epsilon is intentionally not injected into the physical amplitude.
    _require_1d("sum_sq", sum_sq)
    _require_1d("max_abs", max_abs)
    if sum_sq.shape != max_abs.shape:
        raise ValueError("sum_sq and max_abs must have identical shape")
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    ss = sum_sq.to(torch.float64)
    ma = max_abs.to(device=ss.device, dtype=torch.float64)
    if bool((ss < 0).any()) or bool((ma < 0).any()):
        raise ValueError("sum_sq/max_abs must be non-negative")
    rms = torch.sqrt(ss / float(count))
    return torch.sqrt(ma * rms)


def candidate_pts_scales(
    *,
    log2_min: float = -1.0,
    log2_max: float = 1.0,
    points: int = 33,
) -> torch.Tensor:
    if points < 2:
        raise ValueError("points must be >= 2")
    if log2_min > log2_max:
        raise ValueError("log2_min must be <= log2_max")
    return torch.exp2(
        torch.linspace(float(log2_min), float(log2_max), int(points), dtype=torch.float32)
    )


def expand_group_scales(
    group_scales: torch.Tensor,
    *,
    width: int,
    group_size: int = 64,
) -> torch.Tensor:
    _require_1d("group_scales", group_scales)
    if group_size <= 0 or width <= 0 or width % group_size != 0:
        raise ValueError("width must be positive and divisible by group_size")
    expected = width // group_size
    if group_scales.numel() != expected:
        raise ValueError(
            f"expected {expected} group scales for width={width}, got {group_scales.numel()}"
        )
    _validate_positive_scale(group_scales)
    return group_scales.repeat_interleave(group_size)


def _group_unit_max(
    values: torch.Tensor,
    *,
    granularity: int,
    group_size: int,
) -> torch.Tensor:
    _require_1d("values", values)
    if granularity not in _ALLOWED_EQUALIZATION_GRANULARITIES:
        raise ValueError(
            f"granularity must be one of {sorted(_ALLOWED_EQUALIZATION_GRANULARITIES)}, "
            f"got {granularity}"
        )
    if group_size <= 0 or group_size % granularity != 0:
        raise ValueError("group_size must be positive and divisible by granularity")
    if values.numel() == 0 or values.numel() % group_size != 0:
        raise ValueError("value width must be non-zero and divisible by group_size")
    if bool((values < 0).any()) or not torch.isfinite(values).all():
        raise ValueError("amplitude-like values must be finite and non-negative")
    groups = values.reshape(-1, group_size)
    return groups.reshape(groups.shape[0], group_size // granularity, granularity).amax(dim=-1)


def _unit_log_scales_from_amplitude(
    amplitude: torch.Tensor,
    *,
    granularity: int,
    alpha: float,
    group_size: int,
    min_scale: float,
    max_scale: float,
    eps: float,
) -> torch.Tensor:
    if min_scale <= 0 or max_scale <= 0 or min_scale > max_scale:
        raise ValueError("scale bounds must satisfy 0 < min_scale <= max_scale")
    unit_amp = _group_unit_max(
        amplitude.to(torch.float32), granularity=granularity, group_size=group_size
    )
    active = unit_amp > float(eps)
    z = torch.zeros_like(unit_amp)
    for gi in range(unit_amp.shape[0]):
        mask = active[gi]
        if not bool(mask.any()):
            continue
        loga = torch.log2(unit_amp[gi, mask])
        center = loga.mean()
        z[gi, mask] = float(alpha) * (loga - center)
    lo = float(torch.log2(torch.tensor(float(min_scale))).item())
    hi = float(torch.log2(torch.tensor(float(max_scale))).item())
    z = z.clamp(min=lo, max=hi)
    d_unit = torch.exp2(z)
    d_unit = torch.where(active, d_unit, torch.ones_like(d_unit))
    return d_unit


def build_equalization_scale(
    amplitude: torch.Tensor,
    *,
    granularity: int,
    alpha: float,
    group_size: int = 64,
    min_scale: float = 0.5,
    max_scale: float = 2.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Build bounded, group-centered static amplitude equalization scale."""
    _require_1d("amplitude", amplitude)
    d_unit = _unit_log_scales_from_amplitude(
        amplitude,
        granularity=granularity,
        alpha=alpha,
        group_size=group_size,
        min_scale=min_scale,
        max_scale=max_scale,
        eps=eps,
    )
    return d_unit.reshape(-1).repeat_interleave(granularity).to(torch.float32)


def shared_input_weight_stat(weights: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Max |W[:,j]| over output rows and all Linears sharing one input domain."""
    if not weights:
        raise ValueError("weights must be non-empty")
    width: int | None = None
    stats: list[torch.Tensor] = []
    for w in weights:
        if w.ndim != 2:
            raise ValueError(f"weight must be 2D [O,K], got {tuple(w.shape)}")
        if width is None:
            width = int(w.shape[1])
        elif int(w.shape[1]) != width:
            raise ValueError("all shared-input weights must have the same K dimension")
        stats.append(w.detach().abs().to(torch.float32).amax(dim=0))
    return torch.stack(stats, dim=0).amax(dim=0)


def build_weight_aware_equalization_scale(
    amplitude: torch.Tensor,
    weight_stat: torch.Tensor,
    *,
    granularity: int,
    beta: float = 0.5,
    group_size: int = 64,
    min_scale: float = 0.5,
    max_scale: float = 2.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Static SmoothQuant-like comparator with a fixed, non-learned beta."""
    _require_1d("amplitude", amplitude)
    _require_1d("weight_stat", weight_stat)
    if amplitude.shape != weight_stat.shape:
        raise ValueError("amplitude and weight_stat must have identical shape")
    if min_scale <= 0 or max_scale <= 0 or min_scale > max_scale:
        raise ValueError("scale bounds must satisfy 0 < min_scale <= max_scale")

    a_unit = _group_unit_max(
        amplitude.to(torch.float32), granularity=granularity, group_size=group_size
    )
    w_unit = _group_unit_max(
        weight_stat.to(torch.float32), granularity=granularity, group_size=group_size
    )
    active = a_unit > float(eps)
    raw = torch.zeros_like(a_unit)
    safe_a = a_unit.clamp_min(float(eps))
    safe_w = w_unit.clamp_min(float(eps))
    raw[active] = float(beta) * (torch.log2(safe_a[active]) - torch.log2(safe_w[active]))

    z = torch.zeros_like(raw)
    for gi in range(raw.shape[0]):
        mask = active[gi]
        if not bool(mask.any()):
            continue
        z[gi, mask] = raw[gi, mask] - raw[gi, mask].mean()
    lo = float(torch.log2(torch.tensor(float(min_scale))).item())
    hi = float(torch.log2(torch.tensor(float(max_scale))).item())
    z = z.clamp(min=lo, max=hi)
    d_unit = torch.where(active, torch.exp2(z), torch.ones_like(z))
    return d_unit.reshape(-1).repeat_interleave(granularity).to(torch.float32)


def apply_linear_equivalent_scaling(
    x: torch.Tensor,
    w: torch.Tensor,
    d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if w.ndim != 2:
        raise ValueError("w must be 2D [O,K]")
    _require_1d("d", d)
    if x.shape[-1] != w.shape[-1] or d.numel() != w.shape[-1]:
        raise ValueError(
            f"K mismatch: x={x.shape[-1]}, w={w.shape[-1]}, d={d.numel()}"
        )
    _validate_positive_scale(d)
    dx = d.to(device=x.device, dtype=x.dtype)
    dw = d.to(device=w.device, dtype=w.dtype)
    return x / dx, w * dw.unsqueeze(0)


def _gqa_repeat(
    *,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
) -> int:
    if num_attention_heads <= 0 or num_key_value_heads <= 0 or head_dim <= 0:
        raise ValueError("head counts and head_dim must be positive")
    if num_attention_heads % num_key_value_heads != 0:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
    return num_attention_heads // num_key_value_heads


def collapse_gqa_o_amplitude(
    amplitude_o: torch.Tensor,
    *,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    reduction: Literal["max", "geomean"] = "max",
) -> torch.Tensor:
    """Collapse query-head O-input statistics to the unique KV-head scale domain."""
    _require_1d("amplitude_o", amplitude_o)
    repeat = _gqa_repeat(
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
    )
    expected = num_attention_heads * head_dim
    if amplitude_o.numel() != expected:
        raise ValueError(f"expected O amplitude width {expected}, got {amplitude_o.numel()}")
    x = amplitude_o.to(torch.float32).reshape(num_key_value_heads, repeat, head_dim)
    if bool((x < 0).any()) or not torch.isfinite(x).all():
        raise ValueError("amplitude_o must be finite and non-negative")
    if reduction == "max":
        return x.amax(dim=1)
    if reduction == "geomean":
        positive = x > 0
        # A coordinate that is zero for every query-head copy remains exactly zero.
        any_pos = positive.any(dim=1)
        safe = x.clamp_min(torch.finfo(torch.float32).tiny)
        out = torch.exp(torch.log(safe).mean(dim=1))
        return torch.where(any_pos, out, torch.zeros_like(out))
    raise ValueError(f"unsupported GQA reduction {reduction!r}")


def expand_gqa_o_scale(
    d_v_unique: torch.Tensor,
    *,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
) -> torch.Tensor:
    repeat = _gqa_repeat(
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
    )
    if d_v_unique.ndim == 1:
        if d_v_unique.numel() != num_key_value_heads * head_dim:
            raise ValueError("flat unique GQA scale has wrong length")
        unique = d_v_unique.reshape(num_key_value_heads, head_dim)
    elif d_v_unique.shape == (num_key_value_heads, head_dim):
        unique = d_v_unique
    else:
        raise ValueError(
            f"unique GQA scale must have shape {(num_key_value_heads, head_dim)} or flat equivalent"
        )
    _validate_positive_scale(unique)
    return unique.repeat_interleave(repeat, dim=0).reshape(-1)


def validate_gqa_tied_scale(
    d_o: torch.Tensor,
    *,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
) -> None:
    _require_1d("d_o", d_o)
    repeat = _gqa_repeat(
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
    )
    if d_o.numel() != num_attention_heads * head_dim:
        raise ValueError("GQA scale has wrong flattened width")
    _validate_positive_scale(d_o)
    heads = d_o.reshape(num_key_value_heads, repeat, head_dim)
    ref = heads[:, :1, :].expand_as(heads)
    if not torch.equal(heads, ref):
        raise ValueError("GQA deployable O scale must be tied across repeated query heads")


def fold_input_columns(weight: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Fold X/D into a Linear by multiplying its input columns by D."""
    if weight.ndim != 2:
        raise ValueError("weight must be 2D [O,K]")
    _require_1d("d", d)
    if d.numel() != weight.shape[1]:
        raise ValueError("input-column scale length must equal weight K")
    _validate_positive_scale(d)
    return weight * d.to(device=weight.device, dtype=weight.dtype).unsqueeze(0)


def fold_output_rows_inverse(
    weight: torch.Tensor,
    d: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Scale a Linear output by D^-1; bias follows the same row scaling."""
    if weight.ndim != 2:
        raise ValueError("weight must be 2D [O,K]")
    _require_1d("d", d)
    if d.numel() != weight.shape[0]:
        raise ValueError("output-row scale length must equal weight O")
    _validate_positive_scale(d)
    dd = d.to(device=weight.device, dtype=weight.dtype)
    w = weight / dd.unsqueeze(1)
    if bias is None:
        return w, None
    _require_1d("bias", bias)
    if bias.numel() != weight.shape[0]:
        raise ValueError("bias length must equal weight O")
    b = bias / d.to(device=bias.device, dtype=bias.dtype)
    return w, b


def fold_rmsnorm_weight(weight: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    _require_1d("weight", weight)
    _require_1d("d", d)
    if weight.shape != d.shape:
        raise ValueError("RMSNorm weight and scale must have identical shape")
    _validate_positive_scale(d)
    return weight / d.to(device=weight.device, dtype=weight.dtype)
