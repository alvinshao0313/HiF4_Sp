"""Channel statistics, proxy distances, C4/C64 costs, and output NRMSE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .config import SearchConfig
from .hif4_reference import hif4_fake_quantize, s1p2_oracle_quantize_rows

_NEAR_ZERO_FRAC = 0.01
_RANGE_RATIO = 7.0  # 1.75 / 0.25


@dataclass(frozen=True)
class ChannelStatistics:
    # Error-weight convention (diagonal approximation of the two independent
    # error sources): activation-quantization error is weighted by
    # ``weight_energy``; weight-quantization error by ``activation_energy``.
    activation_energy: torch.Tensor  # [d_ff] E[A_c^2]
    weight_energy: torch.Tensor  # [d_ff] sum_o W[o,c]^2 (column energy)
    # Diagnostic / neighbor-recall feature only; NOT a valid joint weight for
    # both error directions.
    output_sensitivity: torch.Tensor  # [d_ff] sqrt(E[A^2]) * ||W_d[:,c]||
    features: torch.Tensor  # [d_ff, 12]
    primary_scale: torch.Tensor  # [d_ff]
    neighbors: torch.Tensor  # [d_ff, neighbor_k] long


def sample_weight_rows(weight: torch.Tensor, rows: int, seed: int) -> torch.Tensor:
    """Deterministic uniform sample of down weight rows; return CPU FP32 [rows, d_ff]."""
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2-D [d_model, d_ff], got {tuple(weight.shape)}")
    d_model, d_ff = weight.shape
    n = min(rows, d_model)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    # Evenly spaced indices with deterministic offset from seed.
    if n == d_model:
        idx = torch.arange(d_model, dtype=torch.long)
    else:
        step = d_model / n
        offset = int(torch.randint(0, max(d_model, 1), (1,), generator=gen).item() % max(int(step), 1))
        idx = (torch.arange(n, dtype=torch.float64) * step + offset).long() % d_model
        idx = torch.unique(idx, sorted=True)
        # If unique shortened, fill remaining deterministically.
        if idx.numel() < n:
            extra = torch.randperm(d_model, generator=gen)
            for e in extra.tolist():
                if e not in set(idx.tolist()):
                    idx = torch.cat([idx, torch.tensor([e], dtype=torch.long)])
                if idx.numel() >= n:
                    break
            idx = idx[:n]
    rows_t = weight.detach().index_select(0, idx.to(device=weight.device)).to(
        device="cpu", dtype=torch.float32
    )
    return rows_t.contiguous()


def _safe_log2(x: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.log2(torch.clamp(x, min=eps))


def _channel_abs_features(x: torch.Tensor, eps: float) -> tuple[torch.Tensor, ...]:
    """x: [rows, d_ff] FP32 -> per-channel feature components (each [d_ff])."""
    abs_x = x.abs()
    rms = torch.sqrt((x * x).mean(dim=0).clamp_min(eps))
    # One sort for all quantiles (faster than three torch.quantile calls).
    n = abs_x.shape[0]
    sorted_abs, _ = torch.sort(abs_x, dim=0)

    def _q(p: float) -> torch.Tensor:
        idx = min(n - 1, max(0, int(round(p * (n - 1)))))
        return sorted_abs[idx]

    q50 = _q(0.50)
    q90 = _q(0.90)
    q99 = _q(0.99)
    log_rms = _safe_log2(rms, eps)
    log_q50 = _safe_log2(q50, eps)
    log_q90 = _safe_log2(q90, eps)
    log_q99 = _safe_log2(q99, eps)
    log_spread = _safe_log2(q99 / q50.clamp_min(eps), eps)
    near_zero = (abs_x < (_NEAR_ZERO_FRAC * q99.unsqueeze(0))).to(torch.float32).mean(dim=0)
    near_zero = torch.where(q99 <= eps, torch.ones_like(near_zero), near_zero)
    return log_rms, log_q50, log_q90, log_q99, log_spread, near_zero, q99


def build_channel_statistics(
    activation: torch.Tensor,
    weight: torch.Tensor,
    config: SearchConfig,
) -> ChannelStatistics:
    """Compute energies, 12-D features, primary scale, and local neighbor table."""
    act = activation.detach().to(device="cpu", dtype=torch.float32)
    w = weight.detach().to(device="cpu", dtype=torch.float32)
    if act.ndim != 2 or w.ndim != 2:
        raise ValueError("activation and weight must be 2-D")
    if act.shape[1] != w.shape[1]:
        raise ValueError(
            f"d_ff mismatch: act {act.shape[1]} vs weight {w.shape[1]}"
        )
    d_ff = act.shape[1]
    if d_ff % 64 != 0:
        raise ValueError(f"d_ff must be divisible by 64, got {d_ff}")

    eps = config.eps
    e_a = (act * act).mean(dim=0)
    e_w = (w * w).mean(dim=0)

    a_feats = _channel_abs_features(act, eps)
    w_feats = _channel_abs_features(w, eps)
    # 12-D: 6 act + 6 weight (excluding raw q99 helpers)
    features = torch.stack(
        [
            a_feats[0],
            a_feats[1],
            a_feats[2],
            a_feats[3],
            a_feats[4],
            a_feats[5],
            w_feats[0],
            w_feats[1],
            w_feats[2],
            w_feats[3],
            w_feats[4],
            w_feats[5],
        ],
        dim=1,
    )  # [d_ff, 12]

    # Column-wise standardize.
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    features_std = (features - mean) / std

    primary_scale = 0.5 * (a_feats[3] + w_feats[3])  # 0.5 * (act_log_q99 + weight_log_q99)

    # Per-channel output sensitivity: diagonal approx of E[Y_c^2].
    # sqrt(E[A_c^2]) * ||W_d[:,c]||
    w_col_norm = w.norm(dim=0)
    output_sensitivity = torch.sqrt(e_a.clamp_min(eps)) * w_col_norm

    neighbors = _build_local_neighbors(
        features_std=features_std,
        primary_scale=primary_scale,
        candidate_window=config.candidate_window,
        neighbor_k=config.neighbor_k,
    )
    return ChannelStatistics(
        activation_energy=e_a.contiguous(),
        weight_energy=e_w.contiguous(),
        output_sensitivity=output_sensitivity.contiguous(),
        features=features_std.contiguous(),
        primary_scale=primary_scale.contiguous(),
        neighbors=neighbors.contiguous(),
    )


def _build_local_neighbors(
    features_std: torch.Tensor,
    primary_scale: torch.Tensor,
    candidate_window: int,
    neighbor_k: int,
) -> torch.Tensor:
    """Local window neighbor recall; never builds [d_ff, d_ff] distance matrix."""
    d_ff = features_std.shape[0]
    order = torch.argsort(primary_scale, stable=True)
    feats_sorted = features_std[order]  # [d_ff, 12]
    half = min(candidate_window, max(d_ff - 1, 0))
    win = 2 * half + 1
    if win > d_ff:
        # Tiny d_ff: fall back to all-others distance within full set.
        half = d_ff - 1
        win = 2 * half + 1

    # Pad along channel axis then unfold fixed windows. Memory O(d_ff * win * 12).
    padded = torch.nn.functional.pad(feats_sorted.T, (half, half))  # [12, d_ff+2half]
    windows = padded.unfold(1, win, 1)  # [12, d_ff, win]
    center = feats_sorted.T.unsqueeze(-1)  # [12, d_ff, 1]
    dist2 = ((windows - center) ** 2).sum(dim=0)  # [d_ff, win]

    # Invalidate center and out-of-bounds pad slots.
    dist2[:, half] = float("inf")
    pos = torch.arange(d_ff).unsqueeze(1)
    win_pos = pos + torch.arange(-half, half + 1).unsqueeze(0)  # [d_ff, win]
    invalid = (win_pos < 0) | (win_pos >= d_ff)
    dist2 = dist2.masked_fill(invalid, float("inf"))

    # Stable tie-break: add tiny sorted-position key.
    key = dist2.to(torch.float64) + win_pos.to(torch.float64) * 1e-12
    # Replace inf keys so topk still works; use large finite.
    key = torch.where(torch.isfinite(dist2), key, torch.full_like(key, 1e30))
    k = min(neighbor_k, max(d_ff - 1, 1))
    top = torch.topk(key, k=k, largest=False).indices  # [d_ff, k]
    # Map window-local indices to original channel ids via order.
    gather_pos = torch.gather(win_pos, 1, top).clamp(0, d_ff - 1)
    neigh_sorted = order[gather_pos]  # [d_ff, k] in sorted-row order
    # Scatter back to original channel rows.
    neighbors = torch.empty(d_ff, k, dtype=torch.long)
    neighbors[order] = neigh_sorted
    if k < neighbor_k:
        # Pad (tiny models only)
        pad_n = neighbor_k - k
        extra = torch.arange(d_ff).unsqueeze(0).expand(d_ff, -1)
        # Not needed for real models; fill with modular others.
        fill = torch.zeros(d_ff, pad_n, dtype=torch.long)
        neighbors = torch.cat([neighbors, fill], dim=1)
    return neighbors


def pair_compatibility_cost(
    i: int,
    j: int,
    activation: torch.Tensor,
    weight_rows: torch.Tensor,
    stats: ChannelStatistics,
    eps: float,
) -> float:
    """Proxy compatibility of two channels on activation and weight row trajectories.

    Cross energy weighting: activation-quantization error matters in proportion
    to the down-weight column energy (``weight_energy``); weight-quantization
    error matters in proportion to the activation energy (``activation_energy``).
    """
    if i == j:
        return 0.0
    act = activation.to(torch.float32)
    w = weight_rows.to(torch.float32)
    act_importance = stats.weight_energy
    w_importance = stats.activation_energy

    def _traj_cost(x: torch.Tensor, importance: torch.Tensor) -> float:
        xi = x[:, i]
        xj = x[:, j]
        large = torch.maximum(xi.abs(), xj.abs())
        small = torch.minimum(xi.abs(), xj.abs())
        log_ratio = torch.log2((large + eps) / (small + 0.01 * large + eps))
        row_weight = importance[i] * (xi * xi) + importance[j] * (xj * xj)
        denom = row_weight.sum()
        if float(denom.item()) <= eps:
            return 0.0
        return float(((log_ratio * log_ratio) * row_weight).sum().item() / denom.item())

    ca = _traj_cost(act, act_importance)
    cw = _traj_cost(w, w_importance)
    return 0.5 * ca + 0.5 * cw


def precompute_neighbor_pair_costs(
    activation: torch.Tensor,
    weight_rows: torch.Tensor,
    stats: ChannelStatistics,
    eps: float,
) -> dict[tuple[int, int], float]:
    """Vectorized pair-cost cache for all (channel, neighbor) edges."""
    act = activation.detach().to(dtype=torch.float32, device="cpu")
    w = weight_rows.detach().to(dtype=torch.float32, device="cpu")
    act_importance = stats.weight_energy.to(dtype=torch.float32, device="cpu")
    w_importance = stats.activation_energy.to(dtype=torch.float32, device="cpu")
    neigh = stats.neighbors
    d_ff, k = neigh.shape

    # Unique undirected edges
    i_idx = torch.arange(d_ff).unsqueeze(1).expand(-1, k).reshape(-1)
    j_idx = neigh.reshape(-1)
    a = torch.minimum(i_idx, j_idx)
    b = torch.maximum(i_idx, j_idx)
    mask = a != b
    a, b = a[mask], b[mask]
    edges = torch.stack([a, b], dim=1)
    edges = torch.unique(edges, dim=0)
    if edges.numel() == 0:
        return {}

    def _batch_traj(x: torch.Tensor, importance: torch.Tensor) -> torch.Tensor:
        # x: [rows, d], edges: [E, 2]
        xi = x[:, edges[:, 0]]  # [rows, E]
        xj = x[:, edges[:, 1]]
        large = torch.maximum(xi.abs(), xj.abs())
        small = torch.minimum(xi.abs(), xj.abs())
        log_ratio = torch.log2((large + eps) / (small + 0.01 * large + eps))
        wi = importance[edges[:, 0]]
        wj = importance[edges[:, 1]]
        row_w = wi * (xi * xi) + wj * (xj * xj)  # [rows, E]
        num = (log_ratio * log_ratio * row_w).sum(dim=0)
        den = row_w.sum(dim=0)
        out = torch.zeros_like(num)
        valid = den > eps
        out[valid] = num[valid] / den[valid]
        return out

    costs = 0.5 * _batch_traj(act, act_importance) + 0.5 * _batch_traj(w, w_importance)
    cache: dict[tuple[int, int], float] = {}
    for e, c in zip(edges.tolist(), costs.tolist()):
        cache[(int(e[0]), int(e[1]))] = float(c)
    return cache


def _weighted_nrmse(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    channel_importance: torch.Tensor,
    eps: float,
) -> float:
    """Energy-weighted normalized MSE over last dim channels.

    x, x_hat: [rows, G]; channel_importance: [G]
    """
    diff = x - x_hat
    w = channel_importance.unsqueeze(0)
    num = (w * diff * diff).sum()
    den = (w * x * x).sum()
    if float(den.item()) <= eps:
        return 0.0
    return float((num / (den + eps)).item())


def _range7_penalty(
    x: torch.Tensor,
    channel_importance: torch.Tensor,
    eps: float,
) -> float:
    """Fraction of weighted energy sitting below M/7 on each row, averaged."""
    abs_x = x.abs()
    m = abs_x.amax(dim=-1, keepdim=True)
    # Elements with 0 < |x| < M/7
    small = (abs_x > eps) & (abs_x < (m / _RANGE_RATIO))
    w = channel_importance.unsqueeze(0)
    energy = w * (x * x)
    row_num = (energy * small.to(energy.dtype)).sum(dim=-1)
    row_den = energy.sum(dim=-1)
    valid = row_den > eps
    if not bool(valid.any()):
        return 0.0
    ratios = torch.zeros_like(row_den)
    ratios[valid] = row_num[valid] / (row_den[valid] + eps)
    return float(ratios[valid].mean().item())


def c4_cost(
    channels: Sequence[int],
    activation: torch.Tensor,
    weight_rows: torch.Tensor,
    stats: ChannelStatistics,
    config: SearchConfig,
) -> float:
    """S1P2 oracle cost for a group of 1..4 channels.

    At the 4-tuple granularity this matches the HiF4 format (2-bit mantissa,
    dynamic range 7, row-local scale); the shared lv1/lv2 scales depend on the
    enclosing 64-group and cancel when comparing candidate tuples.
    """
    ch = list(channels)
    if not (1 <= len(ch) <= 4):
        raise ValueError(f"c4_cost expects 1..4 channels, got {len(ch)}")
    act = activation.to(torch.float32)[:, ch]
    w = weight_rows.to(torch.float32)[:, ch]
    act_importance = stats.weight_energy[ch]
    w_importance = stats.activation_energy[ch]
    eps = config.eps

    act_q = s1p2_oracle_quantize_rows(act, eps=eps)
    w_q = s1p2_oracle_quantize_rows(w, eps=eps)
    e_a_err = _weighted_nrmse(act, act_q, act_importance, eps)
    e_w_err = _weighted_nrmse(w, w_q, w_importance, eps)
    r7 = 0.5 * (
        _range7_penalty(act, act_importance, eps)
        + _range7_penalty(w, w_importance, eps)
    )
    return (
        config.activation_loss_weight * e_a_err
        + config.weight_loss_weight * e_w_err
        + config.range_loss_weight * r7
    )


def _cost_device(*tensors: torch.Tensor) -> torch.device:
    for t in tensors:
        if t.is_cuda:
            return t.device
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def c64_cost(
    channels: Sequence[int],
    activation: torch.Tensor,
    weight_rows: torch.Tensor,
    stats: ChannelStatistics,
    config: SearchConfig,
) -> float:
    """Real HiF4 cost for exactly 64 channels."""
    ch = list(channels)
    if len(ch) != 64:
        raise ValueError(f"c64_cost expects exactly 64 channels, got {len(ch)}")
    # Gather on the source device first; only the 64-wide slices go to CUDA.
    ch_src = torch.as_tensor(ch, dtype=torch.long, device=activation.device)
    act = activation.to(dtype=torch.float32).index_select(1, ch_src)
    ch_w = ch_src.to(device=weight_rows.device)
    w = weight_rows.to(dtype=torch.float32).index_select(1, ch_w)
    act_importance = stats.weight_energy.to(dtype=torch.float32)[ch]
    w_importance = stats.activation_energy.to(dtype=torch.float32)[ch]
    q_device = _cost_device(act, w)
    act = act.to(device=q_device, dtype=torch.float32)
    w = w.to(device=q_device, dtype=torch.float32)
    act_importance = act_importance.to(device=q_device, dtype=torch.float32)
    w_importance = w_importance.to(device=q_device, dtype=torch.float32)
    eps = config.eps

    act_q = hif4_fake_quantize(act)
    w_q = hif4_fake_quantize(w)
    e_a_err = _weighted_nrmse(act, act_q, act_importance, eps)
    e_w_err = _weighted_nrmse(w, w_q, w_importance, eps)
    return 0.5 * e_a_err + 0.5 * e_w_err


def _block_weighted_nrmse(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    channel_importance: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Per-block energy-weighted NRMSE.

    x, x_hat: [rows, n_blocks, 64]; channel_importance: [n_blocks, 64] -> [n_blocks]
    """
    w = channel_importance.unsqueeze(0)
    num = (w * (x - x_hat) * (x - x_hat)).sum(dim=(0, 2))
    den = (w * x * x).sum(dim=(0, 2))
    out = torch.zeros_like(num)
    valid = den > eps
    out[valid] = num[valid] / (den[valid] + eps)
    return out


def full_layout_hif4_loss(
    permutation: torch.Tensor,
    activation: torch.Tensor,
    weight_rows: torch.Tensor,
    stats: ChannelStatistics,
    config: SearchConfig,
) -> tuple[float, list[float]]:
    """Average C64 loss over consecutive 64-channel blocks of the permutation."""
    device = _cost_device(activation, weight_rows)
    perm = permutation.detach().to(device=device, dtype=torch.long)
    d_ff = int(perm.numel())
    if d_ff % 64 != 0:
        raise ValueError(f"permutation length must be divisible by 64, got {d_ff}")
    n_blocks = d_ff // 64
    act = activation.to(device=device, dtype=torch.float32).index_select(1, perm)
    w = weight_rows.to(device=device, dtype=torch.float32).index_select(1, perm)
    act_importance = (
        stats.weight_energy.to(device=device, dtype=torch.float32).index_select(0, perm)
    )
    w_importance = (
        stats.activation_energy.to(device=device, dtype=torch.float32).index_select(0, perm)
    )

    act_q = hif4_fake_quantize(act)
    w_q = hif4_fake_quantize(w)
    e_a_err = _block_weighted_nrmse(
        act.view(act.shape[0], n_blocks, 64),
        act_q.view(act.shape[0], n_blocks, 64),
        act_importance.view(n_blocks, 64),
        config.eps,
    )
    e_w_err = _block_weighted_nrmse(
        w.view(w.shape[0], n_blocks, 64),
        w_q.view(w.shape[0], n_blocks, 64),
        w_importance.view(n_blocks, 64),
        config.eps,
    )
    block = 0.5 * e_a_err + 0.5 * e_w_err
    block_losses = [float(v) for v in block.detach().cpu().tolist()]
    mean_loss = float(block.mean().item()) if block_losses else 0.0
    return mean_loss, block_losses


def batched_full_layout_hif4_loss(
    permutations: torch.Tensor,
    activation: torch.Tensor,
    weight_rows: torch.Tensor,
    stats: ChannelStatistics,
    config: SearchConfig,
    batch_size: int = 8,
) -> torch.Tensor:
    """Batched :func:`full_layout_hif4_loss` over ``permutations`` [K, d_ff].

    Returns FP64 [K] mean block losses. ``batch_size`` bounds the number of
    candidates expanded per quant call to cap memory.
    """
    device = _cost_device(activation, weight_rows)
    perms = permutations.detach().to(device=device, dtype=torch.long)
    if perms.ndim != 2 or perms.shape[1] % 64 != 0:
        raise ValueError(
            f"permutations must be [K, d_ff] with d_ff divisible by 64, got {tuple(perms.shape)}"
        )
    K, d_ff = perms.shape
    n_blocks = d_ff // 64
    act = activation.to(device=device, dtype=torch.float32)
    w = weight_rows.to(device=device, dtype=torch.float32)
    act_imp = stats.weight_energy.to(device=device, dtype=torch.float32)
    w_imp = stats.activation_energy.to(device=device, dtype=torch.float32)
    out = torch.empty(K, dtype=torch.float64)
    for k0 in range(0, K, batch_size):
        pb = perms[k0 : k0 + batch_size]
        B = int(pb.shape[0])
        a = act.index_select(1, pb.reshape(-1)).view(act.shape[0], B * n_blocks, 64)
        wb = w.index_select(1, pb.reshape(-1)).view(w.shape[0], B * n_blocks, 64)
        ai = act_imp[pb].reshape(B * n_blocks, 64)
        wi = w_imp[pb].reshape(B * n_blocks, 64)
        a_q = hif4_fake_quantize(a)
        w_q = hif4_fake_quantize(wb)
        e_a = _block_weighted_nrmse(a, a_q, ai, config.eps)
        e_w = _block_weighted_nrmse(wb, w_q, wi, config.eps)
        losses = (0.5 * e_a + 0.5 * e_w).view(B, n_blocks).mean(dim=1)
        out[k0 : k0 + B] = losses.detach().to(device="cpu", dtype=torch.float64)
    return out


@torch.no_grad()
def build_quantized_swiglu_activation(
    mlp_input: torch.Tensor,
    up_weight: torch.Tensor,
    gate_weight: torch.Tensor,
) -> torch.Tensor:
    """Down input under real W4A4: SiLU(Xq @ Wg_q^T) * (Xq @ Wu_q^T).

    All tensors are fake-quantized along last dim (in_features) with real HiF4.
    X: [rows, d_model]; Wu/Wg: [d_ff, d_model]; returns [rows, d_ff] FP32.
    """
    x = mlp_input.detach().to(dtype=torch.float32)
    wu = up_weight.detach().to(dtype=torch.float32)
    wg = gate_weight.detach().to(dtype=torch.float32)
    device = _cost_device(x, wu, wg)
    x = x.to(device=device)
    wu = wu.to(device=device)
    wg = wg.to(device=device)
    x_q = hif4_fake_quantize(x)
    wu_q = hif4_fake_quantize(wu)
    wg_q = hif4_fake_quantize(wg)
    a = torch.nn.functional.silu(x_q @ wg_q.transpose(0, 1)) * (x_q @ wu_q.transpose(0, 1))
    return a.detach()


@dataclass(frozen=True)
class DeploymentMetrics:
    """Deployment-consistent error decomposition for one candidate permutation."""

    bf16_reorder_drift: float  # ||Y_bf16(P) - Y_bf16(I)||_F / ||Y_bf16(I)||_F
    quantization_residual_nrmse: float  # ||Y_w4a4(P) - Y_bf16(P)||_F / ||Y_bf16(P)||_F
    total_nrmse: float  # ||Y_w4a4(P) - Y_bf16(I)||_F / ||Y_bf16(I)||_F


def _fro_nrmse_fp32(a: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> float:
    num = torch.linalg.norm(a.float() - ref.float(), ord="fro")
    den = torch.linalg.norm(ref.float(), ord="fro")
    return float((num / (den + eps)).item())


class DeploymentMLPContext:
    """Evaluate candidate permutations on the deployment (BF16 RTN) path.

    BF16 reference: ``F.linear`` with the model's original dtype weights, no
    FP32 upcast. W4A4 path: real HiF4 fake quant on X, permuted Wu/Wg, the
    intermediate activation and permuted Wd; quantized tensors keep the model
    dtype (BF16) when entering ``F.linear``, matching the RTN checkpoint +
    QLinear2 fake-quant input path. All norms are computed in FP32.
    """

    @torch.no_grad()
    def __init__(
        self,
        mlp_input: torch.Tensor,
        up_weight: torch.Tensor,
        gate_weight: torch.Tensor,
        down_weight: torch.Tensor,
        device: torch.device,
    ) -> None:
        x = mlp_input.detach().to(device=device)
        wu = up_weight.detach().to(device=device)
        wg = gate_weight.detach().to(device=device)
        wd = down_weight.detach().to(device=device)
        if x.ndim != 2 or wu.ndim != 2 or wg.ndim != 2 or wd.ndim != 2:
            raise ValueError("X/Wu/Wg/Wd must be 2-D")
        if wu.shape != wg.shape:
            raise ValueError(f"up/gate shape mismatch: {wu.shape} vs {wg.shape}")
        d_ff, d_model = wu.shape
        if x.shape[1] != d_model or wd.shape != (d_model, d_ff):
            raise ValueError(
                f"shape mismatch: X {x.shape}, Wu {wu.shape}, Wd {wd.shape}"
            )
        self.x = x
        self.wu = wu
        self.wg = wg
        self.wd = wd
        self.device = device
        self.y_bf16_identity = self._bf16_forward(self.wu, self.wg, self.wd)

    @torch.no_grad()
    def _bf16_forward(
        self, wu: torch.Tensor, wg: torch.Tensor, wd: torch.Tensor
    ) -> torch.Tensor:
        a = torch.nn.functional.silu(torch.nn.functional.linear(self.x, wg)) * (
            torch.nn.functional.linear(self.x, wu)
        )
        return torch.nn.functional.linear(a, wd)

    @torch.no_grad()
    def _w4a4_forward(
        self, wu: torch.Tensor, wg: torch.Tensor, wd: torch.Tensor
    ) -> torch.Tensor:
        x_q = hif4_fake_quantize(self.x)
        wu_q = hif4_fake_quantize(wu)
        wg_q = hif4_fake_quantize(wg)
        a = torch.nn.functional.silu(torch.nn.functional.linear(x_q, wg_q)) * (
            torch.nn.functional.linear(x_q, wu_q)
        )
        a_q = hif4_fake_quantize(a)
        wd_q = hif4_fake_quantize(wd)
        return torch.nn.functional.linear(a_q, wd_q)

    @torch.no_grad()
    def _debug_forward(
        self, permutation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (Y_bf16(P), Y_w4a4(P)); test/diagnostic use only."""
        perm = permutation.detach().to(device=self.device, dtype=torch.long)
        wu_p = self.wu.index_select(0, perm)
        wg_p = self.wg.index_select(0, perm)
        wd_p = self.wd.index_select(1, perm)
        return self._bf16_forward(wu_p, wg_p, wd_p), self._w4a4_forward(wu_p, wg_p, wd_p)

    @torch.no_grad()
    def evaluate(self, permutation: torch.Tensor) -> DeploymentMetrics:
        perm = permutation.detach().to(device=self.device, dtype=torch.long)
        if perm.numel() != self.wd.shape[1]:
            raise ValueError("permutation length mismatch")
        wu_p = self.wu.index_select(0, perm)
        wg_p = self.wg.index_select(0, perm)
        wd_p = self.wd.index_select(1, perm)
        y_bf16_p = self._bf16_forward(wu_p, wg_p, wd_p)
        y_w4a4_p = self._w4a4_forward(wu_p, wg_p, wd_p)
        return DeploymentMetrics(
            bf16_reorder_drift=_fro_nrmse_fp32(y_bf16_p, self.y_bf16_identity),
            quantization_residual_nrmse=_fro_nrmse_fp32(y_w4a4_p, y_bf16_p),
            total_nrmse=_fro_nrmse_fp32(y_w4a4_p, self.y_bf16_identity),
        )


class DeploymentDownContext:
    """Deployment-consistent evaluation for the down-only path (tests/audit).

    Same metric definitions as :class:`DeploymentMLPContext`, restricted to
    the down projection: BF16-path reference vs real HiF4 W4A4 fake quant.
    """

    @torch.no_grad()
    def __init__(
        self,
        activation: torch.Tensor,
        down_weight: torch.Tensor,
        device: torch.device,
    ) -> None:
        a = activation.detach().to(device=device)
        w = down_weight.detach().to(device=device)
        if a.ndim != 2 or w.ndim != 2 or a.shape[1] != w.shape[1]:
            raise ValueError(
                f"activation/down_weight shape mismatch: {a.shape} vs {w.shape}"
            )
        self.a = a
        self.w = w
        self.device = device
        self.y_bf16_identity = torch.nn.functional.linear(self.a, self.w)

    @torch.no_grad()
    def evaluate(self, permutation: torch.Tensor) -> DeploymentMetrics:
        perm = permutation.detach().to(device=self.device, dtype=torch.long)
        if perm.numel() != self.w.shape[1]:
            raise ValueError("permutation length mismatch")
        a_p = self.a.index_select(1, perm)
        w_p = self.w.index_select(1, perm)
        y_bf16_p = torch.nn.functional.linear(a_p, w_p)
        a_q = hif4_fake_quantize(a_p)
        w_q = hif4_fake_quantize(w_p)
        y_w4a4_p = torch.nn.functional.linear(a_q, w_q)
        return DeploymentMetrics(
            bf16_reorder_drift=_fro_nrmse_fp32(y_bf16_p, self.y_bf16_identity),
            quantization_residual_nrmse=_fro_nrmse_fp32(y_w4a4_p, y_bf16_p),
            total_nrmse=_fro_nrmse_fp32(y_w4a4_p, self.y_bf16_identity),
        )


class MLPW4A4Context:
    """Precomputed per-layer quantities so each perm evaluation is just
    index_select + one quant + one matmul (no repeated X/Wu/Wg quant)."""

    @torch.no_grad()
    def __init__(
        self,
        mlp_input: torch.Tensor,
        up_weight: torch.Tensor,
        gate_weight: torch.Tensor,
        down_weight: torch.Tensor,
    ) -> None:
        x = mlp_input.detach().to(dtype=torch.float32)
        wu = up_weight.detach().to(dtype=torch.float32)
        wg = gate_weight.detach().to(dtype=torch.float32)
        wd = down_weight.detach().to(dtype=torch.float32)
        device = _cost_device(x, wu, wg, wd)
        x = x.to(device=device)
        wu = wu.to(device=device)
        wg = wg.to(device=device)
        wd = wd.to(device=device)
        if x.shape[-1] != wu.shape[1] or wu.shape != wg.shape or wd.shape[0] != x.shape[-1]:
            raise ValueError("shape mismatch among X/Wu/Wg/Wd")
        self.y_fp = (
            torch.nn.functional.silu(x @ wg.transpose(0, 1)) * (x @ wu.transpose(0, 1))
        ) @ wd.transpose(0, 1)
        self.y_fp_norm = torch.linalg.norm(self.y_fp, ord="fro")
        self.a_qa = build_quantized_swiglu_activation(x, wu, wg)
        self.wd = wd
        self.device = device

    @torch.no_grad()
    def output_nrmse(self, permutation: torch.Tensor) -> float:
        eps = 1e-8
        perm = permutation.detach().to(device=self.device, dtype=torch.long)
        a = self.a_qa.index_select(1, perm)
        wd_p = self.wd.index_select(1, perm)
        a_q = hif4_fake_quantize(a)
        wd_q = hif4_fake_quantize(wd_p)
        y_q = a_q @ wd_q.transpose(0, 1)
        num = torch.linalg.norm(y_q - self.y_fp, ord="fro")
        return float((num / (self.y_fp_norm + eps)).item())


@torch.no_grad()
def mlp_w4a4_output_nrmse(
    mlp_input: torch.Tensor,
    up_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    down_weight: torch.Tensor,
    permutation: torch.Tensor,
) -> float:
    """NRMSE of full SwiGLU MLP W4A4 fake-quant output vs FP output under perm.

    perm[new_position] = old_channel_index, applied to the intermediate dim.
    """
    ctx = MLPW4A4Context(mlp_input, up_weight, gate_weight, down_weight)
    if permutation.numel() != ctx.wd.shape[1]:
        raise ValueError("permutation length mismatch")
    return ctx.output_nrmse(permutation)


@torch.no_grad()
def down_output_nrmse(
    activation: torch.Tensor,
    down_weight: torch.Tensor,
    permutation: torch.Tensor,
) -> float:
    """NRMSE of W4A4 down projection vs FP output under the given permutation."""
    eps = 1e-8
    device = _cost_device(activation, down_weight)
    perm = permutation.detach().to(device=device, dtype=torch.long)
    a = activation.detach().to(device=device, dtype=torch.float32)
    w = down_weight.detach().to(device=device, dtype=torch.float32)
    if a.shape[1] != w.shape[1] or a.shape[1] != perm.numel():
        raise ValueError("activation/weight/perm d_ff mismatch")

    y_fp = a @ w.transpose(0, 1)
    a_p = a.index_select(1, perm)
    w_p = w.index_select(1, perm)
    a_q = hif4_fake_quantize(a_p)
    w_q = hif4_fake_quantize(w_p)
    y_q = a_q @ w_q.transpose(0, 1)
    num = torch.linalg.norm(y_q - y_fp, ord="fro")
    den = torch.linalg.norm(y_fp, ord="fro")
    if float(den.item()) <= eps:
        return 0.0
    return float((num / (den + eps)).item())
