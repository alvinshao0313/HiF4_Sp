from __future__ import annotations

import torch


def validate_weight_divisible(
    weight: torch.Tensor,
    block_height: int,
    block_width: int,
) -> None:
    d_out, d_in = weight.shape
    if d_out % block_height != 0 or d_in % block_width != 0:
        raise ValueError(
            f"Weight shape {tuple(weight.shape)} is not divisible by "
            f"block_size={block_height}x{block_width}"
        )


def reduce_weight_gradient_to_blocks(
    weight: torch.Tensor,
    grad: torch.Tensor,
    block_height: int,
    block_width: int,
) -> torch.Tensor:
    """Reduce element-wise Taylor signal (W * G) into HxW block sums."""
    if weight.shape != grad.shape:
        raise ValueError(
            f"weight shape {tuple(weight.shape)} != grad shape {tuple(grad.shape)}"
        )
    validate_weight_divisible(weight, block_height, block_width)

    d_out, d_in = weight.shape
    element_signal = weight.float() * grad.float()
    return element_signal.reshape(
        d_out // block_height,
        block_height,
        d_in // block_width,
        block_width,
    ).sum(dim=(1, 3))


def reduce_weight_magnitude_to_blocks(
    weight: torch.Tensor,
    block_height: int,
    block_width: int,
) -> torch.Tensor:
    """Block Frobenius energy ||W_b||_F^2."""
    validate_weight_divisible(weight, block_height, block_width)
    d_out, d_in = weight.shape
    w2 = weight.float().square()
    return w2.reshape(
        d_out // block_height,
        block_height,
        d_in // block_width,
        block_width,
    ).sum(dim=(1, 3))


def reduce_weight_wanda_to_blocks(
    weight: torch.Tensor,
    input_rms: torch.Tensor,
    block_height: int,
    block_width: int,
) -> torch.Tensor:
    """Return block sums of abs(weight) weighted by input-channel RMS."""
    if weight.ndim != 2:
        raise ValueError(f"weight must be rank 2, got shape {tuple(weight.shape)}")
    if input_rms.ndim != 1:
        raise ValueError(
            f"input_rms must be rank 1, got shape {tuple(input_rms.shape)}"
        )
    if input_rms.numel() != weight.shape[1]:
        raise ValueError(
            f"input_rms length {input_rms.numel()} != weight d_in {weight.shape[1]}"
        )
    validate_weight_divisible(weight, block_height, block_width)

    d_out, d_in = weight.shape
    element_score = weight.float().abs() * input_rms.float().unsqueeze(0)
    return element_score.reshape(
        d_out // block_height,
        block_height,
        d_in // block_width,
        block_width,
    ).sum(dim=(1, 3))


def expand_block_mask(
    block_mask: torch.Tensor,
    block_height: int,
    block_width: int,
) -> torch.Tensor:
    """Expand [num_out_blocks, num_in_blocks] mask to element mask."""
    element_mask = block_mask.repeat_interleave(block_height, dim=0)
    element_mask = element_mask.repeat_interleave(block_width, dim=1)
    return element_mask


def block_grid_shape(
    weight_shape: tuple[int, int],
    block_height: int,
    block_width: int,
) -> tuple[int, int]:
    d_out, d_in = weight_shape
    if d_out % block_height != 0 or d_in % block_width != 0:
        raise ValueError(
            f"Weight shape {weight_shape} is not divisible by "
            f"block_size={block_height}x{block_width}"
        )
    return d_out // block_height, d_in // block_width


def active_block_indices(mask: torch.Tensor) -> list[tuple[int, int]]:
    """Return (out_block, in_block) indices where mask is True (kept)."""
    kept = mask.nonzero(as_tuple=False)
    return [(int(r.item()), int(c.item())) for r, c in kept]
