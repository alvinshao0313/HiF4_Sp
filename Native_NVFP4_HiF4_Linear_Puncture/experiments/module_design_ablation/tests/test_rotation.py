"""Mathematical contracts; these checks do not establish model/runtime parity."""
import io

import torch

from ..rotation import CayleyRotation, apply_group_matrix
from ...diag_gradient.r64_transform import r64_matrix


def test_initialization_is_exact_r64_and_only_skew_coordinates_train():
    rotation = CayleyRotation(2)
    assert torch.equal(rotation(), r64_matrix().expand(2, -1, -1))
    assert list(dict(rotation.named_parameters())) == ["coefficients"]
    assert rotation.coefficients.numel() == 2 * 2016


def test_updates_preserve_orthogonality_and_tied_unquantized_linear():
    torch.manual_seed(42)
    rotation = CayleyRotation(2)
    target = torch.randn(2, 64, 64)
    optimizer = torch.optim.AdamW(rotation.parameters(), lr=1e-2, weight_decay=0)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        (rotation() - target).square().mean().backward()
        optimizer.step()
    q = rotation()
    assert not torch.equal(q, r64_matrix().expand_as(q))
    torch.testing.assert_close(q @ q.transpose(-1, -2), torch.eye(64).expand_as(q), rtol=0, atol=2e-6)
    x, weight = torch.randn(3, 5, 128), torch.randn(17, 128)
    actual = apply_group_matrix(x, q, head_dim=128) @ apply_group_matrix(weight, q).T
    torch.testing.assert_close(actual, x @ weight.T, rtol=2e-5, atol=3e-5)


def test_both_sides_contribute_to_tied_gradient():
    torch.manual_seed(43)
    rotation = CayleyRotation(2).double()
    with torch.no_grad():
        rotation.coefficients.normal_(std=0.03)
    x, weight = torch.randn(4, 128, dtype=torch.float64), torch.randn(7, 128, dtype=torch.float64)
    q = rotation()
    # The unquantized mapping is constant in Q. The two gradient contributions
    # must cancel; detaching either use of Q would break this identity.
    loss = (apply_group_matrix(x, q) @ apply_group_matrix(weight, q).T).square().mean()
    gradient, = torch.autograd.grad(loss, rotation.coefficients)
    assert gradient.abs().max().item() < 1e-10


def test_save_reload_preserves_learned_rotation():
    torch.manual_seed(44)
    original = CayleyRotation(2)
    with torch.no_grad():
        original.coefficients.normal_(std=0.1)
    stream = io.BytesIO()
    torch.save(original.state_dict(), stream)
    stream.seek(0)
    restored = CayleyRotation(2)
    restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    assert torch.equal(original.coefficients, restored.coefficients)
    assert torch.equal(original(), restored())
