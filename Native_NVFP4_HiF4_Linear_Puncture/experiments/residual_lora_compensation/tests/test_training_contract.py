import pytest
import torch

from Native_NVFP4_HiF4_Linear_Puncture.experiments.non_equivalent_reconstruction.tests.test_layer import (
    initial, make_state, position_embeddings,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.non_equivalent_reconstruction.student import Student as GroupStudent
from Native_NVFP4_HiF4_Linear_Puncture.experiments.residual_lora_compensation.student import Student
from Native_NVFP4_HiF4_Linear_Puncture.experiments.residual_lora_compensation.losses import masked_router_loss
from Native_NVFP4_HiF4_Linear_Puncture.experiments.residual_lora_compensation.artifact import atomic_save, load


@pytest.mark.parametrize("mode", ["attention", "moe", "both"])
def test_zero_matches_group_and_router_boundary_survives_nonzero_adapters(mode, tmp_path):
    _, base = initial(make_state())
    group = GroupStudent(base, "group")
    student = Student(base, "group", lora_mode=mode)
    x = torch.randn(1, 3, 128, dtype=torch.bfloat16, requires_grad=True)
    positions = position_embeddings(3)
    with torch.no_grad():
        expected = group(x, position_embeddings=positions, use_ste=False).output
        actual = student(x, position_embeddings=positions, use_ste=False).output
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for branch in ("attention", "moe"):
            if mode in (branch, "both"):
                getattr(student.learned, f"{branch}_lora_B").normal_(std=0.01)
    out = student(x, position_embeddings=positions)
    auxiliary = student.router_aux_logits(out.pre_moe_norm)
    masked_router_loss(auxiliary, torch.randn_like(auxiliary), torch.ones(1, 3, dtype=torch.bool),
                       kind="top_mass", top_k=1).backward()
    assert {n for n, p in student.learned.named_parameters() if p.grad is not None} == {"moe_norm", "matrices.router"}
    assert x.grad is None
    atomic_save(student.learned.snapshot(), tmp_path / "parameters.pt")
    restored = Student(base, "group", lora_mode=mode)
    restored.learned.load_state_dict(load(tmp_path / "parameters.pt"), strict=True)
    with torch.no_grad():
        before = student(x, position_embeddings=positions, use_ste=False).output
        after = restored(x, position_embeddings=positions, use_ste=False).output
        torch.testing.assert_close(before, after, rtol=0, atol=0)
