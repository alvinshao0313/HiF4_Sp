import pytest
import torch
from safetensors.torch import save_file

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.modelopt_moe_checkpoint import (
    MoELayerMasterState, MoEExpertMasterState, NativeNvfp4LinearMetadata,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_model_spec import Qwen3MoeModelSpec
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.evaluation.moe_materialize import _state_to_tensors
from Native_NVFP4_HiF4_Linear_Puncture.experiments.non_equivalent_reconstruction.transforms import transformed_state
from Native_NVFP4_HiF4_Linear_Puncture.experiments.non_equivalent_reconstruction.student import Student
from Native_NVFP4_HiF4_Linear_Puncture.experiments.kl_direction_reuse.losses import router_term
from Native_NVFP4_HiF4_Linear_Puncture.experiments.kl_direction_reuse.runtime import TP2Student, FrozenTP2Student, frozen_state


@pytest.fixture
def base():
    torch.manual_seed(17)
    spec = Qwen3MoeModelSpec("test", "Qwen3MoeForCausalLM", "qwen3_moe", 128, 48,
                             4, 2, 128, 2, 2, 128, 1, (), True)
    def rand(*shape):
        return torch.randn(*shape)*.03
    norm = torch.ones(128, dtype=torch.bfloat16)
    meta = NativeNvfp4LinearMetadata(torch.tensor(1.), torch.tensor(1.))
    attention = {"q_proj": rand(512, 128), "k_proj": rand(256, 128),
                 "v_proj": rand(256, 128), "o_proj": rand(128, 512)}
    return MoELayerMasterState(8, spec, norm.clone(), norm.clone(), norm.clone(), norm.clone(),
        rand(2, 128).bfloat16(), attention, {k: meta for k in attention},
        [MoEExpertMasterState(rand(128, 128), rand(128, 128), rand(128, 128), meta, meta, meta) for _ in range(2)])


def positions(n):
    return (torch.ones(1, n, 128, dtype=torch.bfloat16), torch.zeros(1, n, 128, dtype=torch.bfloat16))


def test_router_auxiliary_reaches_attention_and_not_experts(base):
    model = TP2Student(base, "group")
    x = torch.randn(1, 4, 128, dtype=torch.bfloat16)
    output = model(x, position_embeddings=positions(4))
    teacher = torch.randn_like(output.router_logits)
    router_term(output.router_logits, teacher, 4, top_k=2).backward()
    params = dict(model.learned.named_parameters())
    for name in ("input_norm", "moe_norm", "matrices.router", "matrices.q_proj", "matrices.o_proj"):
        assert params[name].grad is not None and params[name].grad.count_nonzero() > 0
    assert all(p.grad is None for name, p in params.items() if "expert_" in name)
    assert x.grad is None


@pytest.mark.parametrize("perturb", [False, True])
def test_transformed_export_and_frozen_suffix_use_same_tp2_math(base, tmp_path, perturb):
    model = TP2Student(base, "group")
    if perturb:
        with torch.no_grad():
            for p in model.learned.parameters():
                p.add_(torch.randn_like(p)*.002)
    tensors = _state_to_tensors(transformed_state(base, model.learned))
    save_file(tensors, tmp_path / "model-layer-00008-of-00048.safetensors")
    frozen = FrozenTP2Student(frozen_state(tmp_path, 8, "cpu", base.spec), model.eps)
    assert not list(frozen.parameters())
    x = torch.randn(1, 4, 128, dtype=torch.bfloat16)
    expected = model(x, position_embeddings=positions(4), use_ste=False)
    actual = frozen(x, position_embeddings=positions(4), use_ste=False)
    assert torch.equal(actual.output, expected.output)
    assert torch.equal(actual.router_logits, expected.router_logits)
    boundary = x.clone().requires_grad_()
    frozen(boundary, position_embeddings=positions(4)).output.float().square().sum().backward()
    assert boundary.grad is not None and boundary.grad.count_nonzero() > 0


def test_tp2_rounding_is_observable_and_causality_preserved(base):
    tp2, tp1 = TP2Student(base, "group"), Student(base, "group")
    x = torch.randn(1, 5, 512, dtype=torch.bfloat16)
    a = tp2.linear(x, "o_proj", base.attention["o_proj"], use_ste=False, head_dim=128)
    b = tp1.linear(x, "o_proj", base.attention["o_proj"], use_ste=False, head_dim=128)
    assert not torch.equal(a, b)
    hidden = torch.randn(1, 5, 128, dtype=torch.bfloat16)
    changed = hidden.clone()
    changed[:, 3:] *= -4
    before = tp2(hidden, position_embeddings=positions(5), use_ste=False).output
    after = tp2(changed, position_embeddings=positions(5), use_ste=False).output
    assert torch.equal(before[:, :3], after[:, :3])

