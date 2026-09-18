from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file, load_file

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.modelopt_moe_checkpoint import (
    MoELayerMasterState, MoEExpertMasterState, NativeNvfp4LinearMetadata,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_model_spec import Qwen3MoeModelSpec
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_fold import fold_fusable_moe_layer_state
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_semantic_hif4 import build_moe_diag_state
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_transforms import apply_r64_no_cross_head
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.evaluation.moe_materialize import _state_to_tensors
from Native_NVFP4_HiF4_Linear_Puncture.experiments.non_equivalent_reconstruction.student import Student, quantize, weighted_linear
from Native_NVFP4_HiF4_Linear_Puncture.experiments.non_equivalent_reconstruction.transforms import (
    fold_initial_diag, transformed_state, projection_weights,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.non_equivalent_reconstruction.losses import masked_router_loss
from Native_NVFP4_HiF4_Linear_Puncture.experiments.non_equivalent_reconstruction.artifact import atomic_save, load


def make_state(device="cpu"):
    torch.manual_seed(17)
    # Keep the real 32Q/4KV head layout required by the existing DIAG fold.
    spec = Qwen3MoeModelSpec("test", "Qwen3MoeForCausalLM", "qwen3_moe", 128, 48,
                             32, 4, 128, 2, 1, 64, 1, (), True)
    rand = lambda *shape: torch.randn(*shape, device=device) * 0.03
    meta = NativeNvfp4LinearMetadata(torch.tensor(1., device=device), torch.tensor(1., device=device))
    attention = {"q_proj": rand(4096, 128), "k_proj": rand(512, 128),
                 "v_proj": rand(512, 128), "o_proj": rand(128, 4096)}
    return MoELayerMasterState(0, spec, torch.ones(128, device=device, dtype=torch.bfloat16),
        torch.ones(128, device=device, dtype=torch.bfloat16),
        torch.ones(128, device=device, dtype=torch.bfloat16), torch.ones(128, device=device, dtype=torch.bfloat16),
        rand(2, 128).to(torch.bfloat16), attention, {k: meta for k in attention},
        [MoEExpertMasterState(rand(64, 128), rand(64, 128), rand(128, 64), meta, meta, meta) for _ in range(2)])


def initial(state):
    diag = build_moe_diag_state(state.spec, "fusable").to(state.router_weight.device)
    with torch.no_grad():
        for p in diag.parameters():
            p.copy_(torch.linspace(-0.3, 0.4, p.numel(), device=p.device).reshape_as(p))
    return diag, fold_initial_diag(state, diag.snapshot())


def position_embeddings(length, device="cpu"):
    return (torch.ones(1, length, 128, device=device, dtype=torch.bfloat16),
            torch.zeros(1, length, 128, device=device, dtype=torch.bfloat16))


@pytest.mark.parametrize("sharing", ["linear", "group"])
def test_initialization_matches_e4(sharing):
    state = make_state()
    diag, base = initial(state)
    student = Student(base, sharing)
    actual = transformed_state(base, student.learned)
    expected = fold_fusable_moe_layer_state(state, diag, use_r64=True)
    for (name, a), (other, b) in zip(projection_weights(actual), projection_weights(expected)):
        assert name == other
        torch.testing.assert_close(a, b)
    torch.testing.assert_close(actual.input_layernorm_weight, expected.input_layernorm_weight)
    torch.testing.assert_close(actual.post_attention_layernorm_weight, expected.post_attention_layernorm_weight)


@pytest.mark.parametrize("sharing", ["linear", "group"])
def test_router_aux_gradients_and_resume(sharing, tmp_path):
    _, base = initial(make_state())
    model = Student(base, sharing)
    x = torch.randn(1, 3, 128, dtype=torch.bfloat16, requires_grad=True)
    out = model(x, position_embeddings=position_embeddings(3))
    logits = model.router_aux_logits(out.pre_moe_norm)
    teacher = torch.randn_like(logits)
    masked_router_loss(logits, teacher, torch.ones(1, 3, dtype=torch.bool), top_k=1).backward()
    gradients = {name for name, p in model.learned.named_parameters() if p.grad is not None}
    assert gradients == {"moe_norm", "matrices.router"}
    assert x.grad is None
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    optimizer.step()
    atomic_save({"parameters": model.learned.snapshot(), "optimizer": optimizer.state_dict()}, tmp_path / "resume.pt")
    copy = Student(base, sharing)
    saved = load(tmp_path / "resume.pt")
    copy.learned.load_state_dict(saved["parameters"])
    with torch.no_grad():
        a = model(x, position_embeddings=position_embeddings(3), use_ste=False).output
        b = copy(x, position_embeddings=position_embeddings(3), use_ste=False).output
    torch.testing.assert_close(a, b, atol=0, rtol=0)


class ExportedStudent(Student):
    def linear(self, x, name, weight, *, use_ste, head_dim=None, routing_weight=None):
        a = quantize(apply_r64_no_cross_head(x.float(), head_dim=head_dim), False)
        return F.linear(a, weight) if routing_weight is None else weighted_linear(a, weight, routing_weight)

    def router(self, normed):
        return F.linear(normed, self.base.router_weight)


def exported_reference(base, model, tmp_path):
    final = transformed_state(base, model.learned)
    tensors = _state_to_tensors(final)
    save_file(tensors, str(tmp_path / "layer.safetensors"))
    tensors = load_file(str(tmp_path / "layer.safetensors"))
    prefix = "model.layers.0"
    attention = {name: tensors[f"{prefix}.self_attn.{name}.weight"] for name in final.attention}
    experts = [replace(expert, **{name: tensors[f"{prefix}.mlp.experts.{i}.{name}.weight"]
                                  for name in ("gate_proj", "up_proj", "down_proj")})
               for i, expert in enumerate(final.experts)]
    final = replace(final, attention=attention, experts=experts)
    return ExportedStudent(final, "linear")


@pytest.mark.parametrize("sharing", ["linear", "group"])
def test_learned_export_matches_training_and_is_causal(sharing, tmp_path):
    _, base = initial(make_state())
    model = Student(base, sharing)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.003)
        exported = exported_reference(base, model, tmp_path)
        x = torch.randn(1, 4, 128, dtype=torch.bfloat16)
        pos = position_embeddings(4)
        out = model(x, position_embeddings=pos, use_ste=False)
        ref = exported(x, position_embeddings=pos, use_ste=False)
        torch.testing.assert_close(out.output, ref.output, rtol=0, atol=0)
        torch.testing.assert_close(out.router_logits, ref.router_logits, rtol=0, atol=0)
        changed = x.clone()
        changed[:, 2:] = 100
        other = model(changed, position_embeddings=pos, use_ste=False)
        torch.testing.assert_close(out.output[:, :2], other.output[:, :2], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for actual vLLM Triton path")
def test_vllm_r64_qdq_and_exported_linear(tmp_path):
    from vllm.model_executor.layers.quantization.hif4_transform_triton import hif4_r64_quantize_hifx4_triton
    torch.backends.cuda.matmul.allow_tf32 = False
    _, base = initial(make_state("cuda"))
    model = Student(base, "group")
    x = torch.randn(17, 128, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        model.learned.matrices["q_proj"].add_(0.002 * torch.randn_like(model.learned.matrices["q_proj"]))
        final = transformed_state(base, model.learned)
        runtime_activation = hif4_r64_quantize_hifx4_triton(x)
        reference_activation = quantize(apply_r64_no_cross_head(x.float()), False)
        torch.testing.assert_close(runtime_activation, reference_activation, rtol=0, atol=0)
        runtime_output = F.linear(runtime_activation, quantize(final.attention["q_proj"], False))
        expected = model.linear(x, "q_proj", base.attention["q_proj"], use_ste=False)
        torch.testing.assert_close(runtime_output, expected, rtol=0, atol=0)
