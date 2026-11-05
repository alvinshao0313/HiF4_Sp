"""Continue B's existing parameterization and add an unquantized folded head."""
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import load_file
from torch.utils.checkpoint import checkpoint

from ..e2e_diag_reconstruction.core.modelopt_moe_checkpoint import load_qwen3_moe_layer_state
from ..e2e_diag_reconstruction.core.moe_semantic_hif4 import _rms_norm
from ..module_design_ablation.student import Student
from ..non_equivalent_reconstruction.transforms import block_right
from .artifacts import load, read_json, state_digest
from .config import NATIVE, PROTOCOL


def non_layer(model, device="cpu"):
    model = Path(model)
    index = read_json(model / "model.safetensors.index.json")["weight_map"]
    return load_file(str(model / index["lm_head.weight"]), device=str(device))


class LastLayer(Student):
    def linear(self, x, name, weight, *, use_ste, head_dim=None, routing_weight=None):
        def project(value, route):
            return super(LastLayer, self).linear(value, name, weight, use_ste=use_ste,
                                                head_dim=head_dim, routing_weight=route)
        if use_ste and torch.is_grad_enabled():
            return checkpoint(project, x, routing_weight, use_reentrant=False)
        return project(x, routing_weight)


class LastHead(nn.Module):
    def __init__(self, manifest, device):
        super().__init__()
        self.base = load_qwen3_moe_layer_state(NATIVE, 47, device)
        self.layer = LastLayer(self.base, "matrix_norm").to(device)
        self.layer.learned.load_state_dict(load(manifest["layer_parameters"])["parameters"], strict=True)
        values = non_layer(manifest["initialization"], device)
        self.register_buffer("head_base", values["lm_head.weight"])
        self.final_norm = nn.Parameter(values["model.norm.weight"].float().clone())
        self.head_matrix = nn.Parameter(torch.eye(64, device=device).repeat(32, 1, 1))
        self.check_trainable()

    def check_trainable(self):
        base_tensors = [self.base.router_weight, self.base.q_norm_weight, self.base.k_norm_weight,
                        self.head_base, *self.base.attention.values()]
        base_tensors += [getattr(e, name) for e in self.base.experts
                         for name in ("gate_proj", "up_proj", "down_proj")]
        if any(t.requires_grad for t in base_tensors):
            raise RuntimeError("base projections/head/router/QKnorm must be frozen")
        names = []
        for name, p in self.named_parameters():
            if p.requires_grad:
                if not (name in ("final_norm", "head_matrix")
                        or name.startswith(("layer.learned.matrices.", "layer.learned.norms."))):
                    raise RuntimeError(f"unexpected trainable parameter: {name}")
                if p.dtype != torch.float32 or "router" in name:
                    raise RuntimeError("trainable matrices/norm must be FP32, with frozen router")
                names.append(name)
        count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if count != PROTOCOL["trainable_parameters"]:
            raise RuntimeError(f"unexpected trainable count {count}")
        return names

    def forward(self, hidden, position_embeddings, *, use_ste):
        out = self.layer(hidden, position_embeddings=position_embeddings, use_ste=use_ste).output
        return _rms_norm(out, self.final_norm, 1e-6)

    def fused_head(self):
        # No activation transform or QDQ at the head. Build this once per batch.
        return block_right(self.head_base, self.head_matrix).to(torch.bfloat16)

    def snapshot(self):
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items() if k != "head_base"}

    def restore(self, values):
        missing, unexpected = self.load_state_dict(values, strict=False)
        if missing != ["head_base"] or unexpected:
            raise RuntimeError("incomplete joint checkpoint")

    def frozen_digest(self):
        tensors = {"router": self.base.router_weight, "q_norm": self.base.q_norm_weight,
                   "k_norm": self.base.k_norm_weight, "head": self.head_base, **self.base.attention}
        for i, expert in enumerate(self.base.experts):
            for name in ("gate_proj", "up_proj", "down_proj"):
                tensors[f"expert_{i}_{name}"] = getattr(expert, name)
        return state_digest(tensors)
