"""Fold RMSNorm gains and rotate the complete residual stream before PTQ."""
import re

import torch

from .operands import nvfp4_to_bf16, operand, random_block


class FusedRotation:
    def __init__(self, source, target, seed, device):
        self.source, self.target, self.device = source, target, device
        self.size = {"hif4": 64, "mxfp4": 32}[target]
        self.rotation = random_block(self.size, seed, device)
        kind = source.config["model_type"]
        cfg = source.config.get("text_config", source.config)
        if source.fp8:
            raise ValueError("Fused BRQ does not support residual FP8 projections")
        self.tied_head = cfg.get("tie_word_embeddings", False)
        if cfg["hidden_size"] % self.size or cfg.get("moe_latent_size") is not None:
            raise ValueError("Unsupported residual width or latent-space MoE for fused BRQ")
        if kind == "nemotron_h":
            self.root, embed, final_norm = "backbone", "embeddings", "norm_f"
        elif kind in ("qwen3_moe", "qwen3_5"):
            self.root = "model" if kind == "qwen3_moe" else "model.language_model"
            embed, final_norm = "embed_tokens", "norm"
        else:
            raise ValueError(f"Unsupported fused BRQ topology: {kind}")
        self.offset = 1 if kind == "qwen3_5" else 0
        self.embedding = f"{self.root}.{embed}.weight"
        self.inputs = {"lm_head": f"{self.root}.{final_norm}.weight"}
        self.outputs = set()
        for key in source.weights:
            match = re.fullmatch(re.escape(self.root) + r"\.layers\.(\d+)\.(.+)\.weight", key)
            if not match:
                continue
            index, path = match.groups()
            prefix = f"{self.root}.layers.{index}"
            name = key.removesuffix(".weight")
            leaf = path.rsplit(".", 1)[-1]
            if kind == "nemotron_h":
                if leaf in ("in_proj", "q_proj", "k_proj", "v_proj", "up_proj", "gate_proj", "gate"):
                    self.inputs[name] = prefix + ".norm.weight"
                elif leaf in ("out_proj", "o_proj", "down_proj"):
                    self.outputs.add(name)
            elif path.startswith(("self_attn.", "linear_attn.")):
                if leaf in ("q_proj", "k_proj", "v_proj", "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"):
                    self.inputs[name] = prefix + ".input_layernorm.weight"
                elif leaf in ("o_proj", "out_proj"):
                    self.outputs.add(name)
            elif path.startswith("mlp."):
                if leaf in ("gate_proj", "up_proj", "gate"):
                    self.inputs[name] = prefix + ".post_attention_layernorm.weight"
                elif leaf == "down_proj":
                    self.outputs.add(name)
        self.norms = set(self.inputs.values())
        required = self.norms | {self.embedding} | (set() if self.tied_head else {"lm_head.weight"})
        if required - source.weights.keys():
            raise ValueError(f"Missing fused rotation state: {required - source.weights.keys()}")
        missing = source.native - self.inputs.keys() - self.outputs
        if missing:
            raise ValueError(f"Unclassified NVFP4 projections: {sorted(missing)}")

    def right(self, value):
        return (value.unflatten(-1, (-1, self.size)) @ self.rotation).reshape_as(value)

    def tensor(self, key, quantize=True):
        source = self.source
        if key in self.norms:
            return torch.full_like(source.tensor(key), 1 - self.offset)
        name, _, suffix = key.rpartition(".")
        if suffix == "bias" and name in self.outputs:
            return self.right(source.tensor(key).to(self.device).float()).to(source.tensor(key).dtype).cpu()
        if suffix != "weight" or (name not in self.inputs and name not in self.outputs and key != self.embedding):
            return source.tensor(key)
        native = name in source.native
        packed = source.tensor(self.embedding if key == "lm_head.weight" and self.tied_head else key)
        shape = (packed.shape[0], packed.shape[1] * 2) if native else packed.shape
        result = torch.empty(shape, dtype=torch.bfloat16 if native else packed.dtype)
        gain = (source.tensor(self.inputs[name]).to(self.device).float() + self.offset
                if name in self.inputs else None)
        if native:
            scales = source.tensor(name + ".weight_scale")
            global_scale = source.tensor(name + ".weight_scale_2").to(self.device)
        # 256 rows contain whole G32/G64 output blocks, so left rotations do
        # not cross tile boundaries. Keep the final tensor on the host.
        for start in range(0, shape[0], 256):
            rows = slice(start, start + 256)
            w = (nvfp4_to_bf16(packed[rows].to(self.device), scales[rows].to(self.device), global_scale)
                 if native else packed[rows].to(self.device)).float()
            if gain is not None:
                w = self.right(w * gain)
            elif key == self.embedding:
                w = self.right(w)
            if name in self.outputs:
                w = self.right(w.T).T.contiguous()
            result[rows] = operand(w, self.target) if native and quantize else w
        return result

    def metadata(self):
        return {"rotation": "block_diag(DH/sqrt(G))", "group_size": self.size,
                "folded_norms": sorted(self.norms), "rotated_input_projections": sorted(self.inputs),
                "rotated_output_projections": sorted(self.outputs), "embedding": self.embedding,
                "activation_rotation": "absorbed_in_residual_stream"}
