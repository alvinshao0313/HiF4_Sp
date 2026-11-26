"""Replay one text decoder block at a time from the published ModelOpt weights."""
import torch
from types import MethodType
from torch import nn
import torch.nn.functional as F
from transformers import NemotronHConfig, Qwen3MoeConfig, Qwen3_5TextConfig
from transformers.masking_utils import create_causal_mask
from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHBlock
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeDecoderLayer, Qwen3MoeRotaryEmbedding
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding
from transformers.models.qwen3_5 import modeling_qwen3_5

from NVFP4.torch_fake import _fake_quant_nvfp4_activation_torch
from NVFP4.triton_fake import fake_quant_nvfp4_activation_triton
from .operands import nvfp4_to_bf16, operand


class Projection(nn.Module):
    def __init__(self, source, name, device):
        super().__init__()
        self.name = name
        self.fp8 = name in source.fp8
        self.target = None
        self.channel_scale = None
        self.weight_only = False
        self.calibrating_awq = False
        self.input_scale = source.tensor(name + ".input_scale").to(device).reshape(())
        if self.fp8:
            weight = source.tensor(name + ".weight").to(device).bfloat16()
            self.weight_scale = source.tensor(name + ".weight_scale").to(device).reshape(())
        else:
            packed = source.tensor(name + ".weight")
            scales = source.tensor(name + ".weight_scale")
            global_scale = source.tensor(name + ".weight_scale_2").to(device)
            weight = torch.empty((packed.shape[0], packed.shape[1] * 2), device=device, dtype=torch.bfloat16)
            for start in range(0, len(weight), 256):
                rows = slice(start, start + 256)
                weight[rows] = nvfp4_to_bf16(packed[rows].to(device), scales[rows].to(device), global_scale)
        self.weight = nn.Parameter(weight, requires_grad=False)
        key = name + ".bias"
        self.bias = nn.Parameter(source.tensor(key).to(device), requires_grad=False) if key in source.weights else None

    def forward(self, x):
        if self.weight_only:
            return F.linear(x, self.weight, self.bias)
        if self.target:
            a = x if self.channel_scale is None else x.float() / self.channel_scale
            a = a.to(x.dtype) if self.calibrating_awq else operand(a, self.target)
            return F.linear(a, self.weight, self.bias)
        if self.fp8:
            a = (x.float() / self.input_scale).clamp(-448, 448).to(torch.float8_e4m3fn).bfloat16()
            flat = a.reshape(-1, a.shape[-1])
            product = (torch.mm(flat, self.weight.T, out_dtype=torch.float32) if flat.is_cuda
                       else flat.float() @ self.weight.float().T)
            y = product * (self.input_scale * self.weight_scale)
            if self.bias is not None:
                y = y + self.bias
            return y.reshape(*x.shape[:-1], -1).to(x.dtype)
        if self.calibrating_awq:
            return F.linear(x, self.weight, self.bias)
        quantize = fake_quant_nvfp4_activation_triton if x.is_cuda else _fake_quant_nvfp4_activation_torch
        return F.linear(quantize(x, self.input_scale.reciprocal(), output_dtype=x.dtype), self.weight, self.bias)


class LogicalExperts(nn.Module):
    """Keep logical expert projections visible to calibration and preserve top-k inference."""
    def __init__(self, original, source, prefix, device):
        super().__init__()
        self.act = original.act_fn
        self.gated = hasattr(original, "gate_up_proj")
        self.experts = nn.ModuleList()
        for index in range(original.num_experts):
            names = ["gate_proj", "up_proj", "down_proj"] if self.gated else ["up_proj", "down_proj"]
            self.experts.append(nn.ModuleDict({n: Projection(source, f"{prefix}.{index}.{n}", device) for n in names}))

    def activated(self, expert, x):
        up = expert["up_proj"](x)
        return self.act(expert["gate_proj"](x)) * up if self.gated else self.act(up)

    def forward(self, x, indices, weights):
        output = torch.zeros_like(x, dtype=x.dtype if self.gated else weights.dtype)
        for index in indices.unique().tolist():
            rows, slots = torch.where(indices == index)
            expert = self.experts[index]
            y = expert["down_proj"](self.activated(expert, x[rows]))
            output.index_add_(0, rows, (y * weights[rows, slots, None]).to(output.dtype))
        return output.to(x.dtype)


class TextBlocks:
    def __init__(self, source, device):
        self.source, self.device = source, torch.device(device)
        kind = source.config["model_type"]
        if kind == "nemotron_h":
            self.cfg = NemotronHConfig.from_dict(source.config)
            self.cls, self.root, self.embedding, self.rotary = NemotronHBlock, "backbone", "embeddings", None
        elif kind == "qwen3_moe":
            self.cfg = Qwen3MoeConfig.from_dict(source.config)
            if self.cfg.sliding_window is not None:
                raise ValueError("This release uses sliding-window attention")
            self.cls, self.root, self.embedding = Qwen3MoeDecoderLayer, "model", "embed_tokens"
            self.rotary = Qwen3MoeRotaryEmbedding(config=self.cfg).to(device)
        elif kind == "qwen3_5":
            self.cfg = Qwen3_5TextConfig.from_dict(source.config["text_config"])
            self.cls, self.root, self.embedding = Qwen3_5DecoderLayer, "model.language_model", "embed_tokens"
            self.rotary = Qwen3_5TextRotaryEmbedding(config=self.cfg).to(device)
        else:
            raise ValueError(f"Unsupported model type {kind}")
        self.cfg._attn_implementation = "sdpa"
        self.kind = kind

    def embed(self, sequences):
        table = self.source.tensor(f"{self.root}.{self.embedding}.weight")
        return [F.embedding(ids, table).unsqueeze(0).bfloat16() for ids in sequences]

    def load(self, index):
        prefix = f"{self.root}.layers.{index}"
        with torch.device("meta"):
            layer = self.cls(self.cfg, layer_idx=index)
            if self.device.type == "cpu" and hasattr(layer, "linear_attn"):
                # Transformers selects these extensions by import availability.
                # Its torch implementations run the same GDN on CPU.
                gdn = layer.linear_attn
                gdn.causal_conv1d_fn = None
                gdn.chunk_gated_delta_rule = modeling_qwen3_5.torch_chunk_gated_delta_rule
                gdn.norm = modeling_qwen3_5.Qwen3_5RMSNormGated(
                    gdn.head_v_dim, eps=gdn.layer_norm_epsilon)
        for path, module in list(layer.named_modules()):
            full = f"{prefix}.{path}"
            if path.endswith(".experts"):
                layer.set_submodule(path, LogicalExperts(module, self.source, full, self.device))
            elif full in self.source.native or full in self.source.fp8:
                layer.set_submodule(path, Projection(self.source, full, self.device))
        # Some fused layers allocate on their explicit device even inside a
        # meta context. Load all original state, including these norm weights.
        for path, tensor in list(layer.named_parameters()) + list(layer.named_buffers()):
            module, _, name = path.rpartition(".")
            dest = layer.get_submodule(module)
            if isinstance(dest, Projection):
                continue
            value = self.source.tensor(f"{prefix}.{path}").to(self.device)
            if value.shape != tensor.shape:
                raise ValueError(f"{prefix}.{path}: {value.shape} != {tensor.shape}")
            if name in dest._parameters:
                value = nn.Parameter(value, requires_grad=False)
            setattr(dest, name, value)
        expected = {name for name in self.source.native if name.startswith(prefix + ".")}
        actual = {m.name for m in layer.modules() if isinstance(m, Projection) and not m.fp8}
        if actual != expected:
            raise ValueError(f"Missing logical projections in {prefix}: {actual ^ expected}")
        if self.kind == "nemotron_h" and self.device.type == "cuda" and hasattr(layer.mixer, "ssm_state_size"):
            from .mamba import calibration_forward
            layer.mixer.forward = MethodType(calibration_forward, layer.mixer)
        return layer.eval()

    def kwargs(self, index, x):
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        causal = (self.kind == "qwen3_moe" or
                  (self.kind == "qwen3_5" and self.cfg.layer_types[index] == "full_attention") or
                  (self.kind == "nemotron_h" and self.cfg.layers_block_type[index] == "attention"))
        mask = create_causal_mask(config=self.cfg, inputs_embeds=x, attention_mask=None,
                                  past_key_values=None, position_ids=pos) if causal else None
        args = {"position_ids": pos, "attention_mask": mask, "use_cache": False}
        if self.rotary is not None:
            rotary_pos = pos.unsqueeze(0).expand(3, -1, -1) if self.kind == "qwen3_5" else pos
            args["position_embeddings"] = self.rotary(x, rotary_pos)
        return args
