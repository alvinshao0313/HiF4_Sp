import inspect
import logging
import math
import pathlib
import sys

import torch
import torch.nn as nn
import tqdm

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from hif4_gpu.quant_cy import QType, quant_dequant_float


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def _move_to_device(obj, device: torch.device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move_to_device(x, device) for x in obj)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    return obj


def _filter_kwargs_for_callable(callable_obj, kwargs):
    signature = inspect.signature(callable_obj)
    params = signature.parameters.values()
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
        return kwargs
    valid_keys = set(signature.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in valid_keys}


def _extract_hidden(output):
    if isinstance(output, tuple):
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    return output


def _run_layer(layer, hidden_states, layer_kwargs):
    call_kwargs = _filter_kwargs_for_callable(layer.forward, layer_kwargs)
    call_kwargs = _move_to_device(call_kwargs, hidden_states.device)
    output = layer(hidden_states, **call_kwargs)
    return _extract_hidden(output)


def _get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise NotImplementedError("Only decoder-only models with model.layers are supported.")


def _is_qwen3_5_text_model(model):
    return getattr(model.config, "model_type", "") == "qwen3_5_text"


def _get_quant_groups(model, layer=None):
    model_type = getattr(model.config, "model_type", "")
    if model_type not in {"llama", "qwen3", "qwen3_5", "qwen3_5_text"}:
        raise NotImplementedError(f"Model type {model_type} is out of scope. Supported: llama, qwen3, qwen3_5.")

    mlp_groups = [
        ["mlp.gate_proj", "mlp.up_proj"],
        ["mlp.down_proj"],
    ]

    if model_type == "qwen3_5_text":
        if layer is None:
            raise ValueError("Qwen3.5 GPTQ requires a concrete decoder layer to select quantization groups.")

        layer_type = getattr(layer, "layer_type", None)
        if layer_type == "linear_attention":
            return [
                ["linear_attn.in_proj_qkv"],
                ["linear_attn.in_proj_z"],
                ["linear_attn.in_proj_b"],
                ["linear_attn.in_proj_a"],
                ["linear_attn.out_proj"],
                *mlp_groups,
            ]
        if layer_type == "full_attention":
            return [
                ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
                ["self_attn.o_proj"],
                *mlp_groups,
            ]
        raise ValueError(f"Unsupported Qwen3.5 layer_type: {layer_type}")

    return [
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        ["self_attn.o_proj"],
        *mlp_groups,
    ]


def _validate_no_padding_attention_mask(batch) -> None:
    if not isinstance(batch, dict) or "attention_mask" not in batch:
        return

    attention_mask = batch["attention_mask"]
    if torch.is_tensor(attention_mask) and not torch.all(attention_mask == 1):
        raise ValueError("Qwen3.5 GPTQ calibration does not support padded attention_mask yet.")


def _make_causal_mask(hidden_states: torch.Tensor) -> torch.Tensor:
    if not torch.is_floating_point(hidden_states):
        raise TypeError("Causal mask requires floating-point hidden states.")

    batch_size, seq_len, _ = hidden_states.shape
    min_value = torch.finfo(hidden_states.dtype).min
    mask = torch.full((seq_len, seq_len), min_value, dtype=hidden_states.dtype, device=hidden_states.device)
    mask = torch.triu(mask, diagonal=1)
    return mask.view(1, 1, seq_len, seq_len).expand(batch_size, 1, seq_len, seq_len)


def _layer_kwargs_for_current_layer(model, layer, hidden_states, base_layer_kwargs):
    layer_kwargs = dict(base_layer_kwargs)
    if not _is_qwen3_5_text_model(model):
        return layer_kwargs

    layer_type = getattr(layer, "layer_type", None)
    if layer_type == "linear_attention":
        layer_kwargs["attention_mask"] = None
    elif layer_type == "full_attention":
        layer_kwargs["attention_mask"] = _make_causal_mask(hidden_states)
    else:
        raise ValueError(f"Unsupported Qwen3.5 layer_type: {layer_type}")
    return layer_kwargs


def find_qlayers(module, layers=None, name=""):
    if layers is None:
        layers = [nn.Linear]
    if type(module) in layers:
        return {name: module}

    res = {}
    for name1, child in module.named_children():
        child_name = name + "." + name1 if name else name1
        res.update(find_qlayers(child, layers=layers, name=child_name))
    return res


class WeightHiFxQuantizer(nn.Module):
    def __init__(self, qtype="hifx4"):
        super().__init__()
        self.qparams = QType(qtype)
        self._cached_scale = None
        self._cached_width = 0
        self._cached_col = 0

    @staticmethod
    def _bf16_round(x):
        return x.to(torch.bfloat16).to(torch.float32)

    @staticmethod
    def _e6m2_round(x):
        e_sf = torch.floor(torch.log2(x))
        return torch.round(x * torch.exp2(2 - e_sf)) * torch.exp2(e_sf - 2)

    def _quantize_with_scale(self, x, scale):
        x_fp32 = x.float()
        scale = scale.float().clamp(min=2 ** (-48))
        sign = torch.sign(x_fp32)
        mant = torch.abs(x_fp32) / scale
        mant = torch.floor(mant * 2 ** (self.qparams.man_bits - 1) + 0.5)
        mant = mant / 2 ** (self.qparams.man_bits - 1)
        mant = torch.clamp(mant, max=2 - 2 ** (-self.qparams.man_bits + 1))
        return (sign * mant * scale).to(x.dtype)

    def _extract_hifx4_1_group_scale(self, x):
        x = x.float()
        orig_cols = x.shape[-1]
        block = self.qparams.blk_size * self.qparams.blk_outer_size
        pad_cols = (block - orig_cols % block) % block
        if pad_cols > 0:
            x = torch.nn.functional.pad(x, (0, pad_cols), value=0.0)

        x_group = x.unflatten(-1, (-1, 64))
        max_abs = torch.max(torch.abs(x_group), dim=-1, keepdim=True)[0]
        max_mant = 2 - 2 ** (-self.qparams.man_bits + 1)
        scale_factor = max_abs * self._bf16_round(torch.ones_like(max_abs) / max_mant)
        scale_factor = self._bf16_round(scale_factor).clip(min=2 ** (-48), max=49152)
        scale_factor = self._e6m2_round(scale_factor)

        full_scale = scale_factor.expand_as(x_group)
        full_scale = full_scale.flatten(-2, -1)[..., :orig_cols].contiguous()
        return full_scale

    def _extract_group_scale(self, x):
        if self.qparams.desc == "hifx4_1":
            return self._extract_hifx4_1_group_scale(x)

        x = x.float()
        orig_cols = x.shape[-1]
        block = self.qparams.blk_size * self.qparams.blk_outer_size
        pad_cols = (block - orig_cols % block) % block
        if pad_cols > 0:
            x = torch.nn.functional.pad(x, (0, pad_cols), value=0.0)

        x_group = x.unflatten(-1, (-1, 8, 2, 4))
        x_unsigned = torch.abs(x_group)

        max_lv3 = torch.max(x_unsigned, dim=-1, keepdim=True)[0]
        max_lv2 = torch.max(max_lv3, dim=-2, keepdim=True)[0]
        max_lv1 = torch.max(max_lv2, dim=-3, keepdim=True)[0]

        div7 = self._bf16_round(torch.ones_like(max_lv1) / 7.0)
        scale_factor = max_lv1 * div7
        scale_factor = self._bf16_round(scale_factor).clip(min=2 ** (-48), max=49152)

        e_sf = torch.floor(torch.log2(scale_factor))
        mant_sf = scale_factor / torch.exp2(e_sf) * 2 ** 7
        scale_factor = torch.round(mant_sf) / 2 ** 7 * torch.exp2(e_sf)

        scale_factor = self._e6m2_round(scale_factor)

        rec_sf = self._bf16_round(1.0 / scale_factor)
        scale_lv2 = torch.exp2(torch.floor((max_lv2 * rec_sf).clip(0, 4) / 4))
        scale_lv3 = torch.exp2(torch.floor((max_lv3 * rec_sf / scale_lv2).clip(0, 2) / 2))

        full_scale = (scale_factor * scale_lv2 * scale_lv3).expand_as(x_group)
        full_scale = full_scale.flatten(-4, -1)[..., :orig_cols].contiguous()
        return full_scale

    def forward(self, x, block_size=1):
        del block_size
        if not x.is_contiguous():
            x = x.contiguous()

        if self._cached_scale is not None and x.ndim == 2 and x.shape[0] == 1:
            if self._cached_col >= self._cached_width:
                raise RuntimeError("WeightHiFxQuantizer cached group columns are exhausted.")
            scale = self._cached_scale[:, self._cached_col].view_as(x)
            qx = self._quantize_with_scale(x, scale)
            self._cached_col += 1
            return qx

        qp_in = self.qparams.dim(-1)
        qx = quant_dequant_float(x, qp_in, force_fp32=True)
        return qx.to(x.dtype)

    def find_params(self, x):
        if not x.is_contiguous():
            x = x.contiguous()
        self._cached_scale = self._extract_group_scale(x)
        self._cached_width = x.shape[-1]
        self._cached_col = 0

    def ready(self):
        return True


class GPTQ:
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        w = layer.weight.data.clone()
        self.columns = w.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        del out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterquant(self, blocksize=128, groupsize=-1, percdamp=0.01):
        W = self.layer.weight.data.clone().float()
        if not self.quantizer.ready():
            self.quantizer.find_params(W)

        H = self.H
        del self.H

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Q = torch.zeros_like(W)
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1 and (i1 + i) % groupsize == 0:
                    self.quantizer.find_params(W[:, (i1 + i) : (i1 + i + groupsize)])

                q = self.quantizer(w.unsqueeze(0)).flatten()
                Q1[:, i] = q

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        if torch.cuda.is_available() and self.layer.weight.is_cuda:
            torch.cuda.synchronize(self.layer.weight.device)

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if torch.any(torch.isnan(self.layer.weight.data)):
            raise ValueError("NaN in quantized weights")

    def free(self):
        self.H = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@torch.no_grad()
def gptq_fwrd(model, dataloader, dev, args):
    logging.info("----- HiFloat4 GPTQ Quantization -----")
    device = torch.device(dev)
    weight_qtype = getattr(args, "hif4_weight_qtype", "hifx4")
    if weight_qtype == "hifx4_1" and getattr(args, "block_size_linear", 64) != 64:
        raise ValueError("hif4-1 GPTQ requires --block_size_linear 64.")

    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = _get_layers(model)

    if hasattr(model.model, "embed_tokens"):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
    if hasattr(model.model, "norm"):
        model.model.norm = model.model.norm.to(device)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(device)
    layers[0] = layers[0].to(device)

    dtype = next(iter(model.parameters())).dtype
    max_samples = args.gptq_cal_nsamples
    inps = torch.zeros((max_samples, args.gptq_cal_seqlen, model.config.hidden_size), dtype=dtype, device=device)
    cache = {"i": 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            idx = cache["i"]
            if idx < max_samples:
                inps[idx] = inp
            cache["i"] += 1
            for key, val in kwargs.items():
                cache[key] = val
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        if cache["i"] >= max_samples:
            break

        if _is_qwen3_5_text_model(model):
            _validate_no_padding_attention_mask(batch)

        if isinstance(batch, (list, tuple)):
            input_ids = batch[0]
        elif isinstance(batch, dict):
            input_ids = batch["input_ids"]
        else:
            input_ids = batch

        try:
            model(input_ids.to(device))
        except ValueError:
            pass

    layers[0] = layers[0].module

    nsamples = min(cache["i"], max_samples)
    if nsamples == 0:
        raise RuntimeError("Calibration dataloader produced zero samples.")

    inps = inps[:nsamples]
    outs = torch.zeros_like(inps)

    layers[0] = layers[0].cpu()
    if hasattr(model.model, "embed_tokens"):
        model.model.embed_tokens = model.model.embed_tokens.cpu()
    if hasattr(model.model, "norm"):
        model.model.norm = model.model.norm.cpu()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    layer_kwargs = {k: v for k, v in cache.items() if k != "i"}

    for i in tqdm.tqdm(range(len(layers)), desc="(GPTQ Quant.) Layers"):
        layer = layers[i].to(device)
        full = find_qlayers(layer, layers=[nn.Linear])
        quant_groups = _get_quant_groups(model, layer)

        for names in quant_groups:
            subset = {name: full[name] for name in names if name in full}
            if not subset:
                continue

            gptq_blocks = {}
            for name, sub_layer in subset.items():
                if "lm_head" in name:
                    continue
                gptq_blocks[name] = GPTQ(sub_layer)
                gptq_blocks[name].quantizer = WeightHiFxQuantizer(qtype=weight_qtype)

            if not gptq_blocks:
                continue

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq_blocks[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = [subset[name].register_forward_hook(add_batch(name)) for name in gptq_blocks]

            for j in range(nsamples):
                layer_input = inps[j].unsqueeze(0)
                current_layer_kwargs = _layer_kwargs_for_current_layer(model, layer, layer_input, layer_kwargs)
                outs[j] = _run_layer(layer, layer_input, current_layer_kwargs)

            for handle in handles:
                handle.remove()

            for block in gptq_blocks.values():
                block.fasterquant(
                    percdamp=args.gptq_percdamp,
                    groupsize=getattr(args, "block_size_linear", 64),
                )
                block.free()

        for j in range(nsamples):
            layer_input = inps[j].unsqueeze(0)
            current_layer_kwargs = _layer_kwargs_for_current_layer(model, layer, layer_input, layer_kwargs)
            outs[j] = _run_layer(layer, layer_input, current_layer_kwargs)

        layers[i] = layer.cpu()
        del layer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    logging.info("----- HiFloat4 GPTQ Quantization Done -----")
