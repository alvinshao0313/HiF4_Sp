"""Process-local extension for learned rotations; shared vLLM files stay intact."""
from __future__ import annotations

from pathlib import Path
import json

import torch
from safetensors import safe_open

RUNTIME_KIND = "module_ablation_rotation_v1"
_MATRICES = {}


def _matrix(spec, path, layer_idx, name, device, width):
    key = (str(path), layer_idx, name, str(device), width)
    if key not in _MATRICES:
        filename = spec["rotation_files"][str(layer_idx)]
        if Path(filename).name != filename:
            raise ValueError("rotation sidecar assets must be files inside the model directory")
        with safe_open(str(Path(path).parent / filename), framework="pt", device="cpu") as handle:
            q = handle.get_tensor(name)
        groups = width // 64
        if width % 64 or q.dtype != torch.float32 or tuple(q.shape[-2:]) != (64, 64):
            raise ValueError("invalid FP32 G64 activation matrix")
        if q.shape[-3] != groups:
            from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
            tp, rank = get_tensor_model_parallel_world_size(), get_tensor_model_parallel_rank()
            if name not in {"o_proj", "down_proj"} or q.shape[-3] != groups * tp:
                raise ValueError("only row-parallel O/down rotation input groups may be TP-sharded")
            q = q.narrow(-3, rank * groups, groups)
        _MATRICES[key] = q.contiguous().to(device)
    return _MATRICES[key]


def install_runtime():
    from vllm.model_executor.layers import linear
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import UnquantizedFusedMoEMethod
    from vllm.model_executor.layers.quantization import hif4_runtime
    from .rotation_kernel import rotate_qdq
    from .rotation_moe import apply_moe

    if getattr(linear.UnquantizedLinearMethod, "_ablation_rotation_installed", False):
        return
    old_linear = linear.UnquantizedLinearMethod.apply
    old_moe = UnquantizedFusedMoEMethod.apply

    def linear_apply(self, layer, x, bias=None):
        path = hif4_runtime._sidecar_path_from_config(self.hif4_runtime_spec_path)
        spec = None if path is None else hif4_runtime.load_hif4_runtime_spec(path)
        if spec is None or spec.get("variant") != RUNTIME_KIND:
            return old_linear(self, layer, x, bias)
        prefix = getattr(layer, "prefix", "")
        gemm = linear.dispatch_unquantized_gemm()
        if prefix == "lm_head" or prefix.endswith((".lm_head", ".mlp.gate")):
            return gemm(layer, x, layer.weight, bias)
        if self.hif4_fake_act or self.nvf4_fake_act:
            raise ValueError("learned rotation cannot combine with another fake-activation pass")
        index = hif4_runtime.layer_index_from_prefix(prefix)
        hif4_runtime.set_current_layer(x.device, index)
        if prefix.endswith(".qkv_proj"):
            sizes = list(layer.output_partition_sizes)
            if len(sizes) != 3 or sum(sizes) != layer.weight.shape[0]:
                raise ValueError("QKV local output partition mismatch")
            result, offset = [], 0
            for name, size in zip(("q_proj", "k_proj", "v_proj"), sizes):
                q = _matrix(spec, path, index, name, x.device, x.shape[-1])
                act = rotate_qdq(x.contiguous(), q)
                result.append(gemm(layer, act, layer.weight[offset:offset + size],
                                   None if bias is None else bias[offset:offset + size]))
                offset += size
            return torch.cat(result, dim=-1)
        if not prefix.endswith(".o_proj"):
            raise ValueError(f"unexpected learned-rotation projection {prefix}")
        q = _matrix(spec, path, index, "o_proj", x.device, x.shape[-1])
        return gemm(layer, rotate_qdq(x.contiguous(), q), layer.weight, bias)

    def moe_apply(self, layer, x, topk_weights, topk_ids, shared_experts_input):
        path = hif4_runtime._sidecar_path_from_config(self.hif4_runtime_spec_path)
        spec = None if path is None else hif4_runtime.load_hif4_runtime_spec(path)
        if spec is None or spec.get("variant") != RUNTIME_KIND:
            return old_moe(self, layer, x, topk_weights, topk_ids, shared_experts_input)
        # The runner passes hidden_states here even without shared experts; any
        # separate shared-expert computation belongs to that runner, as upstream.
        index = hif4_runtime.current_layer(x.device)
        rotations = {name: _matrix(spec, path, index, name, x.device, width)
                     for name, width in (("gate_proj", x.shape[-1]), ("up_proj", x.shape[-1]),
                                         ("down_proj", layer.w2_weight.shape[-1]))}
        return apply_moe(x, layer.w13_weight, layer.w2_weight, topk_weights, topk_ids, rotations)

    linear.UnquantizedLinearMethod.apply = linear_apply
    UnquantizedFusedMoEMethod.apply = moe_apply
    linear.UnquantizedLinearMethod._ablation_rotation_installed = True


def install_engine_hook():
    """Pass an explicit worker class through normal vLLM engine configuration."""
    from vllm.entrypoints.llm import LLM
    if getattr(LLM, "_ablation_engine_hook", False):
        return
    original = LLM.__init__

    def initialize(self, *args, **kwargs):
        additional = kwargs.get("additional_config") or {}
        path = additional.get("hif4_runtime_spec_path")
        if path is not None:
            spec = torch.load(path, map_location="cpu", weights_only=False)
            if spec.get("variant") == RUNTIME_KIND:
                if kwargs.get("worker_cls", "auto") != "auto":
                    raise ValueError("rotation worker must not override another custom worker")
                kwargs["worker_cls"] = "Native_NVFP4_HiF4_Linear_Puncture.experiments.module_design_ablation.worker.Worker"
        original(self, *args, **kwargs)
        engine = self.llm_engine
        parallel = engine.vllm_config.parallel_config
        print("ABLATION_ENGINE_CONFIG " + json.dumps({
            "quantization": str(engine.model_config.quantization),
            "tensor_parallel_size": parallel.tensor_parallel_size,
            "worker_cls": parallel.worker_cls,
        }), flush=True)

    LLM.__init__ = initialize
    LLM._ablation_engine_hook = True
