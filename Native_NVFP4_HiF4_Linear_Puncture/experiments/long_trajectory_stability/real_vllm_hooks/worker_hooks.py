from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

from .hook_spec import predictor_abs_position
from .hook_state import HookCaptureState


_STATE_ATTR = "_hif4_longtraj_diag_state"
_HANDLES_ATTR = "_hif4_longtraj_hook_handles"
_SCHEMA_VERSION = 1


class InstallHooksOp:
    """Picklable apply_model payload. Must not be a dataclass: msgspec maps those to dict."""

    def __init__(self, variant: str, output_root: str, capture_level: str = "core") -> None:
        self.variant = str(variant)
        self.output_root = str(output_root)
        self.capture_level = str(capture_level)

    def __call__(self, model) -> dict:
        return install_hooks(model, self.variant, self.output_root, self.capture_level)


class BeginSampleOp:
    """Picklable apply_model payload. Must not be a dataclass: msgspec maps those to dict."""

    def __init__(
        self,
        sample_key: str,
        prompt_len: int,
        probe_abs_to_decode: dict[int, int],
    ) -> None:
        self.sample_key = str(sample_key)
        self.prompt_len = int(prompt_len)
        self.probe_abs_to_decode = {int(k): int(v) for k, v in probe_abs_to_decode.items()}

    def __call__(self, model) -> dict:
        return begin_sample(
            model,
            self.sample_key,
            self.prompt_len,
            self.probe_abs_to_decode,
        )


def _state(model) -> HookCaptureState:
    state = getattr(model, _STATE_ATTR, None)
    if not isinstance(state, HookCaptureState):
        raise RuntimeError("real-vLLM hook state is not installed")
    return state


def _tensor_cpu_row(tensor: torch.Tensor, row: int) -> torch.Tensor:
    if tensor.ndim == 0:
        raise RuntimeError("cannot capture scalar as token-row tensor")
    if row < 0 or row >= tensor.shape[0]:
        raise RuntimeError(f"row={row} out of tensor shape={tuple(tensor.shape)}")
    return tensor[row].detach().to(device="cpu").clone()


def _capture(
    state: HookCaptureState,
    *,
    boundary: str,
    layer: int | None,
    tensor: torch.Tensor,
    role: str,
) -> None:
    if state.capture_level == "feature_scan" and (
        boundary not in {"input_norm", "post_attn_norm"} or role != "normalized"
    ):
        return
    if state.sample_key is None or not state.active_rows:
        return
    if len(state.active_rows) != len(state.active_decode_indices):
        raise RuntimeError("active row/decode mapping is inconsistent")
    state.fire_counts[boundary] = state.fire_counts.get(boundary, 0) + 1
    for row, decode_index in zip(state.active_rows, state.active_decode_indices, strict=True):
        state.records.append(
            {
                "schema_version": _SCHEMA_VERSION,
                "sample_key": state.sample_key,
                "variant": state.variant,
                "tp_rank": state.rank,
                "tp_world_size": state.world_size,
                "layer": layer,
                "boundary": boundary,
                "role": role,
                "decode_index": int(decode_index),
                "abs_position": predictor_abs_position(state.prompt_len, int(decode_index)),
                "dtype": str(tensor.dtype),
                "original_shape": tuple(int(x) for x in tensor.shape),
                "tensor": _tensor_cpu_row(tensor, row),
            }
        )


def _get_positions(args: tuple, kwargs: dict) -> torch.Tensor:
    positions = kwargs.get("positions")
    if positions is None and len(args) >= 2:
        positions = args[1]
    if not isinstance(positions, torch.Tensor):
        raise RuntimeError("Qwen3MoeModel hook did not receive Tensor positions")
    if positions.ndim != 1:
        positions = positions.reshape(-1)
    return positions


def _model_pre_hook(model, state: HookCaptureState):
    def hook(_module, args: tuple, kwargs: dict) -> None:
        if state.sample_key is None:
            state.active_rows.clear()
            state.active_decode_indices.clear()
            return
        positions = _get_positions(args, kwargs)
        state.active_rows.clear()
        state.active_decode_indices.clear()

        if not state.prefill_done:
            # Prefill may be chunked. We inspect positions only during these few
            # chunks; after prefill decode steps are counted without per-step GPU sync.
            pos_cpu = [int(x) for x in positions.detach().cpu().tolist()]
            for row, abs_pos in enumerate(pos_cpu):
                decode_index = state.probe_abs_to_decode.get(abs_pos)
                if decode_index is not None:
                    state.active_rows.append(row)
                    state.active_decode_indices.append(int(decode_index))
            if state.prompt_len - 1 in pos_cpu:
                state.prefill_done = True
                state.next_decode_index = 1
            return

        if positions.numel() != 1:
            raise RuntimeError(
                "real-vLLM diagnostic requires one-token incremental decode after prefill; "
                f"got positions shape={tuple(positions.shape)}"
            )
        decode_index = state.next_decode_index
        state.next_decode_index += 1
        if decode_index not in state.probe_decode_indices:
            return
        expected_abs = predictor_abs_position(state.prompt_len, decode_index)
        actual_abs = int(positions.item())
        if actual_abs != expected_abs:
            raise RuntimeError(
                f"decode position mismatch for {state.sample_key}: "
                f"j={decode_index} expected_abs={expected_abs} actual_abs={actual_abs}"
            )
        state.active_rows.append(0)
        state.active_decode_indices.append(decode_index)

    return hook


def _parse_layer_inputs(args: tuple, kwargs: dict) -> tuple[torch.Tensor, torch.Tensor | None]:
    hidden = kwargs.get("hidden_states")
    residual = kwargs.get("residual")
    if hidden is None and len(args) >= 2:
        hidden = args[1]
    if residual is None and len(args) >= 3:
        residual = args[2]
    if not isinstance(hidden, torch.Tensor):
        raise RuntimeError("decoder layer did not receive hidden_states Tensor")
    if residual is not None and not isinstance(residual, torch.Tensor):
        raise RuntimeError("decoder layer residual is neither Tensor nor None")
    return hidden, residual


def _layer_pre_hook(state: HookCaptureState, layer_index: int):
    def hook(_module, args: tuple, kwargs: dict) -> None:
        state.current_layer = layer_index
        hidden, residual = _parse_layer_inputs(args, kwargs)
        _capture(state, boundary="layer_in", layer=layer_index, tensor=hidden, role="branch")
        if residual is not None:
            _capture(state, boundary="layer_in", layer=layer_index, tensor=residual, role="residual")

    return hook


def _layer_post_hook(state: HookCaptureState, layer_index: int):
    def hook(_module, _args: tuple, _kwargs: dict, output: Any) -> None:
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError(f"decoder layer output must be tuple[2], got {type(output)}")
        branch, residual = output
        if not isinstance(branch, torch.Tensor) or not isinstance(residual, torch.Tensor):
            raise RuntimeError("decoder layer tuple must contain two Tensors")
        _capture(state, boundary="layer_out", layer=layer_index, tensor=branch, role="branch")
        _capture(state, boundary="layer_out", layer=layer_index, tensor=residual, role="residual")
        state.current_layer = None

    return hook


def _input_norm_hook(state: HookCaptureState, layer_index: int):
    def hook(_module, _args: tuple, _kwargs: dict, output: Any) -> None:
        if layer_index == 0:
            if not isinstance(output, torch.Tensor):
                raise RuntimeError("layer0 input RMSNorm must return Tensor")
            _capture(state, boundary="input_norm", layer=layer_index, tensor=output, role="normalized")
            return
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError("layer1+ input RMSNorm must return tuple[2]")
        normalized, residual = output
        if not isinstance(normalized, torch.Tensor) or not isinstance(residual, torch.Tensor):
            raise RuntimeError("input RMSNorm tuple must contain Tensors")
        _capture(state, boundary="input_norm", layer=layer_index, tensor=normalized, role="normalized")
        _capture(state, boundary="input_norm", layer=layer_index, tensor=residual, role="updated_residual")

    return hook


def _tensor_output_hook(state: HookCaptureState, layer_index: int, boundary: str, role: str):
    def hook(_module, _args: tuple, _kwargs: dict, output: Any) -> None:
        if not isinstance(output, torch.Tensor):
            raise RuntimeError(f"{boundary} must return Tensor, got {type(output)}")
        _capture(state, boundary=boundary, layer=layer_index, tensor=output, role=role)

    return hook


def _tuple_first_hook(state: HookCaptureState, layer_index: int, boundary: str, role: str):
    def hook(_module, _args: tuple, _kwargs: dict, output: Any) -> None:
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError(f"{boundary} must return tuple[2], got {type(output)}")
        tensor, bias = output
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(f"{boundary} first output must be Tensor")
        if bias is not None and not isinstance(bias, torch.Tensor):
            raise RuntimeError(f"{boundary} bias output must be Tensor or None")
        _capture(state, boundary=boundary, layer=layer_index, tensor=tensor, role=role)

    return hook


def _post_attn_norm_hook(state: HookCaptureState, layer_index: int):
    def hook(_module, _args: tuple, _kwargs: dict, output: Any) -> None:
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError("post-attention RMSNorm must return tuple[2]")
        normalized, residual = output
        if not isinstance(normalized, torch.Tensor) or not isinstance(residual, torch.Tensor):
            raise RuntimeError("post-attention RMSNorm tuple must contain Tensors")
        _capture(state, boundary="post_attn_norm", layer=layer_index, tensor=normalized, role="normalized")
        _capture(state, boundary="post_attn_norm", layer=layer_index, tensor=residual, role="updated_residual")

    return hook


def _final_norm_hook(state: HookCaptureState):
    def hook(_module, _args: tuple, _kwargs: dict, output: Any) -> None:
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError("final RMSNorm must return tuple[2]")
        final_hidden, residual = output
        if not isinstance(final_hidden, torch.Tensor) or not isinstance(residual, torch.Tensor):
            raise RuntimeError("final RMSNorm tuple must contain Tensors")
        _capture(state, boundary="final_norm", layer=None, tensor=final_hidden, role="normalized")
        _capture(state, boundary="final_norm", layer=None, tensor=residual, role="updated_residual")

    return hook


def install_hooks(model, variant: str, output_root: str, capture_level: str = "core") -> dict:
    if capture_level not in {"core", "core_qkv", "feature_scan"}:
        raise ValueError(f"unknown capture_level={capture_level}")
    if hasattr(model, _STATE_ATTR) or hasattr(model, _HANDLES_ATTR):
        raise RuntimeError("real-vLLM hooks are already installed")
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise RuntimeError(f"unexpected model type for Qwen3-MoE hooks: {type(model)}")
    layers = list(model.model.layers)
    if len(layers) != 48:
        raise RuntimeError(f"expected 48 Qwen3-MoE layers, got {len(layers)}")

    state = HookCaptureState(
        variant=str(variant),
        rank=int(get_tensor_model_parallel_rank()),
        world_size=int(get_tensor_model_parallel_world_size()),
        output_root=str(output_root),
        capture_level=capture_level,
    )
    handles = []
    # Qwen3MoeModel is decorated with @support_torch_compile, which replaces
    # __call__ to invoke forward() directly and bypasses nn.Module hooks.
    # Position tracking must therefore hang on the outer Qwen3MoeForCausalLM,
    # whose Module.__call__ still runs forward_pre_hooks.
    handles.append(model.register_forward_pre_hook(_model_pre_hook(model, state), with_kwargs=True))
    for layer_index, layer in enumerate(layers):
        for attr in ("input_layernorm", "self_attn", "post_attention_layernorm", "mlp"):
            if not hasattr(layer, attr):
                raise RuntimeError(f"layer {layer_index} missing {attr}")
        if not hasattr(layer.self_attn, "attn") or not hasattr(layer.self_attn, "o_proj"):
            raise RuntimeError(f"layer {layer_index} attention modules incomplete")
        if not hasattr(layer.mlp, "gate"):
            raise RuntimeError(f"layer {layer_index} expected sparse MoE gate")
        if capture_level == "feature_scan":
            handles.append(layer.input_layernorm.register_forward_hook(
                _input_norm_hook(state, layer_index), with_kwargs=True))
            handles.append(layer.post_attention_layernorm.register_forward_hook(
                _post_attn_norm_hook(state, layer_index), with_kwargs=True))
            continue
        if capture_level == "core_qkv":
            handles.append(layer.self_attn.qkv_proj.register_forward_hook(
                _tuple_first_hook(state, layer_index, "qkv_proj", "rank_local"),
                with_kwargs=True))
        handles.append(layer.register_forward_pre_hook(_layer_pre_hook(state, layer_index), with_kwargs=True))
        handles.append(layer.register_forward_hook(_layer_post_hook(state, layer_index), with_kwargs=True))
        handles.append(layer.input_layernorm.register_forward_hook(_input_norm_hook(state, layer_index), with_kwargs=True))
        handles.append(
            layer.self_attn.attn.register_forward_hook(
                _tensor_output_hook(state, layer_index, "attention_core", "rank_local"),
                with_kwargs=True,
            )
        )
        handles.append(
            layer.self_attn.o_proj.register_forward_hook(
                _tuple_first_hook(state, layer_index, "o_proj", "tp_reduced"),
                with_kwargs=True,
            )
        )
        handles.append(
            layer.post_attention_layernorm.register_forward_hook(
                _post_attn_norm_hook(state, layer_index), with_kwargs=True
            )
        )
        handles.append(
            layer.mlp.gate.register_forward_hook(
                _tuple_first_hook(state, layer_index, "router_logits", "logits"),
                with_kwargs=True,
            )
        )
        handles.append(
            layer.mlp.register_forward_hook(
                _tensor_output_hook(state, layer_index, "moe_out", "tp_reduced"),
                with_kwargs=True,
            )
        )
    if capture_level != "feature_scan":
        handles.append(model.model.norm.register_forward_hook(_final_norm_hook(state), with_kwargs=True))
    setattr(model, _STATE_ATTR, state)
    setattr(model, _HANDLES_ATTR, handles)
    return {
        "rank": state.rank,
        "world_size": state.world_size,
        "num_layers": len(layers),
        "num_handles": len(handles),
        "position_hook_module": type(model).__name__,
        "capture_level": capture_level,
    }


def begin_sample(model, sample_key: str, prompt_len: int, probe_abs_to_decode: dict[int, int]) -> dict:
    state = _state(model)
    state.begin_sample(sample_key, prompt_len, probe_abs_to_decode)
    return {"rank": state.rank, "sample_key": state.sample_key, "num_probes": len(probe_abs_to_decode)}


def flush_sample(model) -> dict:
    state = _state(model)
    path = state.output_path
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "variant": state.variant,
        "sample_key": state.sample_key,
        "rank": state.rank,
        "world_size": state.world_size,
        "prompt_len": state.prompt_len,
        "capture_level": state.capture_level,
        "probe_abs_to_decode": dict(state.probe_abs_to_decode),
        "fire_counts": dict(state.fire_counts),
        "records": state.records,
    }
    torch.save(payload, path)
    result = {
        "rank": state.rank,
        "sample_key": state.sample_key,
        "path": str(path),
        "num_records": len(state.records),
        "fire_counts": dict(state.fire_counts),
        "prefill_done": state.prefill_done,
        "next_decode_index": state.next_decode_index,
    }
    state.clear_sample()
    return result


def inspect_hook_state(model) -> dict:
    state = _state(model)
    handles = getattr(model, _HANDLES_ATTR)
    return {
        "rank": state.rank,
        "world_size": state.world_size,
        "variant": state.variant,
        "num_handles": len(handles),
        "sample_key": state.sample_key,
        "num_records": len(state.records),
        "fire_counts": dict(state.fire_counts),
    }


def remove_hooks(model) -> dict:
    state = _state(model)
    if state.sample_key is not None:
        raise RuntimeError(f"cannot remove hooks while sample is active: {state.sample_key}")
    handles = getattr(model, _HANDLES_ATTR)
    for handle in handles:
        handle.remove()
    result = {"rank": state.rank, "removed": len(handles)}
    delattr(model, _HANDLES_ATTR)
    delattr(model, _STATE_ATTR)
    return result
