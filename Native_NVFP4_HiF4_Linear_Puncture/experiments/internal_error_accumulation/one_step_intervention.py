"""One-step causal interventions on real vLLM forward (no semantic replay)."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch

from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.format_conversion.production_puncture import (
    frozen_router,
)

from .qkv_slices import apply_qkv_slice_repair, runtime_qkv_slice_bounds
from .residual_ledger import reject_layer0_state_reset


SESSION_ATTR = "_hif4_iea_intervention_session"
HANDLES_ATTR = "_hif4_iea_intervention_handles"


class ForceClearSampleOp:
    """Clear active hook sample without requiring a successful flush write.

    Intervention loops must always release BeginSampleOp ownership before the
    next BeginSampleOp / remove_hooks; otherwise workers raise
    ``sample already active`` / ``cannot remove hooks while sample is active``.
    """

    def __call__(self, model) -> dict:
        from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.worker_hooks import (
            _state,
        )

        state = _state(model)
        key = state.sample_key
        if key is not None:
            state.clear_sample()
        if hasattr(model, SESSION_ATTR):
            for handle in getattr(model, HANDLES_ATTR, []):
                handle.remove()
            delattr(model, HANDLES_ATTR)
            delattr(model, SESSION_ATTR)
        if hasattr(model, "_hif4_iea_positions"):
            delattr(model, "_hif4_iea_positions")
        return {"rank": state.rank, "cleared_sample_key": key}


class BranchRepairSession:
    """Replace o_proj and/or moe_out token row at a single decode predictor."""

    def __init__(
        self,
        *,
        sample_key: str,
        layer: int,
        decode_index: int,
        abs_position: int,
        repair_attn: torch.Tensor | None,
        repair_moe: torch.Tensor | None,
        mode: str,
    ) -> None:
        if mode not in {"repair", "noop"}:
            raise ValueError(mode)
        self.sample_key = sample_key
        self.layer = int(layer)
        self.decode_index = int(decode_index)
        self.abs_position = int(abs_position)
        self.repair_attn = None if repair_attn is None else repair_attn.detach().cpu().clone()
        self.repair_moe = None if repair_moe is None else repair_moe.detach().cpu().clone()
        self.mode = mode
        self.attn_hits = 0
        self.moe_hits = 0
        self.closed = False

    def _maybe_replace(self, boundary: str, tensor: torch.Tensor, positions: list[int]) -> torch.Tensor:
        if self.closed:
            raise RuntimeError("intervention session already closed")
        if tensor.ndim < 1:
            raise RuntimeError("branch tensor rank")
        # tensor is the module output before row extract; worker hooks capture by row.
        # Here we patch the full activation then let production continue.
        target = self.repair_attn if boundary == "o_proj" else self.repair_moe
        if target is None:
            return tensor
        # Decode path is typically [1, H] or [num_tokens, H]; find matching abs position row.
        if tensor.shape[0] != len(positions):
            # Some modules return already-reduced single-token vectors.
            if tensor.ndim == 1 or tensor.shape[0] == 1:
                out = tensor.clone()
                src = target.to(device=out.device, dtype=out.dtype)
                if out.ndim == 1:
                    if out.numel() != src.numel():
                        raise RuntimeError("branch repair numel mismatch")
                    out.copy_(src.reshape_as(out))
                else:
                    if out.shape[-1] != src.numel():
                        raise RuntimeError("branch repair hidden mismatch")
                    out[0].copy_(src.reshape_as(out[0]))
                if boundary == "o_proj":
                    self.attn_hits += 1
                else:
                    self.moe_hits += 1
                return out
            raise RuntimeError("branch/position mapping mismatch")
        out = tensor.clone()
        for row, pos in enumerate(positions):
            if pos != self.abs_position:
                continue
            src = target.to(device=out.device, dtype=out.dtype)
            if out[row].numel() != src.numel():
                raise RuntimeError("branch repair numel mismatch")
            out[row].copy_(src.reshape_as(out[row]))
            if boundary == "o_proj":
                self.attn_hits += 1
            else:
                self.moe_hits += 1
        return out

    def finish(self) -> dict:
        self.closed = True
        need_attn = self.repair_attn is not None
        need_moe = self.repair_moe is not None
        if need_attn and self.attn_hits != 1:
            raise RuntimeError(f"Attention repair expected 1 hit, got {self.attn_hits}")
        if need_moe and self.moe_hits != 1:
            raise RuntimeError(f"MoE repair expected 1 hit, got {self.moe_hits}")
        return {
            "mode": self.mode,
            "layer": self.layer,
            "attn_hits": self.attn_hits,
            "moe_hits": self.moe_hits,
        }


class InputNormTupleResetSession:
    """Replace layer l=1..47 input_norm (normalized, updated_residual) at target abs position."""

    def __init__(
        self,
        *,
        sample_key: str,
        layer: int,
        abs_position: int,
        normalized: torch.Tensor,
        updated_residual: torch.Tensor,
        mode: str,
    ) -> None:
        reject_layer0_state_reset(layer)
        if mode not in {"reset", "noop"}:
            raise ValueError(mode)
        self.sample_key = sample_key
        self.layer = int(layer)
        self.abs_position = int(abs_position)
        self.normalized = normalized.detach().cpu().clone()
        self.updated_residual = updated_residual.detach().cpu().clone()
        self.mode = mode
        self.hits = 0
        self.closed = False

    def apply(self, output: Any, positions: list[int]) -> Any:
        if self.closed:
            raise RuntimeError("session closed")
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError("input_norm must return (normalized, updated_residual)")
        norm, resid = output
        if norm.shape[0] != len(positions) or resid.shape[0] != len(positions):
            raise RuntimeError("input_norm row/position mismatch")
        out_n, out_r = norm.clone(), resid.clone()
        for row, pos in enumerate(positions):
            if pos != self.abs_position:
                continue
            src_n = self.normalized.to(device=out_n.device, dtype=out_n.dtype)
            src_r = self.updated_residual.to(device=out_r.device, dtype=out_r.dtype)
            out_n[row].copy_(src_n.reshape_as(out_n[row]))
            out_r[row].copy_(src_r.reshape_as(out_r[row]))
            self.hits += 1
        return out_n, out_r

    def finish(self) -> dict:
        self.closed = True
        if self.hits != 1:
            raise RuntimeError(f"input_norm reset expected 1 hit, got {self.hits}")
        return {"mode": self.mode, "layer": self.layer, "hits": self.hits}


class QKVSliceRepairSession:
    def __init__(
        self,
        *,
        sample_key: str,
        layer: int,
        abs_position: int,
        which: str,
        source_qkv: torch.Tensor,
        bounds: dict[str, tuple[int, int]],
        mode: str,
    ) -> None:
        if mode not in {"repair", "noop"}:
            raise ValueError(mode)
        self.sample_key = sample_key
        self.layer = int(layer)
        self.abs_position = int(abs_position)
        self.which = which
        self.source_qkv = source_qkv.detach().cpu().clone()
        self.bounds = bounds
        self.mode = mode
        self.hits = 0
        self.closed = False

    def apply(self, output: Any, positions: list[int]) -> Any:
        if self.closed:
            raise RuntimeError("session closed")
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError("qkv_proj must return (tensor, bias)")
        tensor, bias = output
        if tensor.shape[0] != len(positions):
            if tensor.ndim >= 1 and (tensor.ndim == 1 or tensor.shape[0] == 1):
                src = self.source_qkv.to(device=tensor.device, dtype=tensor.dtype)
                base = tensor.clone()
                if base.ndim == 1:
                    repaired = apply_qkv_slice_repair(base, src.reshape_as(base), which=self.which, bounds=self.bounds)
                else:
                    repaired = base.clone()
                    repaired[0] = apply_qkv_slice_repair(
                        base[0], src.reshape_as(base[0]), which=self.which, bounds=self.bounds
                    )
                self.hits += 1
                return repaired, bias
            raise RuntimeError("qkv/position mismatch")
        out = tensor.clone()
        for row, pos in enumerate(positions):
            if pos != self.abs_position:
                continue
            src = self.source_qkv.to(device=out.device, dtype=out.dtype)
            out[row] = apply_qkv_slice_repair(out[row], src.reshape_as(out[row]), which=self.which, bounds=self.bounds)
            self.hits += 1
        return out, bias

    def finish(self) -> dict:
        self.closed = True
        if self.hits != 1:
            raise RuntimeError(f"QKV slice repair expected 1 hit, got {self.hits}")
        return {"mode": self.mode, "layer": self.layer, "which": self.which, "hits": self.hits}


def _get_positions(model) -> list[int]:
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.worker_hooks import (
        _get_positions as hook_get_positions,
        _state,
    )

    state = _state(model)
    # Positions are set on each forward via model pre-hook into state; reconstruct from last capture machinery.
    # Prefer reading from the temporary attribute set by our positions hook.
    positions = getattr(model, "_hif4_iea_positions", None)
    if positions is None:
        raise RuntimeError("intervention positions not set")
    return [int(x) for x in positions]


class InstallCausalInterventionOp:
    """Install temporary causal hooks; must run after BeginSampleOp."""

    def __init__(self, kind: str, payload: dict) -> None:
        self.kind = kind
        self.payload = payload

    def __call__(self, model) -> dict:
        from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.worker_hooks import (
            _state,
        )

        state = _state(model)
        if hasattr(model, SESSION_ATTR):
            raise RuntimeError("previous causal intervention not flushed")
        layer_idx = int(self.payload["layer"])
        layer = model.model.layers[layer_idx]
        handles = []

        def positions_hook(_module, args, kwargs):
            from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.worker_hooks import (
                _get_positions as gp,
            )

            pos = [int(x) for x in gp(args, kwargs).detach().cpu().reshape(-1).tolist()]
            setattr(model, "_hif4_iea_positions", pos)

        handles.append(model.register_forward_pre_hook(positions_hook, with_kwargs=True, prepend=True))

        if self.kind in {"whole_layer", "attn_only", "moe_only"}:
            repair_attn = self.payload.get("repair_attn")
            repair_moe = self.payload.get("repair_moe")
            if self.kind == "attn_only":
                repair_moe = None
            if self.kind == "moe_only":
                repair_attn = None
            session = BranchRepairSession(
                sample_key=self.payload["sample_key"],
                layer=layer_idx,
                decode_index=int(self.payload["decode_index"]),
                abs_position=int(self.payload["abs_position"]),
                repair_attn=repair_attn,
                repair_moe=repair_moe,
                mode=self.payload.get("mode", "repair"),
            )

            def o_hook(_m, _a, _k, output):
                tensor, bias = output
                patched = session._maybe_replace("o_proj", tensor, getattr(model, "_hif4_iea_positions"))
                return patched, bias

            def moe_hook(_m, _a, _k, output):
                return session._maybe_replace("moe_out", output, getattr(model, "_hif4_iea_positions"))

            if repair_attn is not None:
                handles.append(layer.self_attn.o_proj.register_forward_hook(o_hook, with_kwargs=True, prepend=True))
            if repair_moe is not None:
                handles.append(layer.mlp.register_forward_hook(moe_hook, with_kwargs=True, prepend=True))
            setattr(model, SESSION_ATTR, session)
        elif self.kind == "input_norm_reset":
            session = InputNormTupleResetSession(
                sample_key=self.payload["sample_key"],
                layer=layer_idx,
                abs_position=int(self.payload["abs_position"]),
                normalized=self.payload["normalized"],
                updated_residual=self.payload["updated_residual"],
                mode=self.payload.get("mode", "reset"),
            )

            def in_hook(_m, _a, _k, output):
                return session.apply(output, getattr(model, "_hif4_iea_positions"))

            handles.append(layer.input_layernorm.register_forward_hook(in_hook, with_kwargs=True, prepend=True))
            setattr(model, SESSION_ATTR, session)
        elif self.kind == "qkv_slice":
            bounds = runtime_qkv_slice_bounds(layer.self_attn)
            # qkv_proj is rank-local under TP: each worker must use its own capture.
            if "source_qkv_by_rank" in self.payload:
                by_rank = self.payload["source_qkv_by_rank"]
                if not isinstance(by_rank, dict) or int(state.rank) not in by_rank and str(state.rank) not in by_rank:
                    raise RuntimeError(
                        f"qkv_slice missing source for tp_rank={state.rank}; "
                        f"keys={list(by_rank.keys()) if isinstance(by_rank, dict) else type(by_rank)}"
                    )
                source_qkv = by_rank[int(state.rank)] if int(state.rank) in by_rank else by_rank[str(state.rank)]
            elif "source_qkv" in self.payload:
                # Legacy single-tensor path is only valid for world_size=1.
                if int(getattr(state, "world_size", 1) or 1) != 1:
                    raise RuntimeError(
                        "qkv_slice requires source_qkv_by_rank under TP>1; "
                        "single source_qkv would break captured-E1 no-op identity"
                    )
                source_qkv = self.payload["source_qkv"]
            else:
                raise RuntimeError("qkv_slice payload requires source_qkv_by_rank")
            session = QKVSliceRepairSession(
                sample_key=self.payload["sample_key"],
                layer=layer_idx,
                abs_position=int(self.payload["abs_position"]),
                which=self.payload["which"],
                source_qkv=source_qkv,
                bounds=bounds,
                mode=self.payload.get("mode", "repair"),
            )

            def qkv_hook(_m, _a, _k, output):
                return session.apply(output, getattr(model, "_hif4_iea_positions"))

            handles.append(layer.self_attn.qkv_proj.register_forward_hook(qkv_hook, with_kwargs=True, prepend=True))
            setattr(model, SESSION_ATTR, session)
        elif self.kind == "router_freeze":
            source = self.payload["router_logits"].detach().cpu().clone()
            abs_position = int(self.payload["abs_position"])
            calls = {"count": 0, "hits": 0}

            def gate_hook(_module, _args, output):
                if not isinstance(output, tuple) or len(output) != 2:
                    raise RuntimeError("router must return (logits, bias)")
                original, bias = output
                if not isinstance(original, torch.Tensor):
                    raise RuntimeError("router logits must be Tensor")
                positions = getattr(model, "_hif4_iea_positions", None)
                if positions is None:
                    raise RuntimeError("router freeze positions not set")
                src = source.to(device=original.device, dtype=original.dtype)
                calls["count"] += 1
                # Full-forward gate may see multi-token batches; only patch the
                # target predictor row. Captured E0 logits are a single-token vector.
                if original.ndim == 1:
                    if len(positions) != 1 or positions[0] != abs_position:
                        return original, bias
                    if original.numel() != src.numel():
                        raise RuntimeError(
                            f"frozen router numel mismatch: live={original.numel()} src={src.numel()}"
                        )
                    calls["hits"] += 1
                    return src.reshape_as(original).clone(), bias
                if original.ndim < 2 or original.shape[0] != len(positions):
                    raise RuntimeError(
                        f"router/position mismatch: out={tuple(original.shape)} npos={len(positions)}"
                    )
                out = original.clone()
                for row, pos in enumerate(positions):
                    if pos != abs_position:
                        continue
                    if out[row].numel() != src.numel():
                        raise RuntimeError(
                            f"frozen router row numel mismatch: live={out[row].numel()} src={src.numel()}"
                        )
                    out[row].copy_(src.reshape_as(out[row]))
                    calls["hits"] += 1
                return out, bias

            handles.append(layer.mlp.gate.register_forward_hook(gate_hook))
            setattr(
                model,
                SESSION_ATTR,
                {
                    "kind": "router_freeze",
                    "calls": calls,
                    "layer": layer_idx,
                    "abs_position": abs_position,
                },
            )
        else:
            for h in handles:
                h.remove()
            raise ValueError(f"unknown intervention kind={self.kind}")

        setattr(model, HANDLES_ATTR, handles)
        return {"rank": state.rank, "kind": self.kind, "layer": layer_idx, "num_handles": len(handles)}


class FlushCausalInterventionOp:
    def __call__(self, model) -> dict:
        session = getattr(model, SESSION_ATTR)
        try:
            if isinstance(session, dict):
                if session.get("kind") == "router_freeze":
                    if session["calls"]["hits"] != 1:
                        raise RuntimeError(
                            f"router freeze expected 1 target hit, got hits={session['calls']['hits']} "
                            f"fires={session['calls']['count']}"
                        )
                    payload = {
                        "kind": "router_freeze",
                        "calls": session["calls"]["count"],
                        "hits": session["calls"]["hits"],
                        "layer": session["layer"],
                        "abs_position": session["abs_position"],
                    }
                else:
                    payload = dict(session)
            else:
                payload = session.finish()
            return payload
        finally:
            for handle in getattr(model, HANDLES_ATTR):
                handle.remove()
            delattr(model, HANDLES_ATTR)
            delattr(model, SESSION_ATTR)
            if hasattr(model, "_hif4_iea_positions"):
                delattr(model, "_hif4_iea_positions")


@contextmanager
def temporary_frozen_router(gate, logits: torch.Tensor):
    with frozen_router(gate, logits) as calls:
        yield calls
