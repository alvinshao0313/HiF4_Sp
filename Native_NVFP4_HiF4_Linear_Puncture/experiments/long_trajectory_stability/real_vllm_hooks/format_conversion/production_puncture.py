"""Same-input calls of loaded production modules, with measured identity gates.

No decoder, attention, expert loop, quantizer, or tensor-parallel emulation is
implemented here. Invoke ProductionPunctureOp with the existing LLM.apply_model
so both production TP ranks enter every collective in the same order.
"""
from __future__ import annotations

from contextlib import contextmanager
from itertools import combinations
from pathlib import Path

import torch

OPERATOR_BOUNDARIES = {
    "qkv": (("input_norm", "normalized"), ("qkv_proj", "rank_local")),
    "o_proj": (("attention_core", "rank_local"), ("o_proj", "tp_reduced")),
    "moe": (("post_attn_norm", "normalized"), ("moe_out", "tp_reduced")),
}


def puncture_key(sample_key: str, decode_index: int, layer: int, operator: str, rank: int) -> tuple:
    if operator not in OPERATOR_BOUNDARIES or rank not in (0, 1) or not 0 <= layer < 48:
        raise ValueError("invalid production puncture operator/layer/TP rank")
    if decode_index <= 0:
        raise ValueError("production row puncture requires decode_index > 0; prefill needs its complete original batch")
    return str(sample_key), int(decode_index), int(layer), operator, int(rank)


def _vectors(*values: torch.Tensor) -> list[torch.Tensor]:
    if not values or any(v.shape != values[0].shape for v in values):
        raise ValueError("decomposition tensors must have identical shapes")
    if any(not v.is_floating_point() or not torch.isfinite(v).all() for v in values):
        raise ValueError("decomposition tensors must be finite floating point")
    return [v.detach().to(device="cpu", dtype=torch.float64) for v in values]


def _norm(x: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(x))


def closure_metrics(total: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> dict:
    total, left, right = _vectors(total, left, right)
    remainder = total - (left + right)
    scale = max(float(total.abs().max()), float(left.abs().max()), float(right.abs().max()))
    # This tolerance is solely FP64 arithmetic roundoff in offline subtraction,
    # never a tolerance for a production identity or a kernel discrepancy.
    tolerance = 8 * torch.finfo(torch.float64).eps * scale
    maximum = float(remainder.abs().max())
    passed = maximum <= tolerance
    result = {"closure_max_abs": maximum, "closure_rel_l2": _norm(remainder) / max(_norm(total), 1e-300),
              "closure_roundoff_bound": tolerance, "closure_pass": passed}
    if not passed:
        raise RuntimeError(f"decomposition closure failed: {result}")
    return result


def error_decomposition(y0: torch.Tensor, y1: torch.Tensor, y1_fix: torch.Tensor) -> dict:
    """e=y1-y0, q=y1_fix-y0, p=y1-y1_fix; output metrics include norms."""
    y0, y1, fixed = _vectors(y0, y1, y1_fix)
    e, q, p = y1 - y0, fixed - y0, y1 - fixed
    denominator = _norm(y0)
    result = closure_metrics(e, q, p)
    for name, value in (("e", e), ("q", q), ("p", p)):
        norm = _norm(value)
        result[f"{name}_l2"] = norm
        result[f"{name}_rel_l2"] = norm / denominator if denominator else None
    qnorm, pnorm = _norm(q), _norm(p)
    result.update({"y0_l2": denominator, "cos_q_p": float((q * p).sum()) / (qnorm * pnorm) if qnorm and pnorm else None})
    return result


def router_decomposition(y0: torch.Tensor, y1_fix: torch.Tensor, y1_route0: torch.Tensor) -> dict:
    """Split local q only: all three outputs use the SAME captured E0 input."""
    y0, fixed, frozen = _vectors(y0, y1_fix, y1_route0)
    q, expert, router = fixed - y0, frozen - y0, fixed - frozen
    denominator = _norm(y0)
    result = {f"router_{k}": v for k, v in closure_metrics(q, expert, router).items()}
    for name, value in (("q_expert", expert), ("q_router", router)):
        result[f"{name}_l2"] = _norm(value)
        result[f"{name}_rel_l2"] = _norm(value) / denominator if denominator else None
    return result


def identity_metrics(actual: torch.Tensor, repeats: list[torch.Tensor]) -> dict:
    """Only the measured repeatability envelope can permit non-exact identity."""
    if len(repeats) < 3:
        raise ValueError("identity requires at least three independently executed production repeats")
    actual, *copies = _vectors(actual, *repeats)
    deltas = [a - b for a, b in combinations(copies, 2)]
    noise_max = max(float(d.abs().max()) for d in deltas)
    noise_l2 = max(_norm(d) for d in deltas)
    deviations = [copy - actual for copy in copies]
    observed_max = max(float(d.abs().max()) for d in deviations)
    observed_l2 = max(_norm(d) for d in deviations)
    return {"identity_pass": observed_max <= noise_max and observed_l2 <= noise_l2,
            "identity_exact": observed_max == 0.0, "identity_max_abs": observed_max,
            "identity_l2": observed_l2, "repeatability_max_abs": noise_max,
            "repeatability_l2": noise_l2, "repeat_count": len(copies)}


@contextmanager
def frozen_router(gate: torch.nn.Module, logits: torch.Tensor):
    """Temporarily replace only the production gate output; remove on exception."""
    calls = {"count": 0}
    def hook(_module, _args, output):
        if not isinstance(output, tuple) or len(output) != 2 or not isinstance(output[0], torch.Tensor):
            raise RuntimeError("production router must return (logits, bias)")
        original, bias = output
        if original.shape != logits.shape or original.dtype != logits.dtype or original.device != logits.device:
            raise RuntimeError("frozen-router logits shape/dtype/device mismatch")
        calls["count"] += 1
        return logits.clone(), bias
    handle = gate.register_forward_hook(hook)
    try:
        yield calls
        if calls["count"] == 0:
            raise RuntimeError("frozen-router hook never fired")
    finally:
        handle.remove()


def _capture_index(payload: dict) -> dict:
    result = {}
    for row in payload["records"]:
        key = (row["sample_key"], int(row["decode_index"]), row["layer"], row["boundary"], row["role"])
        if key in result:
            raise RuntimeError(f"duplicate actual-path capture key: {key}")
        result[key] = row
    return result


def _row(index: dict, sample: str, decode: int, layer: int, boundary_role: tuple[str, str]) -> torch.Tensor:
    record = index[(sample, decode, layer, *boundary_role)]
    value = record["tensor"]
    if value.ndim != 1 or tuple(record["original_shape"]) != (1, value.numel()):
        raise RuntimeError("PUNCTURE_SHAPE_BLOCKED: captured row is not a one-token original execution")
    return value.unsqueeze(0)


def _production_call(model, layer_index: int, operator: str, x: torch.Tensor, route: torch.Tensor | None = None) -> torch.Tensor:
    from vllm.config import set_current_vllm_config
    from vllm.forward_context import get_forward_context, set_forward_context
    from vllm.model_executor.layers.quantization import hif4_runtime

    layer = model.model.layers[layer_index]
    experts = layer.mlp.experts
    config = experts.vllm_config
    modules = {"qkv": layer.self_attn.qkv_proj, "o_proj": layer.self_attn.o_proj, "moe": layer.mlp}
    device = next(modules[operator].parameters()).device
    x = x.to(device=device).clone()
    # Online DIAG MoE obtains its original layer identity from this existing
    # runtime metadata, normally set by QKV. Restore it after the local call.
    layer_state = hif4_runtime._CURRENT_LAYER_BY_DEVICE
    key = str(device)
    had_layer = key in layer_state
    previous_layer = layer_state.get(key)
    try:
        hif4_runtime.set_current_layer(device, layer_index)
        with torch.inference_mode(), set_current_vllm_config(config), set_forward_context(None, config, num_tokens=1):
            context = get_forward_context()
            if context.all_moe_layers is not None:
                context.moe_layer_index = context.all_moe_layers.index(experts.layer_name)
            if route is None:
                output = modules[operator](x)
            else:
                if operator != "moe":
                    raise ValueError("frozen router is defined only for production MoE")
                with frozen_router(layer.mlp.gate, route.to(device=device)):
                    output = layer.mlp(x)
            if operator != "moe":
                if not isinstance(output, tuple) or len(output) != 2:
                    raise RuntimeError("production projection did not return (output, bias)")
                output = output[0]
            if not isinstance(output, torch.Tensor):
                raise RuntimeError("production output is not a Tensor")
            return output.detach().cpu().clone()
    finally:
        if had_layer:
            layer_state[key] = previous_layer
        else:
            layer_state.pop(key, None)


def _both_ranks_pass(model, local_pass: bool) -> bool:
    """Keep rank-dependent identity findings from changing collective call order."""
    from vllm.distributed import tensor_model_parallel_all_reduce
    device = next(model.parameters()).device
    vote = torch.tensor([int(local_pass)], dtype=torch.int32, device=device)
    return int(tensor_model_parallel_all_reduce(vote).item()) == 2


class ProductionPunctureOp:
    """Picklable apply_model payload. Baseline identity artifacts gate variants.

    Requests: sample_key, decode_index (>0), layer, operator. Capture roots
    contain <sample_key>/rank<r>.pt; baseline outputs use the same layout.
    All capture manifests must be validated as forced isolated E0 history by
    the parent runner before passing canonical_history_verified=True.
    """
    def __init__(self, requests: list[dict], reference_root: str, actual_root: str,
                 output_root: str, variant: str, *, canonical_history_verified: bool,
                 baseline_puncture_root: str | None = None):
        if variant not in {"E0", "E1", "E2", "E3"} or not canonical_history_verified:
            raise ValueError("puncture requires permitted variant and verified canonical E0 history")
        if variant != "E0" and baseline_puncture_root is None:
            raise ValueError("variant puncture requires E0 identity artifacts")
        self.requests = requests
        self.reference_root, self.actual_root = reference_root, actual_root
        self.output_root, self.variant = output_root, variant
        self.baseline_puncture_root = baseline_puncture_root
        for req in requests:
            puncture_key(req["sample_key"], int(req["decode_index"]), int(req["layer"]), req["operator"], 0)

    def __call__(self, model) -> dict:
        from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
        rank = int(get_tensor_model_parallel_rank())
        if get_tensor_model_parallel_world_size() != 2:
            raise RuntimeError("production puncture requires TP2")
        state = getattr(model, "_hif4_longtraj_diag_state", None)
        if state is not None and state.sample_key is not None:
            raise RuntimeError("production puncture requires flushed inactive capture hooks")
        results, blocked = [], []
        samples = sorted({req["sample_key"] for req in self.requests})
        for sample in samples:
            reference = torch.load(Path(self.reference_root) / sample / f"rank{rank}.pt", map_location="cpu", weights_only=False)
            actual = torch.load(Path(self.actual_root) / sample / f"rank{rank}.pt", map_location="cpu", weights_only=False)
            if reference["variant"] != "E0" or actual["variant"] != self.variant or reference["rank"] != rank or actual["rank"] != rank:
                raise RuntimeError("capture variant/rank mismatch")
            index0, index1 = _capture_index(reference), _capture_index(actual)
            baseline = {}
            if self.baseline_puncture_root:
                saved = torch.load(Path(self.baseline_puncture_root) / sample / f"rank{rank}.pt", map_location="cpu", weights_only=False)
                baseline = {tuple(r["key"]): r for r in saved["records"]}
            records = []
            for req in [r for r in self.requests if r["sample_key"] == sample]:
                decode, layer, operator = int(req["decode_index"]), int(req["layer"]), req["operator"]
                key = puncture_key(sample, decode, layer, operator, rank)
                input_boundary, output_boundary = OPERATOR_BOUNDARIES[operator]
                x0, y0 = _row(index0, sample, decode, layer, input_boundary), _row(index0, sample, decode, layer, output_boundary)
                x1, y1 = _row(index1, sample, decode, layer, input_boundary), _row(index1, sample, decode, layer, output_boundary)
                if self.variant != "E0" and not _both_ranks_pass(model, baseline[key]["identity"]["both_tp_ranks_pass"]):
                    raise RuntimeError(f"{operator.upper()}_PUNCTURE_BLOCKED: baseline identity failed")
                repeats = [_production_call(model, layer, operator, x1) for _ in range(3)]
                identity = identity_metrics(y1, repeats)
                identity["both_tp_ranks_pass"] = _both_ranks_pass(model, identity["identity_pass"])
                if not identity["both_tp_ranks_pass"]:
                    records.append({"key": key, **req, "variant": self.variant, "tp_rank": rank, "identity": identity,
                                    "status": f"{operator.upper()}_PUNCTURE_BLOCKED"})
                    blocked.append(key)
                    continue
                fixed = repeats[0] if self.variant == "E0" else _production_call(model, layer, operator, x0)
                record = {"key": key, **req, "variant": self.variant, "tp_rank": rank, "identity": identity,
                          "y0": y0, "y1": y1, "y1_fix": fixed, "status": "PASS" if identity["identity_pass"] else f"{operator.upper()}_PUNCTURE_BLOCKED"}
                if identity["identity_pass"]:
                    record["decomposition"] = error_decomposition(y0, y1, fixed)
                if operator == "moe":
                    r0 = _row(index0, sample, decode, layer, ("router_logits", "logits"))
                    route_allowed = self.variant == "E0" or baseline[key].get("frozen_router_identity", {}).get("both_tp_ranks_pass", False)
                    if route_allowed:
                        frozen = [_production_call(model, layer, operator, x0, r0) for _ in range(3)]
                        record["y1_route0"] = frozen[0]
                        if self.variant == "E0":
                            # A frozen route must match the natural output within
                            # the already measured natural-repeat envelope.
                            noise = identity_metrics(repeats[0], repeats)
                            differences = [f.to(torch.float64) - repeats[0].to(torch.float64) for f in frozen]
                            maximum = max(float(d.abs().max()) for d in differences)
                            length = max(_norm(d) for d in differences)
                            record["frozen_router_identity"] = {**noise, "identity_max_abs": maximum, "identity_l2": length,
                                "identity_exact": maximum == 0, "identity_pass": maximum <= noise["repeatability_max_abs"] and length <= noise["repeatability_l2"]}
                            record["frozen_router_identity"]["both_tp_ranks_pass"] = _both_ranks_pass(model, record["frozen_router_identity"]["identity_pass"])
                        else:
                            record["frozen_router_identity"] = baseline[key]["frozen_router_identity"]
                        if record["frozen_router_identity"]["both_tp_ranks_pass"]:
                            record["router_decomposition"] = router_decomposition(y0, fixed, frozen[0])
                        else:
                            record["frozen_router_status"] = "FROZEN_ROUTER_BLOCKED"
                    else:
                        record["frozen_router_status"] = "FROZEN_ROUTER_BLOCKED"
                if record["status"] != "PASS" or record.get("frozen_router_status") == "FROZEN_ROUTER_BLOCKED":
                    blocked.append(key)
                records.append(record)
            path = Path(self.output_root) / sample / f"rank{rank}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"variant": self.variant, "rank": rank, "world_size": 2, "sample_key": sample,
                        "scope": "production_incremental_decode_only", "records": records}, path)
            results.append(str(path))
        return {"rank": rank, "paths": results, "blocked_keys": blocked, "pass": not blocked}
