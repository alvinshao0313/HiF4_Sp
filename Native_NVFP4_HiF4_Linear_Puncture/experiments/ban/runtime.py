"""Worker-local Ban routing; the existing HiF4 GEMMs remain the executor."""
from __future__ import annotations

import types

import torch

from .algorithm import ban_route, reference_distribution, reference_kl, top3_ratio, validate_parameters
from .artifact import load_artifact, model_identity


def difference(actual, reference):
    delta = actual.float() - reference.float()
    return {"max_abs": float(delta.abs().max().item()),
            "relative_l2": float((delta.norm() / reference.float().norm().clamp_min(1e-20)).item())}


class BanState:
    def __init__(self, config, device):
        self.config, self.device = config, device
        self.identity = model_identity(config["model_path"], config["spec_path"])
        self.layers, self.top_k = self.identity["num_layers"], self.identity["top_k"]
        self.mode, self.active, self.capture = "native", False, None
        self.execution = config.get("execution", "compact")
        self.layer, self.k_pruned, self.topn = None, 3, 1000
        self.reference, self.reference_logits = None, None
        self.sensitivity, self.r_min, self.r_max = None, None, None
        self.probe = False
        self.load(config.get("artifact_path"))
        self.reset()

    def load(self, path):
        if path:
            artifact = load_artifact(path, self.identity)
            self.sensitivity = artifact["layer_sensitivity"].to(self.device)
            self.r_min, self.r_max = artifact["r_min"], artifact["r_max"]
            self.mode = "ban"

    def reset(self):
        self.hist = torch.zeros((self.layers, self.top_k + 1), device=self.device, dtype=torch.long)
        self.dispatched = torch.zeros(self.layers, device=self.device, dtype=torch.long)
        self.ratio_min = torch.full((self.layers,), float("inf"), device=self.device)
        self.ratio_max = -self.ratio_min.clone()
        self.counts, self.probe_routes, self.probes = {}, {}, {}
        self.captured, self.logit_calls = None, []

    def route(self, original, index, router, x, logits, dtype):
        if not self.active:
            return original(x, logits, dtype)
        if self.mode in {"native", "collect", "identity"}:
            weights, ids = original(x, logits, dtype)
            counts = torch.full((x.shape[0],), self.top_k, device=x.device, dtype=torch.long)
            if self.mode == "collect" and x.shape[0]:
                ratio = top3_ratio(weights.sort(-1, descending=True).values)
                self.ratio_min[index] = torch.minimum(self.ratio_min[index], ratio.min())
                self.ratio_max[index] = torch.maximum(self.ratio_max[index], ratio.max())
            if self.probe:
                other = ban_route(logits, self.top_k, torch.tensor(0., device=x.device), 0., 1.,
                                  0.7, self.top_k, router.renormalize,
                                  torch.int32 if dtype is None else dtype)
                self.probe_routes[index] = (other[0], other[1])
                self.probes[index] = {"same_expert_set": bool((ids.sort(-1).values == other[1].sort(-1).values).all().item()),
                                      "weights_max_abs": float((weights - other[0]).abs().max().item())}
        elif self.mode == "ban" or self.mode == "full":
            sensitivity = torch.tensor(0., device=x.device) if self.mode == "full" else self.sensitivity[index]
            weights, ids, counts = ban_route(
                logits, self.top_k, sensitivity,
                0. if self.mode == "full" else self.r_min, 1. if self.mode == "full" else self.r_max,
                self.config.get("lam", 0.7), self.top_k if self.mode == "full" else self.config.get("k_min", 3),
                router.renormalize, torch.int32 if dtype is None else dtype)
        elif self.mode == "single_layer":
            if index != self.layer:
                weights, ids = original(x, logits, dtype)
                counts = torch.full((x.shape[0],), self.top_k, device=x.device, dtype=torch.long)
            else:
                weights, ids = torch.topk(torch.softmax(logits.float(), -1), self.top_k, -1, sorted=True)
                weights[:, self.k_pruned:] = 0
                if router.renormalize:
                    weights = weights / weights.sum(-1, keepdim=True)
                ids = ids.to(torch.int32 if dtype is None else dtype)
                counts = torch.full((x.shape[0],), self.k_pruned, device=x.device, dtype=torch.long)
        else:
            raise ValueError(f"unknown Ban mode {self.mode}")
        self.hist[index] += torch.bincount(counts, minlength=self.top_k + 1)
        self.counts[index] = counts
        return weights, ids

    def snapshot(self):
        hist = self.hist.cpu()
        tokens = hist.sum(-1)
        selected = (hist * torch.arange(self.top_k + 1)).sum(-1)
        return {"mode": self.mode, "execution": self.execution, "histogram": hist.tolist(),
                "tokens": tokens.tolist(), "selected": selected.tolist(), "dispatched": self.dispatched.cpu().tolist(),
                "ratio_min": [v.item() if torch.isfinite(v) else None for v in self.ratio_min.cpu()],
                "ratio_max": [v.item() if torch.isfinite(v) else None for v in self.ratio_max.cpu()],
                "capture": self.captured, "logit_calls": self.logit_calls, "probes": self.probes,
                "allocated_bytes": torch.cuda.memory_allocated(self.device)}


def install_model(model, state):
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import FusedTopKRouter
    from vllm.model_executor.layers.quantization.hif4_runtime import layer_index_from_prefix
    attached = []
    for name, layer in model.named_modules():
        if not isinstance(layer, FusedMoE):
            continue
        index, router = layer_index_from_prefix(name), layer.router
        if (not isinstance(router, FusedTopKRouter) or router.enable_eplb or router.scoring_func != "softmax"
                or router.top_k != state.top_k or layer._shared_experts is not None or layer.quant_method.is_monolithic):
            raise ValueError(f"unsupported Ban route at {name}")
        original = router._compute_routing

        def route(self, hidden_states, router_logits, indices_type, _old=original, _index=index):
            return state.route(_old, _index, self, hidden_states, router_logits, indices_type)

        router._compute_routing = types.MethodType(route, router)
        attached.append(index)
    if sorted(attached) != list(range(state.layers)):
        raise ValueError(f"Ban did not attach all layers: {attached}")
    original_logits = model.compute_logits

    def compute_logits(self, hidden_states):
        logits = original_logits(hidden_states)
        if state.active and state.capture and logits is not None:
            state.logit_calls.append(list(logits.shape))
            if logits.shape[0] != 1 or len(state.logit_calls) != 1:
                raise ValueError("calibration requires exactly one last-input-position logit capture per request")
            # Copy before the sampler can transform its logits buffer.
            current = logits.detach().float().clone()
            if state.capture == "reference":
                state.reference_logits = current
                state.reference = reference_distribution(current, state.topn)
                state.captured = {"reference": True, "top1": int(current.argmax(-1).item())}
            elif state.capture in {"kl", "compare"}:
                if state.reference is None:
                    raise RuntimeError("missing preceding reference")
                state.captured = {"kl": float(reference_kl(current, state.reference).item()),
                                  "top1": int(current.argmax(-1).item())}
                if state.capture == "compare":
                    state.captured.update(difference(current, state.reference_logits))
            else:
                raise ValueError(f"unknown capture {state.capture}")
        return logits

    model.compute_logits = types.MethodType(compute_logits, model)


def install_dispatch(state):
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import UnquantizedFusedMoEMethod
    from vllm.model_executor.layers.quantization.hif4_runtime import layer_index_from_prefix
    original = UnquantizedFusedMoEMethod.apply

    def apply(self, layer, x, topk_weights, topk_ids, shared_experts_input):
        if not state.active:
            return original(self, layer, x, topk_weights, topk_ids, shared_experts_input)
        index = layer_index_from_prefix(layer.layer_name)
        if state.mode in {"native", "collect"} or state.execution == "masked":
            result = original(self, layer, x, topk_weights, topk_ids, shared_experts_input)
            state.dispatched[index] += topk_ids.numel()
            if state.probe:
                repeat = original(self, layer, x, topk_weights, topk_ids, shared_experts_input)
                weights, ids = state.probe_routes[index]
                sorted_result = original(self, layer, x, weights, ids, shared_experts_input)
                state.probes[index].update(repeat=difference(repeat, result), sorted=difference(sorted_result, result))
            return result
        counts = state.counts[index]
        out = torch.empty_like(x)
        for k in range(1, state.top_k + 1):
            rows = torch.where(counts == k)[0]
            if not rows.numel():
                continue
            full = rows.numel() == x.shape[0]
            inputs = x if full else x.index_select(0, rows)
            weights = (topk_weights if full else topk_weights.index_select(0, rows))[:, :k].contiguous()
            ids = (topk_ids if full else topk_ids.index_select(0, rows))[:, :k].contiguous()
            if k == state.top_k:
                result = original(self, layer, inputs, weights, ids, shared_experts_input if full else None)
            else:
                from .moe import apply_compact_moe
                result = apply_compact_moe(inputs, layer.w13_weight, layer.w2_weight, weights, ids, state.top_k)
            if full:
                out.copy_(result)
            else:
                out.index_copy_(0, rows, result)
            state.dispatched[index] += rows.numel() * k
        return out

    UnquantizedFusedMoEMethod.apply = apply
