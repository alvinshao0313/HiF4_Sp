"""Fit and apply one projection group before observing its downstream inputs."""
from collections import defaultdict
from contextlib import ExitStack
import torch
from torch.utils._pytree import tree_map

from .calibration import awq, gptq, smoothquant, CalibratedWeights
from .layers import LogicalExperts, Projection


def move(tree, device):
    return tree_map(lambda x: x.detach().to(device) if isinstance(x, torch.Tensor) else x, tree)


def capture(block, modules, inputs, kwargs, device):
    calls = {module: [] for module in modules}
    def record(module, args, kw):
        calls[module].append(move((args, kw), "cpu"))
    with ExitStack() as stack:
        for module in modules:
            stack.enter_context(module.register_forward_pre_hook(record, with_kwargs=True))
        for item in inputs:
            x = item.to(device)
            block(x, **kwargs(x))
    if any(len(rows) != len(inputs) for rows in calls.values()):
        raise ValueError("Expected one activation per module and calibration sequence")
    return calls


def projected_inputs(calls):
    matrices = []
    for args, kwargs in calls:
        x = args[0] if args else kwargs["hidden_states"]
        matrices.append(x.reshape(-1, x.shape[-1]))
    return matrices


def fit(nodes, inputs, target, method, parent_outputs=None):
    device = nodes[0].weight.device
    weights = [node.weight.detach() for node in nodes]
    if method == "gptq":
        state = CalibratedWeights([gptq(w, inputs, target).weights[0] for w in weights], None)
    elif method == "smoothquant":
        maximum = torch.stack([x.abs().amax(0).float() for x in inputs]).amax(0).to(device)
        state = smoothquant(weights, maximum, target)
    elif method == "awq":
        rows = sum(len(x) for x in inputs)
        mean = sum(x.abs().sum(0, dtype=torch.float32) for x in inputs).to(device) / rows
        if parent_outputs is None:
            # Tr(ΔW XᵀX ΔWᵀ) is exactly the squared linear-output error.
            # It avoids allocating token-by-vocabulary logits for the QAD head.
            width = weights[0].shape[-1]
            gram = None
            if rows >= width:
                gram = torch.zeros((width, width), dtype=torch.float32, device=device)
                for x in inputs:
                    x = x.to(device, torch.float32)
                    gram.addmm_(x.T, x)
            def loss(candidate):
                error = torch.zeros((), device=device)
                for original, effective in zip(weights, candidate, strict=True):
                    for start in range(0, len(original), 256):
                        delta = effective[start:start+256] - original[start:start+256].float()
                        if gram is None:
                            for x in inputs:
                                error += (x.to(device, torch.float32) @ delta.T).square().sum()
                        else:
                            error += ((delta @ gram) * delta).sum()
                return float(error) / rows
        else:
            for node in nodes:
                node.weight_only = True
            references = [out.detach().cpu() for out in parent_outputs()]
            def loss(candidate):
                for node, effective in zip(nodes, candidate, strict=True):
                    node.weight.data = effective.to(node.weight.dtype)
                squared = torch.zeros((), device=device)
                count = 0
                for y, ref in zip(parent_outputs(), references, strict=True):
                    squared += (y.float() - ref.to(device).float()).square().sum()
                    count += y.numel()
                return float(squared) / count
        try:
            state = awq(weights, mean, target, loss)
        finally:
            for node, original in zip(nodes, weights, strict=True):
                node.weight.data = original
                node.weight_only = False
    else:
        raise ValueError(method)
    for node, value in zip(nodes, state.weights, strict=True):
        node.weight.data = value
        node.channel_scale = state.input_scale
        node.target = target
    return {"names": [node.name for node in nodes], "tokens": sum(len(x) for x in inputs),
            "sequences": len(inputs), "awq_ratio": state.ratio}


def calibrate_block(block, hidden, kwargs, target, method):
    nodes = {n: m for n, m in block.named_modules() if isinstance(m, Projection) and not m.fp8}
    device = hidden[0].device if not nodes else next(iter(nodes.values())).weight.device
    for node in nodes.values():
        node.calibrating_awq = method == "awq"
    groups = defaultdict(dict)
    expert_layers = {n: m for n, m in block.named_modules() if isinstance(m, LogicalExperts)}
    for path, node in nodes.items():
        if not any(path.startswith(n + ".") for n in expert_layers):
            parent, _, name = path.rpartition(".")
            groups[parent][name] = node
    records = []
    # QKV / GDN QKV+Z and gate+up share one input scale in fused inference.
    order = [("q_proj", "k_proj", "v_proj"), ("in_proj_qkv", "in_proj_z"),
             ("in_proj",), ("o_proj",), ("out_proj",), ("gate_proj", "up_proj"), ("down_proj",)]
    for parent_name, children in groups.items():
        parent = block.get_submodule(parent_name)
        for names in order:
            selected = [children[n] for n in names if n in children]
            if not selected:
                continue
            need_parent = method == "awq" and len(selected) > 1
            calls = capture(block, [selected[0], parent] if need_parent else [selected[0]],
                            hidden, kwargs, device)
            xs = projected_inputs(calls[selected[0]])
            def parent_outputs():
                for args, kw in calls[parent]:
                    args, kw = move((args, kw), device)
                    out = parent(*args, **kw)
                    yield out[0] if isinstance(out, tuple) else out
            records.append(fit(selected, xs, target, method, parent_outputs if need_parent else None))
    for experts in expert_layers.values():
        calls = capture(block, [experts], hidden, kwargs, device)
        xs = projected_inputs(calls[experts])
        for expert in experts.experts:
            first = [expert[n] for n in ("gate_proj", "up_proj") if n in expert]
            down = expert["down_proj"]
            def expert_outputs():
                for x in xs:
                    yield down(experts.activated(expert, x.to(device)))
            records.append(fit(first, xs, target, method, expert_outputs if method == "awq" else None))
            # All tokens inform each expert. The real router still selects top-k
            # in the block forward used to propagate the next layer's inputs.
            inner = [experts.activated(expert, x.to(device)).cpu() for x in xs]
            records.append(fit([down], inner, target, method))
    expected = {node.name for node in nodes.values()}
    covered = [name for record in records for name in record["names"]]
    if len(covered) != len(expected) or set(covered) != expected:
        raise ValueError(f"Calibration coverage mismatch: {expected ^ set(covered)}")
    return records
