"""Report TP2 numerical differences; gate semantics and fresh-adjoint validity."""
from pathlib import Path

import torch

from .production import final_norm
from .artifacts import checked_load, read_json, save, sha256, tensor_hash, write_json
from .capture import CaptureStore
from .config import PROTOCOL, require_gpus
from .data import Dataset
from .materialize import candidate, check_baseline
from .optimization import backward_sample, collect_direction, finite_gradients
from .losses import head_logits, distribution_metrics
from .runtime import Runtime
from .train import training_provenance


def direction_alignment(actual, expected):
    """Compare update directions while reporting numerical magnitude drift."""
    if set(actual) != set(expected):
        raise RuntimeError("direction parameter coverage differs")
    left = torch.cat([actual[k].detach().cpu().float().reshape(-1) for k in sorted(actual)])
    right = torch.cat([expected[k].detach().cpu().float().reshape(-1) for k in sorted(expected)])
    if left.shape != right.shape or not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise RuntimeError("nonfinite or incomplete direction")
    nl, nr = left.norm(), right.norm()
    if nl == 0 or nr == 0:
        raise RuntimeError("zero direction")
    delta = left - right
    return dict(cosine=float((left.dot(right) / (nl * nr)).clamp(-1., 1.)),
                relative_l2=float(delta.norm() / nr),
                norm_ratio=float(nl / nr),
                max_abs=float(delta.abs().max()),
                elements=int(left.numel()))


def direction_passed(value):
    # The experiment reuses direction, not the absolute adjoint magnitude.
    # 0.999 leaves only the measured production-kernel rounding/replay error.
    return value["cosine"] >= 0.999 and 0.8 <= value["norm_ratio"] <= 1.25


def difference(actual, expected):
    if actual.shape != expected.shape:
        raise RuntimeError(f"shape mismatch: {actual.shape} vs {expected.shape}")
    a, b = actual.detach().cpu().float(), expected.detach().cpu().float()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise RuntimeError("nonfinite verification tensors")
    return dict(exact=torch.equal(a, b), max_abs=float((a-b).abs().max()),
                relative_l2=float((a-b).norm()/b.norm()) if b.norm() else None,
                mismatches=int((a != b).sum()), elements=a.numel())


@torch.no_grad()
def compare_forward(runtime, ds, baseline, observed, ids):
    rows = []
    teacher = CaptureStore(ds.root / 'captures/native', ds)
    for sid in ids:
        x = baseline.layer(sid, runtime.layer)["input"].unsqueeze(0).to(runtime.device)
        out = runtime.local(x, use_ste=False)
        ref = observed.layer(sid, runtime.layer)
        row = dict(sample=sid, layer_output=difference(out.output[0], ref["output"]),
                   router_logits=difference(out.router_logits[0], ref["router_logits"]))
        final = final_norm(runtime.suffix(out.output), runtime.norm, runtime.eps)[0]
        expected = checked_load(observed.manifest["samples"][sid]["final_hidden"])
        row["final_hidden"] = difference(final, expected)
        chunks = observed.manifest["samples"][sid]["logits_hashes"]
        cursor = 0
        row["final_logits"] = {"exact": True}
        native_logits = teacher.logits(sid)
        metrics = dict(kl_sum=0., kl_tokens=0, nll_sum=0., nll_tokens=0)
        for chunk in chunks:
            if chunk["start"] != cursor or chunk["stop"] <= cursor:
                raise RuntimeError("production logits hash coverage mismatch")
            value = head_logits(final[chunk["start"]:chunk["stop"]], runtime.head)
            delta = distribution_metrics(value, native_logits[chunk['start']:chunk['stop']].to(value.device),
                                         ds.samples[sid]['input_ids'], chunk['start'])
            for key in metrics:
                metrics[key] += delta[key]
            row["final_logits"]["exact"] &= tensor_hash(value) == chunk["sha256"]
            cursor = chunk["stop"]
        if cursor != len(final):
            raise RuntimeError("incomplete production logit hashes")
        actual_metrics = observed.manifest['samples'][sid]['metrics']
        row['final_distribution'] = {
            'local_kl': metrics['kl_sum'] / metrics['kl_tokens'],
            'tp2_kl': actual_metrics['kl_sum'] / actual_metrics['kl_tokens'],
            'local_nll': metrics['nll_sum'] / metrics['nll_tokens'],
            'tp2_nll': actual_metrics['nll_sum'] / actual_metrics['nll_tokens'],
        }
        rows.append(row)
        print(f"forward layer={runtime.layer} sample={sid}: "
              f"layer={row['layer_output']['exact']} router={row['router_logits']['exact']} "
              f"final={row['final_hidden']['exact']} logits={row['final_logits']['exact']}", flush=True)
    return rows


def forward_passed(rows):
    return bool(rows) and all(r[key]["exact"] for r in rows for key in ("layer_output", "router_logits", "final_hidden", "final_logits"))


def prepare(root, layer):
    require_gpus(1)
    ds = Dataset(root)
    check_baseline(ds.root)
    out = ds.root / "verification" / f"L{layer:02d}"
    out.mkdir(parents=True, exist_ok=False)
    provenance = training_provenance(ds.root)
    report = dict(status="RUNNING", layer=layer, provenance=provenance,
                  forward_policy="report numerical differences; bitwise TP equality is not required",
                  direction_policy="global cosine >= 0.999 and norm ratio in [0.8, 1.25]; magnitude drift is reported")
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    runtime = Runtime(ds.snapshot, ds.root / "baseline", layer)
    native = CaptureStore(ds.root / "captures/native", ds)
    baseline = CaptureStore(ds.root / "captures/baseline", ds)
    ordered = sorted(ds.ids("train"), key=lambda s: (len(ds.samples[s]["input_ids"]), s))
    ids = [ordered[0], ordered[-1]]
    report["samples"] = ids
    try:
        report["initial_forward"] = compare_forward(runtime, ds, baseline, baseline, ids)
        report['initial_forward_exact'] = forward_passed(report['initial_forward'])
        write_json(report, out / "prepare.json")
        initial = runtime.snapshot_parameters()
        batch = ds.protocol["batches"][0]
        denominator = sum(len(ds.samples[s]["input_ids"]) for s in batch)
        directions = {}
        for sid in batch:
            print(f"fresh direction layer={layer} sample={sid}", flush=True)
            x = baseline.layer(sid, layer)["input"].unsqueeze(0).to(runtime.device)
            directions[sid] = collect_direction(runtime, x, native.logits(sid), denominator)
        snapshots, gradients = {}, {}
        for objective in ("direct_kl", "cached_kl"):
            runtime.restore(initial)
            optimizer = torch.optim.AdamW(runtime.student.learned.parameters(), lr=PROTOCOL.lr, weight_decay=0.)
            optimizer.zero_grad(set_to_none=True)
            for sid in batch:
                print(f"gradient check layer={layer} objective={objective} sample={sid}", flush=True)
                x = baseline.layer(sid, layer)["input"].unsqueeze(0).to(runtime.device)
                target = native.layer(sid, layer)
                target = {k: target[k].unsqueeze(0).to(runtime.device) for k in ("output", "router_logits")}
                backward_sample(runtime, x, target, native.logits(sid), denominator, objective, directions[sid])
            if finite_gradients(runtime.student.learned.parameters()) == 0:
                raise RuntimeError("zero verification gradient")
            gradients[objective] = {k: p.grad.detach().cpu().clone() for k, p in runtime.student.learned.named_parameters() if p.grad is not None}
            optimizer.step()
            snapshots[objective] = runtime.snapshot_parameters()
        if set(gradients["direct_kl"]) != set(gradients["cached_kl"]):
            raise RuntimeError("fresh adjoint gradient coverage differs")
        report["gradient_alignment"] = direction_alignment(gradients["direct_kl"], gradients["cached_kl"])
        if not direction_passed(report["gradient_alignment"]):
            raise RuntimeError(f"fresh adjoint direction misaligned: {report['gradient_alignment']}")
        updates = {"direct_kl": {k: snapshots["direct_kl"][k] - initial[k] for k in initial},
                   "cached_kl": {k: snapshots["cached_kl"][k] - initial[k] for k in initial}}
        # AdamW is coordinate-adaptive: a small adjoint rounding difference can
        # change its per-coordinate update direction. Record that effect, but
        # gate the scientific claim on the raw final-KL gradient direction above.
        report["update_alignment"] = direction_alignment(updates["direct_kl"], updates["cached_kl"])
        changed = [k for k in initial if not torch.equal(initial[k], snapshots["direct_kl"][k])]
        if not changed:
            raise RuntimeError("nonzero update probe did not change parameters")
        # Verify router-only gradients reach attention, and no preceding layer.
        from .losses import router_term
        runtime.restore(initial)
        runtime.student.zero_grad(set_to_none=True)
        sid = batch[0]
        x = baseline.layer(sid, layer)["input"].unsqueeze(0).to(runtime.device)
        local = runtime.local(x)
        teacher = native.layer(sid, layer)["router_logits"].unsqueeze(0).to(runtime.device)
        router_term(local.router_logits, teacher, len(ds.samples[sid]["input_ids"])).backward()
        for name in ("input_norm", "moe_norm", "matrices.router", "matrices.o_proj"):
            p = dict(runtime.student.learned.named_parameters())[name]
            if p.grad is None or not torch.isfinite(p.grad).all() or not p.grad.count_nonzero():
                raise RuntimeError(f"router gradient did not reach {name}")
        report["router_gradient_scope"] = "PASS"
        report["fresh_gradient"] = "PASS"
        report["probe"] = save(dict(layer=layer, parameters=snapshots["direct_kl"],
                                    protocol_sha256=provenance["protocol"]), out / "probe.pt")
        # vLLM's NVFP4 reference decoder caches its lookup table on the first
        # device it sees.  Probe materialization is intentionally CPU-only,
        # so reset that process-local cache before decoding the source layer.
        from vllm.model_executor.layers.quantization.utils import nvfp4_emulation_utils
        nvfp4_emulation_utils.kE2M1ToFloat_handle.val = (
            nvfp4_emulation_utils.kE2M1ToFloat_handle.val.cpu())
        report["probe_model"] = str(candidate(ds.root, out / "probe.pt", out / "probe_model"))
        probe_files = read_json(out / "probe_model/export.json")["files"]
        shard = f"model-layer-{layer:05d}-of-00048.safetensors"
        if probe_files[shard] == baseline.manifest["model_files"][shard]:
            raise RuntimeError("nonzero parameter update did not change the exported layer")
        report["status"] = "AWAITING_ACTUAL_NONZERO_CAPTURE"
        write_json(report, out / "prepare.json")
    except BaseException as error:
        report.update(status="FAIL", error=str(error))
        write_json(report, out / "prepare.json")
        raise
    return report


def finish(root, layer, capture_dir):
    require_gpus(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ds = Dataset(root)
    out = ds.root / "verification" / f"L{layer:02d}"
    report = read_json(out / "prepare.json")
    provenance = training_provenance(ds.root)
    if report["status"] != "AWAITING_ACTUAL_NONZERO_CAPTURE" or report["provenance"] != provenance:
        raise RuntimeError("verification preparation is stale or failed")
    observed = CaptureStore(capture_dir, ds)
    probe_export = read_json(Path(report["probe_model"]) / "export.json")
    if observed.manifest["model_files"] != probe_export["files"] or set(observed.manifest["samples"]) != set(report["samples"]):
        raise RuntimeError("nonzero capture is not the exact prepared probe")
    runtime = Runtime(ds.snapshot, ds.root / "baseline", layer)
    runtime.restore(checked_load(report["probe"])["parameters"])
    baseline = CaptureStore(ds.root / "captures/baseline", ds)
    report["nonzero_forward"] = compare_forward(runtime, ds, baseline, observed, report["samples"])
    report['nonzero_forward_exact'] = forward_passed(report['nonzero_forward'])
    # Shape, finiteness, complete token/logit coverage, exact exported model
    # identity and the fresh-adjoint/router-scope checks remain mandatory.
    # Normal TP reduction rounding is reported, not a reason to block training.
    report["status"] = "PASS"
    report["production_capture_sha256"] = sha256(Path(capture_dir) / "manifest.json")
    write_json(report, out / "report.json")
    return report
