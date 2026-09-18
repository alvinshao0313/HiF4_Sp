from pathlib import Path
import torch

from .artifacts import checked_load, check_budget, parameters_hash, read_json, save, write_json
from .optimization import collect_direction


class Directions:
    def __init__(self, path, *, provenance, ds, layer):
        self.path = Path(path)
        self.manifest = read_json(self.path / "manifest.json")
        m = self.manifest
        if m["status"] != "COMPLETE" or m["provenance"] != provenance or m["layer"] != layer:
            raise RuntimeError("direction cache provenance mismatch")
        if set(m["samples"]) != set(ds.ids("train")) or m["batches"] != ds.protocol["batches"]:
            raise RuntimeError("direction cache coverage or batch normalization changed")
        if parameters_hash(checked_load(m["anchor_parameters"])) != m["parameters_sha256"]:
            raise RuntimeError("direction anchor parameters changed")
        self.ds = ds

    def sample(self, sid, denominator):
        item = checked_load(self.manifest["samples"][sid])
        if item["batch_tokens"] != denominator or item["token_sha256"] != self.ds.samples[sid]["token_sha256"]:
            raise RuntimeError("direction sample or token denominator mismatch")
        if item["anchor"].shape != item["gradient"].shape or item["anchor"].shape[1] != len(self.ds.samples[sid]["input_ids"]):
            raise RuntimeError("direction tensor coverage mismatch")
        if not torch.isfinite(item["anchor"]).all() or not torch.isfinite(item["gradient"]).all():
            raise RuntimeError("nonfinite cached direction")
        return item


def build(path, *, runtime, ds, native, baseline, epoch, step, provenance, clock, budget, previous=None):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    anchor = runtime.snapshot_parameters()
    anchor_entry = save(anchor, path / "anchor_parameters.pt")
    manifest = dict(status="RUNNING", layer=runtime.layer, epoch=epoch, step=step,
                    provenance=provenance, batches=ds.protocol["batches"], samples={},
                    parameters_sha256=parameters_hash(anchor), anchor_parameters=anchor_entry)
    write_json(manifest, path / "manifest.json")
    totals = dict(kl_sum=0., tokens=0, predicted_change_sum=0., dot=0., old_norm2=0., new_norm2=0.)
    for batch in ds.protocol["batches"]:
        n = sum(len(ds.samples[s]["input_ids"]) for s in batch)
        for sid in batch:
            check_budget(clock, budget)
            x = baseline.layer(sid, runtime.layer)["input"].unsqueeze(0).to(runtime.device)
            record = collect_direction(runtime, x, native.logits(sid), n)
            record.update(batch_tokens=n, token_sha256=ds.samples[sid]["token_sha256"])
            if previous is not None:
                old = previous.sample(sid, n)
                g, g0 = record["gradient"].double(), old["gradient"].double()
                # Each cached g is normalized by its fixed batch token count.
                totals["predicted_change_sum"] += float((g0 * (record["anchor"].double()-old["anchor"].double())).sum()) * n
                totals["dot"] += float((g * g0).sum())
                totals["old_norm2"] += float(g0.square().sum())
                totals["new_norm2"] += float(g.square().sum())
            totals["kl_sum"] += record["kl_sum"]
            totals["tokens"] += x.shape[1]
            manifest["samples"][sid] = save(record, path / f"{sid}.pt")
            write_json(manifest, path / "manifest.json")
            check_budget(clock, budget)
    if parameters_hash(runtime.snapshot_parameters()) != manifest["parameters_sha256"]:
        raise RuntimeError("parameters changed during a full-dataset direction refresh")
    diagnostics = dict(anchor_kl=totals["kl_sum"] / totals["tokens"])
    if previous is not None:
        scale = (totals["old_norm2"] * totals["new_norm2"]) ** .5
        diagnostics.update(predicted_kl_change=totals["predicted_change_sum"] / totals["tokens"],
                           actual_kl_change=diagnostics["anchor_kl"]-previous.manifest["diagnostics"]["anchor_kl"],
                           gradient_cosine=totals["dot"]/scale if scale else None)
    manifest.update(status="COMPLETE", diagnostics=diagnostics)
    write_json(manifest, path / "manifest.json")
    return Directions(path, provenance=provenance, ds=ds, layer=runtime.layer)
