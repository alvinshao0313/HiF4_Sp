from types import SimpleNamespace

import pytest
import torch

from Native_NVFP4_HiF4_Linear_Puncture.experiments.kl_direction_reuse.artifacts import (
    ActiveClock, BudgetExpired, check_budget, checked_load, parameters_hash, save, tensor_hash, write_json,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.kl_direction_reuse.capture import Logits
from Native_NVFP4_HiF4_Linear_Puncture.experiments.kl_direction_reuse.config import recipes, require_gpus
from Native_NVFP4_HiF4_Linear_Puncture.experiments.kl_direction_reuse.data import fixed_batches, epoch_order
from Native_NVFP4_HiF4_Linear_Puncture.experiments.kl_direction_reuse.directions import Directions
from Native_NVFP4_HiF4_Linear_Puncture.experiments.kl_direction_reuse.report import aggregate, paired_gain
from Native_NVFP4_HiF4_Linear_Puncture.experiments.kl_direction_reuse.train import require_verification


def test_recipes_and_fixed_batch_membership():
    all_recipes = recipes()
    assert len(all_recipes) == len({r["name"] for r in all_recipes}) == 15
    assert {r["reuse_epochs"] for r in all_recipes if r["objective"] == "cached_kl"} == {1, 2, 4}
    samples = [{"id": str(i), "input_ids": torch.arange(i+2)} for i in reversed(range(32))]
    batches = fixed_batches(samples)
    assert all(len(b) == 4 for b in batches)
    assert batches[0] == ["0", "1", "2", "3"]
    assert epoch_order(8, 0) == epoch_order(8, 0)
    assert epoch_order(8, 0) != epoch_order(8, 1)
    assert sorted(epoch_order(8, 3)) == list(range(8))


def test_changed_artifact_fails_instead_of_silently_loading(tmp_path):
    path = tmp_path / "state.pt"
    entry = save({"p": torch.tensor([1.])}, path)
    assert checked_load(entry)["p"].item() == 1
    save({"p": torch.tensor([2.])}, path)
    with pytest.raises(RuntimeError, match="artifact changed"):
        checked_load(entry)


def test_capture_provenance_separates_training_fixes_from_capture_changes(tmp_path):
    import shutil
    from pathlib import Path
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.kl_direction_reuse import artifacts
    for path in Path(artifacts.__file__).parent.glob('*.py'):
        shutil.copy2(path, tmp_path / path.name)
    before = artifacts.capture_fingerprint(tmp_path)
    with (tmp_path / 'runtime.py').open('a') as f:
        f.write('\n# Differentiable runtime change\n')
    assert artifacts.capture_fingerprint(tmp_path) == before
    assert artifacts.source_fingerprint(tmp_path) != artifacts.source_fingerprint()
    path = tmp_path / 'losses.py'
    path.write_text(path.read_text().replace('z = logits.float()', 'z = logits.float() + 1.'))
    assert artifacts.capture_fingerprint(tmp_path) != before


def test_logit_reader_covers_slices_across_chunks_and_rejects_holes(tmp_path):
    z = torch.arange(35).reshape(7, 5).float()
    entries = [{**save(z[a:b], tmp_path/f"{a}.pt"), "start": a, "stop": b, "vocab": 5}
               for a, b in ((0, 3), (3, 6), (6, 7))]
    reader = Logits(entries, 7)
    assert torch.equal(reader[2:7], z[2:7])
    assert torch.equal(reader[0:1], z[0:1])
    with pytest.raises(RuntimeError, match="coverage"):
        Logits([entries[0], entries[2]], 7)


def test_direction_cache_rejects_partial_wrong_batch_and_modified_files(tmp_path):
    tokens = torch.tensor([1, 2, 3])
    samples = {"one": {"input_ids": tokens, "token_sha256": tensor_hash(tokens)}}
    ds = SimpleNamespace(samples=samples, protocol={"batches": [["one"]]}, ids=lambda split: ["one"])
    record = dict(anchor=torch.zeros(1, 3, 2), gradient=torch.ones(1, 3, 2),
                  batch_tokens=3, token_sha256=tensor_hash(tokens))
    entry = save(record, tmp_path / "one.pt")
    anchor = {"weight": torch.tensor([1.])}
    manifest = dict(status="RUNNING", provenance={"p": "hash"}, layer=8, samples={"one": entry}, batches=[["one"]],
                    anchor_parameters=save(anchor, tmp_path / "anchor.pt"), parameters_sha256=parameters_hash(anchor))
    write_json(manifest, tmp_path / "manifest.json")
    with pytest.raises(RuntimeError, match="provenance"):
        Directions(tmp_path, provenance={"p": "hash"}, ds=ds, layer=8)
    manifest["status"] = "COMPLETE"
    write_json(manifest, tmp_path / "manifest.json")
    cache = Directions(tmp_path, provenance={"p": "hash"}, ds=ds, layer=8)
    assert torch.equal(cache.sample("one", 3)["gradient"], record["gradient"])
    with pytest.raises(RuntimeError, match="denominator"):
        cache.sample("one", 4)


def test_deadline_is_inclusive_and_resume_does_not_reset_budget():
    now = [100.]
    clock = ActiveClock(7000., now=lambda: now[0])
    now[0] = 299.
    check_budget(clock, 7200.)
    now[0] = 300.
    with pytest.raises(BudgetExpired):
        check_budget(clock, 7200.)


def test_gpu_allocation_is_explicit(monkeypatch):
    for value in ("", "0,1", "-1"):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)
        with pytest.raises(RuntimeError, match="explicit"):
            require_gpus(1)


def test_failed_or_stale_production_gate_cannot_start_training(tmp_path):
    path = tmp_path / "verification/L08/report.json"
    provenance = {"source": "new"}
    for status, source in (("FAIL", "new"), ("PASS", "old")):
        write_json(dict(status=status, provenance={"source": source}, layer=8), path)
        with pytest.raises(RuntimeError, match="verification"):
            require_verification(tmp_path, 8, provenance)
    write_json(dict(status="PASS", provenance=provenance, layer=8), path)
    assert require_verification(tmp_path, 8, provenance)["status"] == "PASS"


def test_paired_bootstrap_and_token_weighted_metrics():
    base = [dict(kl_sum=n*2., kl_tokens=n, nll_sum=n*1.5, nll_tokens=n) for n in (2, 10, 20)]
    candidate = [{**r, "kl_sum": r["kl_sum"]-.25*r["kl_tokens"]} for r in base]
    gain = paired_gain(candidate, base)
    assert gain["kl_gain"] == .25
    assert gain["ci95"] == pytest.approx([.25, .25])
    assert aggregate(base)["kl"] == 2
    assert aggregate(base)["nll"] == 1.5
    with pytest.raises(ValueError, match="coverage"):
        paired_gain([{**candidate[0], "kl_tokens": 3}], base[:1])
