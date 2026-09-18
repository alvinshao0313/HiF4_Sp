"""CPU unit tests of the real training loop; CUDA accounting is isolated explicitly."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from Native_NVFP4_HiF4_Linear_Puncture.experiments.kl_direction_reuse import train as training
from Native_NVFP4_HiF4_Linear_Puncture.experiments.kl_direction_reuse.artifacts import ActiveClock, load, read_json
from Native_NVFP4_HiF4_Linear_Puncture.experiments.kl_direction_reuse.config import PROTOCOL


class ScalarRuntime:
    def __init__(self, snapshot, baseline, layer):
        self.layer, self.device = layer, torch.device("cpu")
        self.student = torch.nn.Module()
        self.student.learned = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.student.learned.weight.fill_(1.)

    def snapshot_parameters(self):
        return {"weight": self.student.learned.weight.detach().clone()}

    def restore(self, state):
        self.student.learned.load_state_dict(state)


@pytest.fixture
def loop(tmp_path, monkeypatch):
    now = [0.]
    store = SimpleNamespace(manifest={"model_files": {}},
        layer=lambda sid, layer: {"input": torch.ones(1, 1), "output": torch.zeros(1, 1), "router_logits": torch.zeros(1, 1)})
    ds = SimpleNamespace(root=tmp_path, snapshot=tmp_path, protocol={"batches": [["a", "b", "c", "d"]]},
                         samples={s: {"input_ids": torch.tensor([1])} for s in "abcd"})
    monkeypatch.setattr(training, "PROTOCOL", replace(PROTOCOL, budget_seconds=10.))
    monkeypatch.setattr(training, "CaptureStore", lambda *a: store)
    monkeypatch.setattr(training, "check_baseline", lambda *a: None)
    monkeypatch.setattr(training, "model_identity", lambda *a: {})
    original_read = training.read_json
    monkeypatch.setattr(training, "read_json", lambda p: {"files": {}} if str(p).endswith("export.json") else original_read(p))
    monkeypatch.setattr(training, "training_provenance", lambda *a: {"protocol": "test"})
    monkeypatch.setattr(training, "require_verification", lambda *a: None)
    monkeypatch.setattr(training, "Runtime", ScalarRuntime)
    monkeypatch.setattr(training, "ActiveClock", lambda prior: ActiveClock(prior, now=lambda: now[0]))
    for name in ("reset_peak_memory_stats", "synchronize"):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    recipe = dict(name="unit_mse", layer=8, objective="mse", reuse_epochs=0)
    out = tmp_path / "run"
    out.mkdir()
    return ds, recipe, out, now


def test_partial_batch_at_deadline_is_discarded(loop, monkeypatch):
    ds, recipe, out, now = loop
    def backward(runtime, *args):
        runtime.student.learned.weight.square().sum().backward()
        now[0] += 1.5
        return dict(main=1., router=0.)
    monkeypatch.setattr(training, "backward_sample", backward)
    manifest = training._train(ds, recipe, out, False)
    assert manifest["status"] == "COMPLETE" and manifest["steps"] == 1
    assert manifest["committed_seconds"] == 6.
    assert manifest["charged_seconds"] == 10.5
    cp = load(out / "final.pt")
    assert cp["step"] == 1 and cp["tokens_seen"] == 4
    assert cp["parameters"]["weight"].item() == pytest.approx(.9999, abs=1e-7)


def test_optimizer_update_finishing_after_deadline_is_not_committed(loop, monkeypatch):
    ds, recipe, out, now = loop
    def backward(runtime, *args):
        runtime.student.learned.weight.square().sum().backward()
        now[0] += 2.
        return dict(main=1., router=0.)
    original_step = torch.optim.AdamW.step
    def slow_step(self, *args, **kwargs):
        result = original_step(self, *args, **kwargs)
        now[0] += 3.
        return result
    monkeypatch.setattr(training, "backward_sample", backward)
    monkeypatch.setattr(torch.optim.AdamW, "step", slow_step)
    with pytest.raises(RuntimeError, match="no parameter update"):
        training._train(ds, recipe, out, False)
    manifest = read_json(out / "manifest.json")
    assert manifest["status"] == "NO_UPDATE_WITHIN_BUDGET"
    assert manifest["steps"] == 0
    assert load(out / "final.pt")["parameters"]["weight"].item() == 1.


def test_interruption_resume_keeps_spent_time_and_optimizer_state(loop, monkeypatch):
    ds, recipe, out, now = loop
    calls = [0]
    def backward(runtime, *args):
        calls[0] += 1
        now[0] += 1.
        if calls[0] == 5:
            raise KeyboardInterrupt()
        runtime.student.learned.weight.square().sum().backward()
        return dict(main=1., router=0.)
    monkeypatch.setattr(training, "backward_sample", backward)
    with pytest.raises(KeyboardInterrupt):
        training._train(ds, recipe, out, False)
    state = read_json(out / "state.json")
    assert state["status"] == "INTERRUPTED" and state["charged_seconds"] == 5.
    assert load(out / "resume.pt")["step"] == 1
    manifest = training._train(ds, recipe, out, True)
    assert manifest["steps"] == 2 and manifest["charged_seconds"] == 10.
    cp = load(out / "final.pt")
    assert cp["optimizer"]["state"][0]["step"].item() == 2
    assert cp["parameters"]["weight"].item() == pytest.approx(.9998, abs=1e-7)
