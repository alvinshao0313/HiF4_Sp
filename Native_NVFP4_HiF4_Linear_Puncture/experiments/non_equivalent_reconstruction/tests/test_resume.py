import torch
import pytest

from Native_NVFP4_HiF4_Linear_Puncture.experiments.non_equivalent_reconstruction.artifact import (
    atomic_save, load, cpu_tree, load_manifest, write_json, KIND, SCHEMA_VERSION,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.non_equivalent_reconstruction.config import Config, DEFAULT_INIT


def test_optimizer_scheduler_resume_preserves_next_step(tmp_path):
    p = torch.nn.Parameter(torch.randn(64, 64))
    opt = torch.optim.AdamW([p], lr=1e-4, weight_decay=0)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5)

    def step(param, optimizer, scheduler):
        optimizer.zero_grad(set_to_none=True)
        (param.square().mean() + param.mean()).backward()
        optimizer.step()
        scheduler.step()

    step(p, opt, schedule)
    atomic_save({"p": p.detach(), "optimizer": cpu_tree(opt.state_dict()), "scheduler": schedule.state_dict()}, tmp_path / "resume.pt")
    state = load(tmp_path / "resume.pt")
    other = torch.nn.Parameter(state["p"].clone())
    opt2 = torch.optim.AdamW([other], lr=1e-4, weight_decay=0)
    schedule2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=5)
    opt2.load_state_dict(state["optimizer"])
    schedule2.load_state_dict(state["scheduler"])
    step(p, opt, schedule)
    step(other, opt2, schedule2)
    torch.testing.assert_close(p, other, rtol=0, atol=0)
    assert schedule.get_last_lr() == schedule2.get_last_lr()


def test_default_initialization_exists_and_config_rejects_invalid():
    assert DEFAULT_INIT.is_file()
    Config(output_dir="test").validate()
    with pytest.raises(ValueError):
        Config(output_dir="test", router_temperature=0).validate()


def test_incomplete_artifact_cannot_be_exported(tmp_path):
    write_json({"kind": KIND, "schema_version": SCHEMA_VERSION, "completed_layers": []}, tmp_path / "manifest.json")
    assert load_manifest(tmp_path)["completed_layers"] == []
    with pytest.raises(ValueError, match="48"):
        load_manifest(tmp_path, require_complete=True)
