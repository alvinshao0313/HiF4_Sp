"""Protocol and real local-artifact contracts; no substituted model/runtime."""
from pathlib import Path

import pytest
import torch

from ...e2e_diag_reconstruction.core.moe_model_spec import Qwen3MoeModelSpec
from ..artifacts import read_json, write_json
from ..config import NATIVE, PROTOCOL, ROOT, TASKS, task_protocol
from ..evaluate import validate_result
from ..export import require_training
from ..report import compare
from ..run import parse_gpus, stages
from ..training import optimizer_digest


def test_real_native_235b_geometry():
    spec = Qwen3MoeModelSpec.from_snapshot(NATIVE)
    assert (spec.num_layers, spec.hidden_size, spec.num_attention_heads, spec.num_key_value_heads) == (94, 4096, 64, 4)
    assert spec.target_linear_count == 36472


def test_full_paired_protocol_and_aime_multiplicity():
    assert {name: task_protocol(name)["count"] for name in TASKS} == {
        "gsm8k": 1319, "mmlu_pro": 12032, "lcb": 175, "aime25": 30}
    for name in TASKS:
        full, short = task_protocol(name), task_protocol(name, smoke=True)
        assert full["tensor_parallel_size"] == short["tensor_parallel_size"] == 8
        assert short == {**full, "smoke_only": True, "count": 1, "max_samples": 1}
        assert full["generations_per_doc"] == (5 if name == "aime25" else 1)
    assert task_protocol("aime25")["metric"] == "avg@n:n=5"
    assert PROTOCOL["training"]["router_kl_weight"] == 0
    assert not PROTOCOL["training"]["pin_memory"]


def test_only_time_and_data_volume_change_in_training_short_test():
    sequence = list(stages(Path("/unused"), parse_gpus("0,1,2,3,4,5,6,7")))
    names = [row[0] for row in sequence]
    assert names.index("generation") > names.index("native_nvfp4_aime25_qualification")
    assert names.index("training") > names.index("training_qualification")
    assert names.index("matrix_norm_gsm8k_formal") > names.index("matrix_norm_aime25_qualification")
    assert all("arc" not in name for name in names)
    for name, command, devices, artifacts, review in sequence:
        assert len(devices) == (4 if name.startswith("training") else 1 if name == "export" else 8)
        assert not any("resume" in arg for arg in command)


@pytest.mark.parametrize("value", ["0,1,2,3", "0,1,2,3,4,5,6,6", "0,1,2,3,4,5,6,8", "invalid"])
def test_reject_changed_or_ambiguous_allocation(value):
    with pytest.raises(ValueError):
        parse_gpus(value)


@pytest.mark.parametrize("complete,qualification,layers", [(False, False, 94), (True, True, 2), (True, False, 48)])
def test_partial_or_diagnostic_training_cannot_be_formally_exported(tmp_path, complete, qualification, layers):
    write_json({"complete": complete, "qualification": qualification, "source_snapshot": str(NATIVE),
                "completed_layers": list(range(layers))}, tmp_path / "training/manifest.json")
    with pytest.raises(ValueError, match="94 selected"):
        require_training(tmp_path, Qwen3MoeModelSpec.from_snapshot(NATIVE))


def test_gains_do_not_cancel_other_task_losses():
    native = dict.fromkeys(TASKS, .7)
    candidate = {**native, "gsm8k": .75, "mmlu_pro": .68, "lcb": .69}
    result = compare(native, candidate)
    assert result["mean_positive_loss_pp"] == pytest.approx(.75)
    assert result["positive_loss_pp"]["gsm8k"] == 0
    assert "threshold" not in result
    with pytest.raises(ValueError):
        compare(native, {**candidate, "aime25": True})
    with pytest.raises(ValueError):
        compare(native, {k: v for k, v in candidate.items() if k != "aime25"})


def test_historical_tp2_artifact_is_not_a_tp8_result():
    source = ROOT / "results/downstream_accuracy_recovery/paired_full_20260930T131000Z/formal/native_nvfp4/eval/gsm8k/metrics.json"
    if not source.is_file():
        pytest.skip("historical real TP2 artifact is unavailable")
    result = read_json(source)
    with pytest.raises(ValueError, match="protocol"):
        validate_result(result, "gsm8k", {}, model_kind="native_nvfp4")


def test_real_adam_scalar_state_is_hashed_without_losing_shape():
    # Exercise actual PyTorch optimizer state; this is not a model substitute.
    module = torch.nn.Module()
    module.learned = torch.nn.Module()
    module.learned.matrices = torch.nn.ParameterDict({"q_proj": torch.nn.Parameter(torch.eye(64)[None])})
    optimizer = torch.optim.AdamW(module.parameters(), lr=3e-4, weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    module.learned.matrices["q_proj"].square().sum().backward()
    optimizer.step()
    scheduler.step()
    first = optimizer_digest(module, optimizer, scheduler)
    assert first["metadata"]["learned.matrices.q_proj.step.shape"] == []
    optimizer.state[module.learned.matrices["q_proj"]]["step"].add_(1)
    assert optimizer_digest(module, optimizer, scheduler)["tensors"] != first["tensors"]
