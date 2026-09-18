import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from Native_NVFP4_HiF4_Linear_Puncture.experiments.kl_direction_reuse import runner
from Native_NVFP4_HiF4_Linear_Puncture.experiments.kl_direction_reuse.artifacts import read_json
from Native_NVFP4_HiF4_Linear_Puncture.experiments.kl_direction_reuse.host import allocation


def test_background_failure_records_child_and_stops_queue(tmp_path, monkeypatch):
    root, control = tmp_path / "run", tmp_path / "control"
    monkeypatch.setenv("KLD_CONTROL_DIR", str(control))
    monkeypatch.delenv("KLD_EXPECTED_SOURCE", raising=False)
    original = subprocess.Popen
    calls = []
    def child(argv, **kwargs):
        assert not root.exists(), "controller must not contaminate the empty prepare root"
        calls.append(argv)
        return original([sys.executable, "-c", "print('deliberate stage failure'); raise SystemExit(3)"], **kwargs)
    monkeypatch.setattr(subprocess, "Popen", child)
    with pytest.raises(subprocess.CalledProcessError) as error:
        runner.suite(root, "prepare", "2", "2,3")
    assert error.value.returncode == 3 and len(calls) == 1
    current = read_json(control / "current.json")
    assert current["status"] == "FAILED" and current["pid"] > 0
    assert current["returncode"] == 3
    assert "deliberate stage failure" in open(current["log"]).read()
    ledger = [json.loads(line) for line in (root / "stages.jsonl").read_text().splitlines()]
    assert len(ledger) == 1 and ledger[0]["command"] == "prepare"


def test_child_receives_only_explicit_gpu_allocation(tmp_path, monkeypatch):
    root, control = tmp_path / "run", tmp_path / "control"
    monkeypatch.setenv("KLD_CONTROL_DIR", str(control))
    monkeypatch.delenv("KLD_EXPECTED_SOURCE", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    monkeypatch.setattr(runner, "idle_devices", lambda ids: [{"index": ids}])
    original = subprocess.Popen
    def child(argv, **kwargs):
        return original([sys.executable, "-c", "import os; print(os.environ['CUDA_VISIBLE_DEVICES'])"], **kwargs)
    monkeypatch.setattr(subprocess, "Popen", child)
    runner.invoke(root, "train", "--recipe", "L08_mse", gpu_ids="2")
    current = read_json(control / "current.json")
    assert current["status"] == "COMPLETE" and current["devices"] == "2"
    assert open(current["log"]).read().strip() == "2"


def test_busy_requested_gpus_fail_without_touching_other_jobs(monkeypatch):
    def query(argv, **kwargs):
        if "--query-gpu=" in argv[1]:
            return "2, GPU-two, A800, 81920\n3, GPU-three, A800, 81920\n"
        return "GPU-three, 123, another_job\n"
    monkeypatch.setattr(subprocess, "check_output", query)
    with pytest.raises(RuntimeError, match="active processes"):
        allocation("2", "2,3")
    with pytest.raises(ValueError, match="one of those"):
        allocation("0", "2,3")


def test_busy_stage_is_rejected_before_starting_a_child(tmp_path, monkeypatch):
    control = tmp_path / "control"
    monkeypatch.setenv("KLD_CONTROL_DIR", str(control))
    monkeypatch.delenv("KLD_EXPECTED_SOURCE", raising=False)
    def busy(ids):
        raise RuntimeError("requested GPUs have active processes")
    def no_child(*args, **kwargs):
        pytest.fail("a child must not be created on an occupied GPU")
    monkeypatch.setattr(runner, "idle_devices", busy)
    monkeypatch.setattr(subprocess, "Popen", no_child)
    with pytest.raises(RuntimeError, match="active processes"):
        runner.invoke(tmp_path / "run", "capture", "--variant", "native", gpu_ids="2,3")
    status = read_json(control / "current.json")
    assert status["status"] == "FAILED" and status["pid"] is None


def test_capture_chooses_spawn_before_cuda_discovery(monkeypatch, tmp_path):
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.kl_direction_reuse import capture
    from vllm.utils.system_utils import get_mp_context
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "fork")
    def stop_before_cuda(count):
        assert count == 2
        assert get_mp_context().get_start_method() == "spawn"
        raise RuntimeError("test ended before CUDA discovery")
    monkeypatch.setattr(capture, "require_gpus", stop_before_cuda)
    with pytest.raises(RuntimeError, match="test ended"):
        capture.capture(tmp_path, "native")


def test_after_baseline_continues_all_remaining_stages(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "validate_after_baseline", lambda root: calls.append(("validated",)))
    monkeypatch.setattr(runner, "invoke", lambda root, command, *args, **kwargs: calls.append((command, *args)))
    monkeypatch.setattr(runner, "read_json", lambda path: {"probe_model": "model", "samples": ["sample"]})
    monkeypatch.setattr(runner, "evaluate_suite", lambda root, gpus: calls.append(("evaluate", gpus)))
    runner.suite(tmp_path, "all", "2", "2,3", after_baseline=True)
    assert calls[:3] == [("validated",), ("capture", "--variant", "native"), ("capture", "--variant", "baseline")]
    assert not any(c[0] in {"prepare", "baseline"} for c in calls)
    assert sum(c[0] == "verify-finish" for c in calls) == 3
    assert sum(c[0] == "train" for c in calls) == 15
    assert calls[-2:] == [("evaluate", "2,3"), ("report",)]


def test_after_baseline_rejects_existing_capture_outputs(tmp_path, monkeypatch):
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.kl_direction_reuse import materialize
    monkeypatch.setattr(runner, "Dataset", lambda root: SimpleNamespace(root=tmp_path, snapshot=tmp_path / "native"))
    monkeypatch.setattr(materialize, "check_baseline", lambda root: {"source_files": {"weight": "hash"}})
    monkeypatch.setattr(materialize, "model_identity", lambda root: {"weight": "hash"})
    assert runner.validate_after_baseline(tmp_path)["source_files"] == {"weight": "hash"}
    (tmp_path / "captures/native").mkdir(parents=True)
    with pytest.raises(FileExistsError, match="downstream output"):
        runner.validate_after_baseline(tmp_path)


def test_after_captures_runs_verification_and_all_training(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, 'validate_after_captures', lambda root: calls.append(('validated',)))
    monkeypatch.setattr(runner, 'invoke', lambda root, command, *args, **kwargs: calls.append((command, *args)))
    monkeypatch.setattr(runner, 'read_json', lambda path: {'probe_model': 'model', 'samples': ['sample']})
    monkeypatch.setattr(runner, 'evaluate_suite', lambda root, gpus: calls.append(('evaluate', gpus)))
    runner.suite(tmp_path, 'all', '6', '6,7', after_captures=True)
    assert calls[0] == ('validated',)
    assert calls[1] == ('verify-prepare', '--layer', 8)
    assert sum(c[0] == 'capture' for c in calls) == 3
    assert sum(c[0] == 'verify-finish' for c in calls) == 3
    assert sum(c[0] == 'train' for c in calls) == 15
    assert calls[-2:] == [('evaluate', '6,7'), ('report',)]
