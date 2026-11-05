"""CLI checks that do not start a model or require a calibration file."""
from pathlib import Path
import subprocess
import sys

import pytest


REPO = Path(__file__).resolve().parents[4]
MODULE = "Native_NVFP4_HiF4_Linear_Puncture.experiments.ban.reasoning_main"


def run(*args):
    return subprocess.run([sys.executable, "-m", MODULE, *args], cwd=REPO,
                          capture_output=True, text=True, check=False)


def test_default_none_does_not_require_calibration():
    result = run("--ban_artifact_path", "/nonexistent/calibration.pt", "--help")
    assert result.returncode == 0
    assert "--expert_pruning_method" in result.stdout
    assert "--hif4_runtime_spec" in result.stdout


def test_enabled_ban_requires_calibration():
    result = run("--expert_pruning_method", "Ban")
    assert result.returncode == 2
    assert "requires --ban_artifact_path" in result.stderr


@pytest.mark.parametrize("dp", [["--data_parallel_size", "2"], ["--data_parallel_size=2"]])
def test_ban_rejects_ray_path_before_loading_model(dp):
    result = run("--expert_pruning_method", "Ban", "--ban_artifact_path", "/nonexistent/calibration.pt",
                 "--model_path=/nonexistent/model", "--hif4_runtime_spec=/nonexistent/spec.pt", *dp)
    assert result.returncode == 2
    assert "data_parallel_size=1" in result.stderr
