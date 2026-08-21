"""Task 7: main.py CLI plumbing for NVFP4 emulation backends / KV dtype."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LIGHTEVAL_SRC = REPO_ROOT / "3rdparty" / "lighteval" / "src"
if str(LIGHTEVAL_SRC) not in sys.path:
    sys.path.insert(0, str(LIGHTEVAL_SRC))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import main as eval_main  # noqa: E402
from lighteval.models.vllm.vllm_model import (  # noqa: E402
    VLLMModel,
    VLLMModelConfig,
)


def test_parse_args_backend_defaults():
    with patch.object(
        sys,
        "argv",
        ["main.py", "--model_path", "dummy-model", "--datasets", "gsm8k"],
    ):
        args = eval_main.parse_args()
    assert args.linear_backend == "auto"
    assert args.moe_backend == "auto"
    assert args.kv_cache_dtype == "auto"


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "bfloat16"])
def test_vllm_model_passes_backend_kwargs_to_llm(kv_cache_dtype: str):
    config = VLLMModelConfig(
        model_name="dummy-model",
        max_model_length=128,
        linear_backend="emulation",
        moe_backend="emulation",
        kv_cache_dtype=kv_cache_dtype,
    )
    with patch("lighteval.models.vllm.vllm_model.LLM") as mock_llm:
        mock_llm.return_value = MagicMock()
        inst = object.__new__(VLLMModel)
        inst._max_length = 128
        inst._create_auto_model(config)

    assert mock_llm.call_count == 1
    kwargs = mock_llm.call_args.kwargs
    assert kwargs["linear_backend"] == "emulation"
    assert kwargs["moe_backend"] == "emulation"
    assert kwargs["kv_cache_dtype"] == kv_cache_dtype


def test_fake_act_nvfp4_conflicts_with_linear_backend_emulation():
    with patch.object(
        sys,
        "argv",
        [
            "main.py",
            "--model_path",
            "dummy-model",
            "--datasets",
            "gsm8k",
            "--fake_act_quant",
            "nvfp4",
            "--linear_backend",
            "emulation",
        ],
    ):
        with pytest.raises(ValueError, match="不能同时启用"):
            eval_main.main()
