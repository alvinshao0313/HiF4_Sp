# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from pydantic import ValidationError

from vllm.config.kernel import KernelConfig


def test_kernel_config_accepts_linear_backend_emulation():
    cfg = KernelConfig(linear_backend="emulation")
    assert cfg.linear_backend == "emulation"


def test_kernel_config_accepts_moe_backend_emulation():
    cfg = KernelConfig(moe_backend="emulation")
    assert cfg.moe_backend == "emulation"


@pytest.mark.parametrize(
    ("field", "raw", "expected"),
    [
        ("linear_backend", "Emulation", "emulation"),
        ("linear_backend", "EMULATION", "emulation"),
        ("linear_backend", "emulation", "emulation"),
        ("moe_backend", "Emulation", "emulation"),
        ("moe_backend", "EMULATION", "emulation"),
        ("moe_backend", "emulation", "emulation"),
    ],
)
def test_kernel_config_backend_case_normalization(field, raw, expected):
    cfg = KernelConfig(**{field: raw})
    assert getattr(cfg, field) == expected


@pytest.mark.parametrize(
    ("field", "raw", "expected"),
    [
        ("linear_backend", "flashinfer-cutlass", "flashinfer_cutlass"),
        ("moe_backend", "flashinfer-trtllm", "flashinfer_trtllm"),
    ],
)
def test_kernel_config_backend_hyphen_normalization(field, raw, expected):
    cfg = KernelConfig(**{field: raw})
    assert getattr(cfg, field) == expected


def test_kernel_config_rejects_unknown_linear_backend_alias():
    with pytest.raises(ValidationError):
        KernelConfig(linear_backend="nvfp4_emulation")


def test_kernel_config_rejects_unknown_moe_backend_alias():
    with pytest.raises(ValidationError):
        KernelConfig(moe_backend="nvfp4_emulation")
