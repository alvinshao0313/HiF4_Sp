# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused NVFP4 linear kernel selection (v0.27.0 emulation backport).

Kept separate from ``kernels.linear.__init__`` so ModelOpt / CT / tests can
select the emulation kernel without importing the full mixed-precision stack.
"""

from __future__ import annotations

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.nvfp4.base import (
    NvFp4LinearKernel,
    NvFp4LinearLayerConfig,
)
from vllm.model_executor.kernels.linear.nvfp4.emulation import (
    EmulationNvFp4LinearKernel,
)

logger = init_logger(__name__)


def _get_linear_backend() -> str:
    """Get the linear_backend setting from the current vllm config."""
    from vllm.config import get_current_vllm_config_or_none

    config = get_current_vllm_config_or_none()
    if config is not None:
        return config.kernel_config.linear_backend
    return "auto"


def init_nvfp4_linear_kernel(use_a16: bool = False) -> NvFp4LinearKernel:
    """Select and instantiate an NVFP4 linear kernel.

    Focused backport of v0.27.0 selection:
    - ``linear_backend="emulation"`` (or legacy env
      ``VLLM_USE_NVFP4_CT_EMULATIONS`` under auto) →
      ``EmulationNvFp4LinearKernel`` (never Marlin).
    - ``auto`` / other native names → ``LegacyNvFp4LinearKernel`` wrapping
      the existing ``nvfp4_utils`` Cutlass/FlashInfer/Marlin path.
    """
    config = NvFp4LinearLayerConfig()
    linear_backend = _get_linear_backend()

    # Compat: old env-var emulation entry under auto.
    if linear_backend == "auto" and envs.VLLM_USE_NVFP4_CT_EMULATIONS:
        linear_backend = "emulation"

    if linear_backend == "emulation":
        if use_a16:
            raise ValueError(
                "EmulationNvFp4LinearKernel does not support W4A16; "
                "requested linear_backend=emulation with use_a16=True."
            )
        logger.info_once("Using EmulationNvFp4LinearKernel for NVFP4 GEMM")
        return EmulationNvFp4LinearKernel(config)

    # Native / auto: reuse pre-existing nvfp4_utils selection (lazy import).
    from vllm.model_executor.kernels.linear.nvfp4.legacy import (
        LegacyNvFp4LinearKernel,
    )
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
        NvFp4LinearBackend,
        select_nvfp4_linear_backend,
    )

    if linear_backend == "auto":
        if use_a16:
            backend = NvFp4LinearBackend.MARLIN
            logger.info_once("Using %s for NVFP4 GEMM (W4A16)", backend)
        else:
            backend = select_nvfp4_linear_backend()
    else:
        backend_name_map = {
            "cutlass": NvFp4LinearBackend.VLLM_CUTLASS,
            "flashinfer_cutlass": NvFp4LinearBackend.FLASHINFER_CUTLASS,
            "flashinfer_trtllm": NvFp4LinearBackend.FLASHINFER_TRTLLM,
            "flashinfer_cudnn": NvFp4LinearBackend.FLASHINFER_CUDNN,
            "marlin": NvFp4LinearBackend.MARLIN,
            "fbgemm": NvFp4LinearBackend.FBGEMM,
        }
        if linear_backend not in backend_name_map:
            raise ValueError(
                f"--linear-backend={linear_backend} was requested but no "
                f"'{linear_backend}' kernel exists for NVFP4 layers."
            )
        if use_a16 and backend_name_map[linear_backend] != NvFp4LinearBackend.MARLIN:
            raise ValueError(
                f"{backend_name_map[linear_backend]} does not support W4A16"
            )
        backend = backend_name_map[linear_backend]
        logger.info_once("Using %s for NVFP4 GEMM", backend)

    if backend == NvFp4LinearBackend.EMULATION:
        if use_a16:
            raise ValueError(
                "EmulationNvFp4LinearKernel does not support W4A16."
            )
        logger.info_once("Using EmulationNvFp4LinearKernel for NVFP4 GEMM")
        return EmulationNvFp4LinearKernel(config)

    return LegacyNvFp4LinearKernel(config, backend)


# Alias matching upstream naming in call sites / docs.
select_nvfp4_linear_kernel = init_nvfp4_linear_kernel
