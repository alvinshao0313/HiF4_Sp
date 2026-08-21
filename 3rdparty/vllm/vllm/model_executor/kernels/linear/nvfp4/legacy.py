# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapter that reuses the pre-v0.27 nvfp4_utils native GEMM path.

Used for ``linear_backend="auto"`` (and explicit native backend names) so we
do not need to backport Cutlass/Marlin/FlashInfer NVFP4 kernel classes.
Explicit ``linear_backend="emulation"`` must use ``EmulationNvFp4LinearKernel``
instead of this adapter.
"""

from __future__ import annotations

import torch

from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    NvFp4LinearBackend,
    apply_nvfp4_linear,
    convert_to_nvfp4_linear_kernel_format,
)

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig


class LegacyNvFp4LinearKernel(NvFp4LinearKernel):
    """Wraps ``NvFp4LinearBackend`` enum + ``nvfp4_utils`` apply/convert."""

    def __init__(
        self,
        config: NvFp4LinearLayerConfig,
        backend: NvFp4LinearBackend,
    ) -> None:
        if backend == NvFp4LinearBackend.EMULATION:
            raise ValueError(
                "LegacyNvFp4LinearKernel must not be used for emulation; "
                "use EmulationNvFp4LinearKernel instead."
            )
        # Bypass base asserts that call cls.is_supported without backend.
        self.config = config
        self.backend = backend

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        return True, None

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        convert_to_nvfp4_linear_kernel_format(self.backend, layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return apply_nvfp4_linear(
            backend=self.backend,
            layer=layer,
            x=x,
            bias=bias,
        )
