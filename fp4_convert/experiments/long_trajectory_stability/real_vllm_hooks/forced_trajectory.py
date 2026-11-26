from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from vllm import SamplingParams
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor, RequestLogitsProcessor


FORCED_IDS_KEY = "hif4_forced_token_ids"
RELEASE_AFTER_PREFIX_KEY = "hif4_release_after_prefix"
CAPTURE_SAMPLE_KEY = "hif4_capture_sample_key"
CAPTURE_VARIANT_KEY = "hif4_capture_variant"
CAPTURE_PROBES_KEY = "hif4_capture_probe_decode_indices"
CAPTURE_LOGITS_ROOT_KEY = "hif4_capture_logits_root"


class _ForcedTrajectoryRequestProcessor:
    def __init__(
        self,
        forced_token_ids: list[int],
        *,
        sample_key: str | None,
        variant: str | None,
        probe_decode_indices: set[int],
        logits_root: str | None,
        release_after_prefix: bool = False,
    ) -> None:
        if not forced_token_ids:
            raise ValueError("forced_token_ids must not be empty")
        self.forced_token_ids = [int(x) for x in forced_token_ids]
        self.sample_key = sample_key
        self.variant = variant
        self.probe_decode_indices = {int(x) for x in probe_decode_indices}
        self.logits_root = logits_root
        self.release_after_prefix = bool(release_after_prefix)
        self.rank = int(get_tensor_model_parallel_rank())
        if self.probe_decode_indices and not all((sample_key, variant, logits_root)):
            raise ValueError("raw-logit capture requires sample_key, variant and logits_root")

    def _capture_raw_logits(self, step: int, logits: torch.Tensor) -> None:
        if step not in self.probe_decode_indices:
            return
        assert self.sample_key is not None
        assert self.variant is not None
        assert self.logits_root is not None
        out = Path(self.logits_root) / self.variant / self.sample_key
        out.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "variant": self.variant,
                "sample_key": self.sample_key,
                "tp_rank": self.rank,
                "decode_index": int(step),
                "dtype": str(logits.dtype),
                "shape": tuple(int(x) for x in logits.shape),
                "logits": logits.detach().to(device="cpu").clone(),
            },
            out / f"rank{self.rank}_decode{step}.pt",
        )

    def __call__(self, output_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
        step = len(output_ids)
        if step >= len(self.forced_token_ids):
            if self.release_after_prefix:
                self._capture_raw_logits(step, logits)
                return logits
            raise RuntimeError(
                f"forced trajectory exhausted: step={step} len={len(self.forced_token_ids)}"
            )
        target = self.forced_token_ids[step]
        if logits.ndim != 1:
            raise RuntimeError(f"request logits must be 1D, got shape={tuple(logits.shape)}")
        if not 0 <= target < logits.numel():
            raise RuntimeError(f"forced token id out of range: {target} vs vocab={logits.numel()}")
        self._capture_raw_logits(step, logits)
        keep = logits[target].clone()
        logits.fill_(float("-inf"))
        logits[target] = keep
        return logits


class ForcedTrajectoryLogitsProcessor(AdapterLogitsProcessor):
    @classmethod
    def validate_params(cls, params: SamplingParams) -> None:
        extra = params.extra_args or {}
        value: Any | None = extra.get(FORCED_IDS_KEY)
        if value is None:
            return
        if not isinstance(value, list) or not value or not all(isinstance(x, int) for x in value):
            raise ValueError(f"{FORCED_IDS_KEY} must be a non-empty list[int]")
        if not isinstance(extra.get(RELEASE_AFTER_PREFIX_KEY, False), bool):
            raise ValueError(f"{RELEASE_AFTER_PREFIX_KEY} must be bool")
        probes = extra.get(CAPTURE_PROBES_KEY, [])
        if not isinstance(probes, list) or not all(isinstance(x, int) and x >= 0 for x in probes):
            raise ValueError(f"{CAPTURE_PROBES_KEY} must be list[non-negative int]")

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(
        self, params: SamplingParams
    ) -> RequestLogitsProcessor | None:
        extra = params.extra_args or {}
        value = extra.get(FORCED_IDS_KEY)
        if value is None:
            return None
        return _ForcedTrajectoryRequestProcessor(
            value,
            sample_key=extra.get(CAPTURE_SAMPLE_KEY),
            variant=extra.get(CAPTURE_VARIANT_KEY),
            probe_decode_indices=set(extra.get(CAPTURE_PROBES_KEY, [])),
            logits_root=extra.get(CAPTURE_LOGITS_ROOT_KEY),
            release_after_prefix=extra.get(RELEASE_AFTER_PREFIX_KEY, False),
        )


def make_forced_sampling_params(
    forced_token_ids: list[int],
    *,
    max_tokens: int | None = None,
    sample_key: str | None = None,
    variant: str | None = None,
    probe_decode_indices: list[int] | None = None,
    logits_root: str | None = None,
) -> SamplingParams:
    ids = [int(x) for x in forced_token_ids]
    if not ids:
        raise ValueError("forced_token_ids must not be empty")
    n = len(ids) if max_tokens is None else min(int(max_tokens), len(ids))
    if n <= 0:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")
    ids = ids[:n]
    probes = sorted({int(x) for x in (probe_decode_indices or []) if int(x) < n})
    extra: dict[str, Any] = {FORCED_IDS_KEY: ids}
    if probes:
        if not sample_key or not variant or not logits_root:
            raise ValueError("probe logits capture requires sample_key, variant and logits_root")
        extra.update(
            {
                CAPTURE_SAMPLE_KEY: str(sample_key),
                CAPTURE_VARIANT_KEY: str(variant),
                CAPTURE_PROBES_KEY: probes,
                CAPTURE_LOGITS_ROOT_KEY: str(logits_root),
            }
        )
    return SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=n,
        ignore_eos=True,
        extra_args=extra,
    )
