from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class PrefixStats:
    calls: int = 0
    tokens: int = 0
    kb: int = 0
    keep_count: int = 0
    realized_keep_sum: float = 0.0
    predictor_time_s: float = 0.0
    mask_apply_time_s: float = 0.0
    debug_masks: list[list[list[bool]]] = field(default_factory=list)


class DynamicInputTelemetry:
    def __init__(self, debug_first_masks: int = 0) -> None:
        self.debug_first_masks = int(debug_first_masks)
        self._stats: dict[str, PrefixStats] = {}

    def record(
        self,
        prefix: str,
        *,
        tokens: int,
        kb: int,
        keep_count: int,
        realized_keep: float,
        predictor_time_s: float,
        mask_apply_time_s: float,
        mx: torch.Tensor | None = None,
    ) -> None:
        st = self._stats.setdefault(prefix, PrefixStats())
        st.calls += 1
        st.tokens += int(tokens)
        st.kb = int(kb)
        st.keep_count = int(keep_count)
        st.realized_keep_sum += float(realized_keep) * int(tokens)
        st.predictor_time_s += float(predictor_time_s)
        st.mask_apply_time_s += float(mask_apply_time_s)
        if (
            self.debug_first_masks > 0
            and mx is not None
            and len(st.debug_masks) < self.debug_first_masks
        ):
            take = min(int(mx.shape[0]), self.debug_first_masks - len(st.debug_masks))
            st.debug_masks.extend(mx[:take].detach().cpu().tolist())

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for prefix, st in sorted(self._stats.items()):
            realized = (
                st.realized_keep_sum / st.tokens if st.tokens > 0 else 0.0
            )
            out[prefix] = {
                "calls": st.calls,
                "tokens": st.tokens,
                "kb": st.kb,
                "keep_count": st.keep_count,
                "realized_keep_ratio": realized,
                "predictor_time_s": st.predictor_time_s,
                "mask_apply_time_s": st.mask_apply_time_s,
                "debug_masks": st.debug_masks,
            }
        return out

    def dump(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


_GLOBAL: DynamicInputTelemetry | None = None
_DUMP_PATH: str | None = None
_ATEXIT_REGISTERED = False


def get_telemetry() -> DynamicInputTelemetry | None:
    return _GLOBAL


def enable_telemetry(
    debug_first_masks: int = 0,
    dump_path: str | None = None,
) -> DynamicInputTelemetry:
    """Enable process-local telemetry; optionally atexit-dump to dump_path."""
    global _GLOBAL, _DUMP_PATH, _ATEXIT_REGISTERED
    if _GLOBAL is None:
        _GLOBAL = DynamicInputTelemetry(debug_first_masks=debug_first_masks)
    if dump_path:
        _DUMP_PATH = str(dump_path)
        if not _ATEXIT_REGISTERED:
            import atexit

            atexit.register(_atexit_dump)
            _ATEXIT_REGISTERED = True
    return _GLOBAL


def _atexit_dump() -> None:
    if _GLOBAL is None or not _DUMP_PATH:
        return
    try:
        _GLOBAL.dump(_DUMP_PATH)
    except Exception:
        pass


def disable_telemetry() -> None:
    global _GLOBAL
    _GLOBAL = None


def maybe_enable_from_additional(additional_config: dict | None) -> None:
    """Worker-side enable when telemetry_dir is present in additional_config."""
    cfg = additional_config or {}
    tele_dir = str(cfg.get("dynamic_input_telemetry_dir", "") or "")
    method = str(cfg.get("dynamic_input_sparse_method", "none"))
    if method == "none" or not tele_dir:
        return
    from pathlib import Path

    Path(tele_dir).mkdir(parents=True, exist_ok=True)
    enable_telemetry(
        debug_first_masks=16,
        dump_path=str(Path(tele_dir) / "telemetry.json"),
    )


class _Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.t0
