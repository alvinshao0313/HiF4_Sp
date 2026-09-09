from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class HookCaptureState:
    variant: str
    rank: int
    world_size: int
    output_root: str
    capture_level: str = "core"
    sample_key: str | None = None
    prompt_len: int = 0
    probe_abs_to_decode: dict[int, int] = field(default_factory=dict)
    probe_decode_indices: set[int] = field(default_factory=set)
    prefill_done: bool = False
    next_decode_index: int = 1
    active_rows: list[int] = field(default_factory=list)
    active_decode_indices: list[int] = field(default_factory=list)
    current_layer: int | None = None
    records: list[dict] = field(default_factory=list)
    fire_counts: dict[str, int] = field(default_factory=dict)

    @property
    def output_path(self) -> Path:
        if self.sample_key is None:
            raise RuntimeError("sample is not active")
        return Path(self.output_root) / self.variant / self.sample_key / f"rank{self.rank}.pt"

    def begin_sample(
        self,
        sample_key: str,
        prompt_len: int,
        probe_abs_to_decode: dict[int, int],
    ) -> None:
        if self.sample_key is not None:
            raise RuntimeError(f"sample already active: {self.sample_key}")
        self.sample_key = str(sample_key)
        self.prompt_len = int(prompt_len)
        self.probe_abs_to_decode = {int(k): int(v) for k, v in probe_abs_to_decode.items()}
        self.probe_decode_indices = set(self.probe_abs_to_decode.values())
        self.prefill_done = False
        self.next_decode_index = 1
        self.active_rows.clear()
        self.active_decode_indices.clear()
        self.current_layer = None
        self.records.clear()
        self.fire_counts.clear()

    def clear_sample(self) -> None:
        self.sample_key = None
        self.prompt_len = 0
        self.probe_abs_to_decode.clear()
        self.probe_decode_indices.clear()
        self.prefill_done = False
        self.next_decode_index = 1
        self.active_rows.clear()
        self.active_decode_indices.clear()
        self.current_layer = None
        self.records.clear()
        self.fire_counts.clear()
