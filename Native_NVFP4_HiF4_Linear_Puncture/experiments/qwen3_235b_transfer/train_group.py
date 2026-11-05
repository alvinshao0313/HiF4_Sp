"""Run the fixed full 94-layer Group-top-mass recipe after its real short test."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from Native_NVFP4_HiF4_Linear_Puncture.experiments.non_equivalent_reconstruction.artifact import write_json
from Native_NVFP4_HiF4_Linear_Puncture.experiments.non_equivalent_reconstruction.train import train
from .config import group_config
from .execution import SavedTensorsOnGpus, memory_peaks, reset_memory_peaks


def main(run_root: Path) -> None:
    for relative in ("smoke/e4/status.json", "smoke/group_top_mass/status.json", "e4/status.json"):
        status = json.loads((run_root / relative).read_text())
        if status["status"] != "complete":
            raise ValueError(f"required preceding stage is incomplete: {relative}")
    storage = SavedTensorsOnGpus()
    reset_memory_peaks()
    record = {"status": "running", "execution": storage.metadata(),
              "started_at": datetime.now(timezone.utc).isoformat()}
    write_json(record, run_root / "group_execution.json")
    try:
        with storage:
            train(group_config(run_root))
        record["status"] = "complete"
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record.update(execution=storage.metadata(), gpu_memory_peaks=memory_peaks(),
                      finished_at=datetime.now(timezone.utc).isoformat())
        write_json(record, run_root / "group_execution.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", required=True, type=Path)
    main(parser.parse_args().run_root.resolve())
