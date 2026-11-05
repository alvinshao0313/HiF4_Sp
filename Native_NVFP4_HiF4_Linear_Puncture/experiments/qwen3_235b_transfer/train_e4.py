"""Run the unchanged DIAG trainer with the verified 235B geometry."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.optim.optimizer import register_optimizer_step_post_hook

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.training.moe_trainer import train_qwen3_moe_lazy
from Native_NVFP4_HiF4_Linear_Puncture.experiments.non_equivalent_reconstruction.artifact import load, write_json
from .config import diag_config
from .execution import SavedTensorsOnGpus, memory_peaks, reset_memory_peaks
from .source import check_source


def train(run_root: Path, *, smoke: bool) -> None:
    source = check_source(run_root, smoke=smoke)
    cfg = diag_config(run_root, smoke=smoke)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("training requires CUDA in hif4")
    torch.manual_seed(cfg.calib_seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    storage = SavedTensorsOnGpus()
    reset_memory_peaks()
    record = {"status": "running", "smoke_only": smoke,
              "started_at": datetime.now(timezone.utc).isoformat(), "source": source,
              "execution": storage.metadata()}
    write_json(record, out / "status.json")
    started = time.monotonic()
    steps = 0

    def record_step(optimizer, args, kwargs):
        nonlocal steps
        steps += 1
        print(json.dumps({"event": "optimizer_step", "optimizer_steps": steps,
                          "lr": optimizer.param_groups[0]["lr"],
                          "elapsed_seconds": round(time.monotonic() - started, 2)}), flush=True)

    hook = register_optimizer_step_post_hook(record_step)
    try:
        with storage:
            train_qwen3_moe_lazy(cfg, torch.device("cuda:0"))
        artifact = load(out / "checkpoint/final_model/conversion_state.pt")
        if artifact["num_layers"] != 94 or set(artifact["layers"]) != {str(i) for i in range(cfg.end_layer + 1)}:
            raise ValueError("saved artifact layer coverage differs from the run configuration")
        record.update(status="complete", peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        hook.remove()
        record["optimizer_steps"] = steps
        record["execution"] = storage.metadata()
        record["gpu_memory_peaks"] = memory_peaks()
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(record, out / "status.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("smoke", "formal"))
    args = parser.parse_args()
    train(args.run_root.resolve(), smoke=args.mode == "smoke")
