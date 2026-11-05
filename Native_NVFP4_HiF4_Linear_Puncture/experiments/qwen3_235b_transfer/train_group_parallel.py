"""Continue a completed 235B E4 artifact with the fixed Group-top-mass recipe."""
from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import torch
from torch.optim.optimizer import register_optimizer_step_post_hook

from Native_NVFP4_HiF4_Linear_Puncture.experiments.non_equivalent_reconstruction.artifact import (
    atomic_save, load, load_initialization, sha256, write_json,
)
from .config import group_config
from .execution import SavedTensorsOnGpus, memory_peaks
from . import smoke_group


OUTPUT_NAMES = {
    "reference-smoke": "group_smoke_reference",
    "smoke": "group_smoke_parallel",
    "checkpoint-smoke": "group_smoke_checkpoint",
    "formal": "group_top_mass",
}


def configuration(run_root: Path, mode: str):
    cfg = dataclasses.replace(group_config(run_root),
                              calib_cache_dir=str(run_root / "e4/data"),
                              output_dir=str(run_root / OUTPUT_NAMES[mode]))
    if mode != "formal":
        provenance = json.loads((run_root / "e4/provenance.json").read_text())
        smoke_data = Path(provenance["reference_run_root"]) / "smoke/data"
        cfg = dataclasses.replace(cfg, epochs=1 if mode == "checkpoint-smoke" else 2,
                                  calib_nsamples=4, calib_val_nsamples=4,
                                  calib_cache_dir=str(smoke_data))
    cfg.validate()
    return cfg


def run(run_root: Path, mode: str) -> None:
    parallel = mode != "reference-smoke"
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if parallel and int(os.environ.get("WORLD_SIZE", "1")) != 4:
        raise ValueError("parallel Group requires torchrun --nproc-per-node=4")
    cfg = configuration(run_root, mode)
    out = Path(cfg.output_dir)
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.manual_seed(cfg.calib_seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    record = {"status": "running", "rank": rank, "mode": mode,
              "smoke_only": mode != "formal", "started_at": datetime.now(timezone.utc).isoformat(),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "initialization": cfg.init_artifact, "global_batch_size": cfg.batch_size,
              "epochs": cfg.epochs, "calibration_samples": [cfg.calib_nsamples, cfg.calib_val_nsamples]}
    if rank == 0:
        e4_status = json.loads((run_root / "e4/status.json").read_text())
        if e4_status["status"] != "complete" or e4_status["validation"]["status"] != "passed":
            raise ValueError("completed and validated E4 artifact is required")
        digest = sha256(cfg.init_artifact)
        if digest != e4_status["validation"]["final_artifact_sha256"]:
            raise ValueError("completed E4 artifact hash changed")
        load_initialization(cfg.init_artifact, cfg.model_path)
        record["initialization_sha256"] = digest
        if mode == "formal":
            for name in ("group_smoke_reference", "group_smoke_parallel", "group_smoke_checkpoint"):
                status = json.loads((run_root / name / "status.json").read_text())
                if status["status"] != "complete" or status["initialization_sha256"] != digest:
                    raise ValueError(f"required Group check is incomplete or has different E4 input: {name}")
        out.mkdir(parents=True, exist_ok=False)
        write_json(cfg.to_dict(), out / "config.json")
        write_json(record, out / "status.json")

    hook = None
    step_count = 0

    def observe_first_update(optimizer, _args, _kwargs):
        nonlocal step_count
        step_count += 1
        if step_count == 1:
            parameters = [p for group in optimizer.param_groups for p in group["params"]]
            atomic_save({"gradients": [None if p.grad is None else p.grad.detach().cpu().clone()
                                       for p in parameters],
                         "parameters": [p.detach().cpu().clone() for p in parameters]},
                        out / "first_update.pt")

    original_reset = smoke_group.reset_memory_peaks
    storage = None
    try:
        if rank == 0 and mode in ("reference-smoke", "smoke"):
            hook = register_optimizer_step_post_hook(observe_first_update)
        if parallel:
            from .group_parallel_driver import run_parallel_group
            smoke_group.reset_memory_peaks = lambda: torch.cuda.reset_peak_memory_stats(device)
            runner = None
            if mode == "smoke":
                def runner():
                    smoke_group.execute(cfg, load(cfg.init_artifact), out, record)
            elif mode == "checkpoint-smoke":
                from .group_checkpoint_smoke import run_checkpoint_smoke
                def runner():
                    run_checkpoint_smoke(cfg, record)
            run_parallel_group(cfg, device, runner=runner)
        else:
            storage = SavedTensorsOnGpus()
            with storage:
                smoke_group.execute(cfg, load(cfg.init_artifact), out, record)
        if rank == 0 and mode == "formal":
            summary = json.loads((out / "summary.json").read_text())
            manifest = json.loads((out / "manifest.json").read_text())
            if (summary.get("complete") is not True or len(summary["layers"]) != 94
                    or manifest["completed_layers"] != list(range(94))):
                raise ValueError("Group final artifact does not cover all 94 layers")
            record["completed_layers"] = 94
        record["status"] = "complete"
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if hook is not None:
            hook.remove()
        smoke_group.reset_memory_peaks = original_reset
        record.update(finished_at=datetime.now(timezone.utc).isoformat(),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        if storage is not None:
            record.update(execution=storage.metadata(), gpu_memory_peaks=memory_peaks())
        write_json(record, out / f"execution_rank_{rank}.json")
        if rank == 0:
            record["observed_smoke_updates"] = step_count if mode != "formal" else None
            write_json(record, out / "status.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=tuple(OUTPUT_NAMES))
    args = parser.parse_args()
    run(args.run_root.resolve(), args.mode)
