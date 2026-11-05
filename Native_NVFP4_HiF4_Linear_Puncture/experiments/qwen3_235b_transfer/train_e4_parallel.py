"""Run an isolated four-GPU E4 candidate while retaining canonical layer stages."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import torch

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.training import moe_trainer
from Native_NVFP4_HiF4_Linear_Puncture.experiments.non_equivalent_reconstruction.artifact import write_json
from .parallel_driver import run_parallel_training
from .parallel_run import make_config, prepare_run, validate_completed


def run(args) -> None:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        raise ValueError("launch with torchrun --nproc-per-node=4")
    options = dict(smoke=args.mode == "smoke", start_layer=args.start_layer,
                   end_layer=args.end_layer, prefix_output=args.prefix_output)
    if rank == 0:
        cfg = prepare_run(args.reference_run_root, args.output, **options)
    else:
        cfg = make_config(args.reference_run_root, args.output, **options)
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.manual_seed(cfg.calib_seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats(device)
    record = {"status": "running", "rank": rank,
              "started_at": datetime.now(timezone.utc).isoformat(),
              "candidate_only": True, "smoke_only": args.mode == "smoke",
              "start_layer": cfg.start_layer, "end_layer": cfg.end_layer,
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "execution": "four ranks; global batch 4; canonical rank-0 nontraining stages"}
    if rank == 0:
        write_json(record, args.output / "status.json")

    original_propagate = moe_trainer._propagate

    def observed_propagate(*positional, **keywords):
        started = time.monotonic()
        runtime = positional[0]
        layer = runtime.state.layer_idx
        print(json.dumps({"event": "propagate_start", "layer": layer,
                          "prefix_replay": layer < cfg.start_layer}), flush=True)
        result = original_propagate(*positional, **keywords)
        print(json.dumps({"event": "propagate_complete", "layer": layer,
                          "prefix_replay": layer < cfg.start_layer,
                          "seconds": time.monotonic() - started}), flush=True)
        return result

    if rank == 0:
        moe_trainer._propagate = observed_propagate
    try:
        run_parallel_training(cfg, device, resume=cfg.start_layer > 0)
        if rank == 0:
            record["validation"] = validate_completed(cfg)
        record["status"] = "complete"
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if rank == 0:
            moe_trainer._propagate = original_propagate
        record.update(finished_at=datetime.now(timezone.utc).isoformat(),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        write_json(record, args.output / f"execution_rank_{rank}.json")
        if rank == 0:
            write_json(record, args.output / "status.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("smoke", "formal"))
    parser.add_argument("--start-layer", required=True, type=int)
    parser.add_argument("--end-layer", required=True, type=int)
    parser.add_argument("--prefix-output", type=Path)
    args = parser.parse_args()
    args.reference_run_root = args.reference_run_root.resolve()
    args.output = args.output.resolve()
    if args.prefix_output is not None:
        args.prefix_output = args.prefix_output.resolve()
    run(args)
