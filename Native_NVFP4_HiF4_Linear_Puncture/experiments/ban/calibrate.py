"""Compute Ban calibration on the real HiF4 vLLM path."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import time

import torch

from .algorithm import SOURCE_COMMIT, SOURCE_URL, normalize_layer_kl
from .artifact import model_identity, sha256, write_json
from .data import sample_c4
from .diagnose import generate
from .engine import configure, create_engine, snapshot


def calibrate(args, engine=None, samples=None, data=None):
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    identity = model_identity(args.model_path, args.hif4_runtime_spec)
    if not 1 <= args.k_pruned <= identity["top_k"] or args.topn < 1:
        raise ValueError("invalid calibration k_pruned/topn")
    if samples is None:
        samples, data = sample_c4(args.model_path, args.c4_arrow, args.nsamples, args.seqlen, args.seed)
    torch.save(samples, root / "input_ids.pt")
    manifest = {"status": "running", "phase": args.phase, "model_identity": identity,
                "model_path": str(Path(args.model_path).resolve()), "spec_path": str(Path(args.hif4_runtime_spec).resolve()),
                "data": data, "source_commit": SOURCE_COMMIT, "source_url": SOURCE_URL,
                "k_pruned": args.k_pruned, "topn": args.topn, "kl_positions": "last",
                "tensor_parallel_size": args.tensor_parallel_size, "command": sys.argv,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "environment": {"python": sys.version, "torch": torch.__version__},
                "code_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}}
    write_json(root / "manifest.json", manifest)
    started = time.monotonic()
    try:
        if engine is None:
            engine = create_engine(args.model_path, args.hif4_runtime_spec, args.tensor_parallel_size,
                                   max_model_len=max(4096, args.seqlen + 1))
        engine.collective_rpc("ban_context")
        raw_sum = torch.zeros(identity["num_layers"], dtype=torch.float64)
        sample_values, r_min, r_max = [], float("inf"), -float("inf")
        for sample_idx, ids in enumerate(samples):
            configure(engine, mode="collect", capture="reference", topn=args.topn)
            generate(engine, ids)
            native = snapshot(engine)
            if native["tokens"] != [args.seqlen] * identity["num_layers"] or native["capture"] is None:
                raise RuntimeError("incomplete reference forward or missing last-position logits")
            r_min, r_max = min(r_min, min(native["ratio_min"])), max(r_max, max(native["ratio_max"]))
            values = []
            for layer in range(identity["num_layers"]):
                configure(engine, mode="single_layer", layer=layer, k_pruned=args.k_pruned,
                          capture="kl", execution="compact", topn=args.topn)
                generate(engine, ids)
                observation = snapshot(engine)
                if (observation["capture"] is None or observation["dispatched"][layer] != args.seqlen * args.k_pruned
                        or observation["tokens"] != [args.seqlen] * identity["num_layers"]):
                    raise RuntimeError("single-layer calibration/capture did not execute correctly")
                values.append(observation["capture"]["kl"])
                if (layer + 1) % 8 == 0:
                    print(f"BAN_CALIB sample={sample_idx + 1}/{len(samples)} layer={layer + 1}/{identity['num_layers']}", flush=True)
            raw_sum += torch.tensor(values, dtype=torch.float64)
            sample_values.append(values)
            write_json(root / "progress.json", {"completed_samples": sample_idx + 1, "total_samples": len(samples),
                       "elapsed_seconds": time.monotonic() - started, "r_min": r_min, "r_max": r_max,
                       "per_sample_layer_kl": sample_values})
        raw_kl = (raw_sum / len(samples)).float()
        if not torch.isfinite(raw_kl).all() or not 0 <= r_min <= r_max <= 1 + 1e-6:
            raise RuntimeError("invalid layer KL or router mass ratio")
        if r_max <= r_min:
            r_max = r_min + 1e-6
        artifact = {"format": "hif4_ban_v1", "source_commit": SOURCE_COMMIT, "model_identity": identity,
                    "phase": args.phase, "layer_sensitivity": normalize_layer_kl(raw_kl), "raw_layer_kl": raw_kl,
                    "r_min": r_min, "r_max": r_max, "k_base": identity["top_k"], "k_pruned": args.k_pruned,
                    "topn": args.topn, "kl_positions": "last", "num_samples": len(samples), "data": data}
        path = root / "c4_ban.pt"
        torch.save(artifact, path)
        write_json(root / "c4_ban.json", {k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in artifact.items()})
        manifest.update(status="complete", artifact_path=str(path), artifact_sha256=sha256(path),
                        elapsed_seconds=time.monotonic() - started)
        write_json(root / "manifest.json", manifest)
        print(f"BAN_CALIB_COMPLETE {path}", flush=True)
        return engine, artifact, samples
    except Exception as error:
        manifest.update(status="failed", error=str(error), elapsed_seconds=time.monotonic() - started)
        write_json(root / "manifest.json", manifest)
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model_path", required=True)
    result.add_argument("--hif4_runtime_spec", required=True)
    result.add_argument("--c4_arrow", nargs="+", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--nsamples", type=int, default=512)
    result.add_argument("--seqlen", type=int, default=2048)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--k_pruned", type=int, default=3)
    result.add_argument("--topn", type=int, default=1000)
    result.add_argument("--tensor_parallel_size", type=int, default=2)
    result.add_argument("--phase", choices=["formal", "diagnostic"], default="formal")
    return result


if __name__ == "__main__":
    calibrate(parser().parse_args())
