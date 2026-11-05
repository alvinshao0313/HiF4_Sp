"""Forward the existing main.py CLI, with an explicit none/Ban switch."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(add_help=False, description=__doc__)
    parser.add_argument("--expert_pruning_method", choices=["none", "Ban"], default="none")
    parser.add_argument("--ban_artifact_path")
    parser.add_argument("--ban_lambda", type=float, default=0.7)
    parser.add_argument("--ban_k_min", type=int, default=3)
    parser.add_argument("--ban_execution", choices=["masked", "compact"], default="compact")
    parser.add_argument("--ban_stats_dir")
    args, remaining = parser.parse_known_args()
    if "--help" in remaining or "-h" in remaining:
        parser.print_help()
    elif args.expert_pruning_method == "Ban":
        if not args.ban_artifact_path:
            parser.error("--expert_pruning_method Ban requires --ban_artifact_path")
        forwarded = argparse.ArgumentParser(add_help=False)
        forwarded.add_argument("--model_path")
        forwarded.add_argument("--hif4_runtime_spec")
        forwarded.add_argument("--data_parallel_size", type=int)
        base_args, base_remaining = forwarded.parse_known_args(remaining)
        if base_args.model_path is None or base_args.hif4_runtime_spec is None:
            parser.error("Ban requires --model_path and --hif4_runtime_spec")
        if base_args.data_parallel_size not in (None, 1):
            parser.error("Ban currently supports data_parallel_size=1; Ray workers do not inherit this extension")
        from .engine import ban_config, install_engine_hook
        settings = dict(artifact_path=args.ban_artifact_path, lam=args.ban_lambda,
                        k_min=args.ban_k_min, execution=args.ban_execution)
        # Validate before creating files or starting an engine.
        ban_config(base_args.model_path, base_args.hif4_runtime_spec, **settings)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        root = (Path(args.ban_stats_dir).resolve() if args.ban_stats_dir else
                Path(args.ban_artifact_path).resolve().parent / f"inference_{stamp}")
        root.mkdir(parents=True, exist_ok=False)
        model = Path(base_args.model_path).resolve()
        view = root / "model_view"
        view.mkdir()
        # Current lighteval caches predictions below absolute model_name. A new
        # file-only view isolates lambda/k_min without copying model weights.
        for path in model.iterdir():
            if path.is_file():
                (view / path.name).symlink_to(path.resolve())
        remaining = [*base_remaining, "--model_path", str(view), "--hif4_runtime_spec", base_args.hif4_runtime_spec]
        if base_args.data_parallel_size is not None:
            remaining += ["--data_parallel_size", str(base_args.data_parallel_size)]
        install_engine_hook(settings, root)
    # With none (the default), do not import vLLM, install a worker, load an
    # artifact or create a view. The original main.py owns inference unchanged.
    sys.argv = ["main.py", *remaining]
    runpy.run_path(str(Path(__file__).resolve().parents[3] / "main.py"), run_name="__main__")


if __name__ == "__main__":
    main()
