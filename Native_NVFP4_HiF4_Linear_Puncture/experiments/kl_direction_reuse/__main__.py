"""Entry points for the agreed, independent single-layer comparison."""
import argparse
import json
from pathlib import Path
import signal

from .config import require_environment, recipes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list-recipes")
    for name in ("prepare", "baseline", "capture", "verify-prepare", "verify-finish", "train", "export", "report", "suite"):
        p = commands.add_parser(name)
        p.add_argument("--root", type=Path, required=True)
        if name == "capture":
            p.add_argument("--variant", choices=("native", "baseline", "candidate"), required=True)
            p.add_argument("--model-dir", type=Path)
            p.add_argument("--output", type=Path)
            p.add_argument("--split", choices=("all", "train", "val", "test"), default="all")
            p.add_argument("--sample-ids", help="comma separated exact sample IDs")
        elif name.startswith("verify-"):
            p.add_argument("--layer", type=int, choices=(8, 24, 40), required=True)
            if name == "verify-finish":
                p.add_argument("--capture-dir", type=Path, required=True)
        elif name == "train":
            p.add_argument("--recipe", choices=[r["name"] for r in recipes()], required=True)
            p.add_argument("--resume", action="store_true")
            p.add_argument("--budget-seconds", type=float,
                           help="override the fixed budget for an explicitly isolated smoke run")
            p.add_argument("--train-sample-ids",
                           help="comma-separated train IDs for an explicitly isolated smoke run")
        elif name == "export":
            p.add_argument("--checkpoint", type=Path, required=True)
            p.add_argument("--output", type=Path, required=True)
        elif name == "suite":
            p.add_argument("--stage", choices=("all", "prepare", "verify", "train", "evaluate"), required=True)
            p.add_argument("--train-gpu", required=True)
            p.add_argument("--eval-gpus", required=True)
    args = parser.parse_args()
    require_environment()
    if args.command == "list-recipes":
        print(json.dumps(recipes(), indent=2))
        return
    if args.command == "prepare":
        from .data import prepare
        prepare(args.root)
    elif args.command == "baseline":
        from .materialize import baseline
        baseline(args.root)
    elif args.command == "capture":
        from .capture import capture
        capture(args.root, args.variant, model_dir=args.model_dir, output=args.output, split=args.split,
                sample_ids=args.sample_ids.split(",") if args.sample_ids else None)
    elif args.command == "verify-prepare":
        from .verify import prepare
        prepare(args.root, args.layer)
    elif args.command == "verify-finish":
        from .verify import finish
        finish(args.root, args.layer, args.capture_dir)
    elif args.command == "train":
        from .train import train
        def terminate(signum, frame):
            raise KeyboardInterrupt("SIGTERM: commit the last complete update and account elapsed time")
        signal.signal(signal.SIGTERM, terminate)
        train(args.root, args.recipe, resume=args.resume,
              budget_seconds=args.budget_seconds,
              train_sample_ids=args.train_sample_ids.split(",") if args.train_sample_ids else None)
    elif args.command == "export":
        from .materialize import candidate
        candidate(args.root, args.checkpoint, args.output)
    elif args.command == "report":
        from .report import summarize
        summarize(args.root)
    elif args.command == "suite":
        from .runner import suite
        suite(args.root, args.stage, args.train_gpu, args.eval_gpus)


if __name__ == "__main__":
    main()
