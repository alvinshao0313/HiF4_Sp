"""Stage entrypoints; CUDA visibility is chosen by the authorized caller."""
import argparse
from pathlib import Path

from .config import CANDIDATES, TASKS


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    for name in ("initialize", "prepare", "bind", "train", "export", "development", "select", "evaluate", "report"):
        stage = sub.add_parser(name)
        stage.add_argument("--run", type=Path, required=True)
        if name in ("train", "export", "development", "evaluate", "report"):
            stage.add_argument("--candidate", choices=tuple(CANDIDATES) + (("native_nvfp4",)
                               if name == "evaluate" else ()), required=True)
        if name in ("train", "export"):
            stage.add_argument("--qualification", action="store_true")
        if name in ("development", "evaluate"):
            stage.add_argument("--smoke", action="store_true")
            stage.add_argument("--model-dir", type=Path)
        if name == "evaluate":
            stage.add_argument("--task", choices=TASKS, required=True)
    args = parser.parse_args()
    if args.stage == "initialize":
        from .artifacts import initialize
        initialize(args.run)
    elif args.stage == "prepare":
        from .evaluate import prepare as evaluation_prepare
        from .prepare import prepare
        from .development import bind_inputs
        evaluation_prepare(args.run)
        prepare(args.run)
        bind_inputs(args.run)
    elif args.stage == "bind":
        from .artifacts import bind_sources
        bind_sources(args.run)
    elif args.stage == "train":
        from .training import train
        train(args.run, args.candidate, qualification=args.qualification)
    elif args.stage == "export":
        from .export import materialize
        materialize(args.run, args.candidate, qualification=args.qualification)
    elif args.stage == "development":
        from .development import evaluate
        evaluate(args.run, args.candidate, model_dir=args.model_dir, smoke=args.smoke)
    elif args.stage == "evaluate":
        from .evaluate import evaluate
        evaluate(args.run, args.candidate, args.task, model_dir=args.model_dir, smoke=args.smoke)
    elif args.stage == "select":
        from .development import select
        select(args.run)
    else:
        from .report import report
        report(args.run, args.candidate)


if __name__ == "__main__":
    main()
