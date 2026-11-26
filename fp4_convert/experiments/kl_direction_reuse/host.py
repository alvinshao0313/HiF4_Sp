"""Run the full agreed queue once, with durable status and no automatic retries."""
import argparse
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import signal
import sys
import traceback

from .artifacts import read_json, source_fingerprint, write_json
from .config import PROTOCOL, idle_devices, recipes, require_environment
from .runner import suite


def allocation(train_gpu, eval_gpus):
    selected = eval_gpus.split(",")
    if len(selected) != 2 or len(set(selected)) != 2 or train_gpu not in selected or not all(s.isdigit() for s in selected):
        raise ValueError("provide two distinct physical GPU indices and train on one of those GPUs")
    return idle_devices(eval_gpus)


def run(root, control, train_gpu, eval_gpus, *, after_baseline=False, after_captures=False):
    require_environment()
    root, control = Path(root).resolve(), Path(control).resolve()
    if control == root or root in control.parents:
        raise ValueError("control files must live outside the initially empty data root")
    control.mkdir(parents=True, exist_ok=True)
    # All continuation controls share the original run's lock.
    original_control = root.with_name(root.name + ".control")
    original_control.mkdir(parents=True, exist_ok=True)
    with (original_control / "host.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (control / "status.json").exists():
            raise FileExistsError("this queue has already been launched; inspect its failure before any manual continuation")
        status = dict(status="STARTING", pid=os.getpid(), root=str(root), control=str(control),
                      train_gpu=train_gpu, eval_gpus=eval_gpus, after_baseline=after_baseline, after_captures=after_captures,
                      started_at=datetime.now(timezone.utc).isoformat())
        write_json(status, control / "status.json")
        try:
            continuing = after_baseline or after_captures
            if not continuing and root.exists() and any(root.iterdir()):
                raise FileExistsError("full background queue requires a new empty experiment root")
            if continuing and (control == original_control or read_json(original_control / "status.json")["status"] not in {"FAILED", "INTERRUPTED"}):
                raise RuntimeError("continuation requires a stopped original queue and a new control directory")
            devices = allocation(train_gpu, eval_gpus)
            source = source_fingerprint()
            launch = dict(**status, devices=devices, python=sys.executable, source_sha256=source,
                          protocol=PROTOCOL.to_dict(), recipes=recipes(),
                          stages=["prepare", "verify", "train", "evaluate"], automatic_retry=False)
            write_json(launch, control / "launch.json")
            os.environ["KLD_CONTROL_DIR"] = str(control)
            os.environ["KLD_EXPECTED_SOURCE"] = source
            status["status"] = "RUNNING"
            write_json(status, control / "status.json")
            suite(root, "all", train_gpu, eval_gpus, after_baseline=after_baseline, after_captures=after_captures)
            report = read_json(root / "report.json")
            if len(report["candidates"]) != 15 or not report["cost"]["stage_ledger_complete"]:
                raise RuntimeError("queue returned without a complete 15-configuration report and stage ledger")
            status.update(status="COMPLETE", report=str(root / "report.json"))
        except BaseException as error:
            status.update(status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED",
                          error=repr(error), traceback=traceback.format_exc())
            raise
        finally:
            status["finished_at"] = datetime.now(timezone.utc).isoformat()
            write_json(status, control / "status.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--train-gpu", required=True)
    parser.add_argument("--eval-gpus", required=True)
    boundary = parser.add_mutually_exclusive_group()
    boundary.add_argument("--after-baseline", action="store_true",
                        help="explicitly reuse the completed data/baseline and execute every remaining stage")
    boundary.add_argument('--after-captures', action='store_true',
                          help='reuse certified complete native/baseline captures and start at verification')
    args = parser.parse_args()
    def terminate(signum, frame):
        raise KeyboardInterrupt("background host received SIGTERM")
    signal.signal(signal.SIGTERM, terminate)
    run(args.root, args.control, args.train_gpu, args.eval_gpus, after_baseline=args.after_baseline,
        after_captures=args.after_captures)


if __name__ == "__main__":
    main()
