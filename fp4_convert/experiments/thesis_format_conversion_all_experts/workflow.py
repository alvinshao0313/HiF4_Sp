"""One finite qualified conversion job, with recorded command, logs and exit state."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from ..ban_trace_calibration.processes import require_free, join_session
from ..thesis_format_conversion_supplement.artifacts import process_record, read_json, run_root, write_json
from ..thesis_format_conversion_supplement.config import REPO
from .artifacts import MODULE, prepare, verify


def execute(root, action):
    root = run_root(root)
    verify(root)
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if gpu not in ("0", "1"):
        raise ValueError("this conversion uses one allocated GPU, 0 or 1")
    require_free(gpu)
    if action == "convert" and not read_json(root / "qualification/gptq_all_experts.json")["complete"]:
        raise ValueError("all-expert GPU qualification incomplete")
    directory = root / "jobs" / action
    directory.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-u", "-m", MODULE + ".workflow", action, "--run", str(root), "--worker"]
    log = directory / "worker.log"
    state = {**process_record(sys.argv), "status": "running", "command": command, "log": str(log)}
    write_json(state, directory / "status.json")
    started = time.monotonic()
    with log.open("w") as stream:
        child = subprocess.Popen(command, cwd=REPO, stdin=subprocess.DEVNULL, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        state.update(child_pid=child.pid, child_session_id=child.pid)
        write_json(state, directory / "status.json")
        code = child.wait()
    state.update(children=join_session(child.pid), exit_code=code, seconds=time.monotonic() - started,
                 status="failed" if code else "complete")
    write_json(state, directory / "status.json")
    if code:
        raise RuntimeError(f"{action} exit={code}; see {log}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("action", choices=("prepare", "qualify", "convert"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        if args.source is None:
            parser.error("prepare requires --source with an audited previous run")
        prepare(args.run, args.source)
    elif args.worker:
        from .convert import convert
        convert(args.run, qualification=args.action == "qualify")
    else:
        execute(args.run, args.action)
