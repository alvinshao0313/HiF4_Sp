"""One-shot, CPU-only preparation of the two independent checkpoints."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from .artifacts import PACKAGE, REPO, save
from .convert import SOURCE


def prepare(root):
    root = Path(root).absolute()
    status_path = root / "weights_status.json"
    if status_path.exists():
        raise FileExistsError(status_path)
    status = {"status": "running", "pid": os.getpid(), "stages": []}
    try:
        for variant in ("direct", "r64"):
            command = [sys.executable, "-u", "-m", PACKAGE + ".convert", "--source", str(SOURCE),
                       "--output", str(root / "checkpoints" / variant), "--variant", variant, "--threads", "8"]
            stage = {"variant": variant, "command": command, "status": "running",
                     "log": str(root / f"convert_{variant}.log")}
            status["stages"].append(stage)
            with Path(stage["log"]).open("x") as log:
                process = subprocess.Popen(command, cwd=REPO, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
                stage["pid"] = process.pid
                save(status_path, status)
                stage["returncode"] = process.wait()
            if stage["returncode"]:
                raise RuntimeError(f"Conversion failed: {variant}; see {stage['log']}")
            stage["status"] = "completed"
            save(status_path, status)
        status["status"] = "completed"
    except Exception as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save(status_path, status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    prepare(parser.parse_args().root)
