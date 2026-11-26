"""Two finite TP4 lanes; qualify all eight, then rerun GPQA without retries."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import threading

from .common import LANES, OLD_RUNS, PACKAGE, REPO, check_hashes, now, read, save


def stage(root, variant, phase):
    import torch
    from ..nvfp4_gpqa.run import run as run_evaluation
    from ..qwen3_mxfp4.evidence import summarize_phase
    from ..qwen3_mxfp4.workflow import verify_worker_release
    from .report import final_submission_summary, validate_runtime
    torch.set_num_threads(8)
    root = Path(root).resolve()
    check_hashes(read(root / "workflow_manifest.json")["source_sha256"])
    os.environ["QWEN3_MXFP4_AUDIT_DIR"] = str(root / variant / phase / "audit")
    run_evaluation(root / variant, phase)
    summarize_phase(root / variant, phase, root / "protocol_inputs.json", root / variant / phase / "summary.json")
    validate_runtime(root, variant, phase)
    if variant != "native":
        verify_worker_release(root / variant / phase / "audit")
    if phase == "formal":
        final_submission_summary(root, variant)


def run(root, phase):
    from ..qwen3_mxfp4_rotation_eval.workflow import available
    root = Path(root).resolve()
    target = root / f"{phase}_workflow_status.json"
    if target.exists():
        raise FileExistsError("This one-shot phase already has a status; no automatic retry")
    manifest = read(root / "workflow_manifest.json")
    check_hashes(manifest["source_sha256"])
    if phase == "formal":
        previous = read(root / "probe_workflow_status.json")
        if previous["status"] != "completed" or len(previous["stages"]) != 8:
            raise ValueError("All eight prompt-only TP4 probes must pass first")
    equivalence = read(root / "direct_r64_preservation.json")
    if equivalence["status"] != "passed":
        raise ValueError("Direct/R64 kernel preservation verification is missing")
    state = {"status": "running", "pid": os.getpid(), "started_at": now(), "phase": phase, "stages": {}}
    lock, failed = threading.Lock(), threading.Event()
    save(target, state)

    def lane(item):
        gpus, variants = item
        for variant in variants:
            if failed.is_set():
                return
            record = {"status": "starting", "devices": gpus, "started_at": now(),
                      "command": [sys.executable, "-u", "-m", PACKAGE + ".workflow", "stage",
                                  "--root", str(root), "--variant", variant, "--phase", phase],
                      "log": str(root / variant / phase / "controller.log")}
            try:
                available(gpus)
                with lock:
                    state["stages"][variant] = record
                    save(target, state)
                with Path(record["log"]).open("x") as stream:
                    proc = subprocess.Popen(record["command"], cwd=REPO, stdin=subprocess.DEVNULL,
                                            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                                            env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpus))
                    with lock:
                        record.update(pid=proc.pid, status="running")
                        save(target, state)
                    code = proc.wait()
                if code:
                    raise RuntimeError(f"{variant}/{phase} exited {code}; see {record['log']}")
                with lock:
                    record.update(status="completed", exit_code=code, finished_at=now())
                    save(target, state)
                print(f"Completed {variant}/{phase}", flush=True)
            except Exception as error:
                failed.set()
                with lock:
                    record.update(status="failed", error=str(error), finished_at=now())
                    state["stages"][variant] = record
                    save(target, state)
                raise

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lane, LANES.items()))
        if failed.is_set() or len(state["stages"]) != len(OLD_RUNS):
            raise RuntimeError("Incomplete workflow")
        state["status"] = "completed"
        if phase == "formal":
            summaries = {v: read(root / v / "formal/final_submission_summary.json") for v in OLD_RUNS}
            save(root / "comparison.json", {"status": "awaiting_blind_review" if any(
                x["pending_review_count"] for x in summaries.values()) else "completed", "results": summaries,
                "protocol": "AISBench zero-shot CoT prompt; original GPQA parameters, BF16 KV and scoring rules"})
    except Exception as error:
        state.update(status="failed", error=str(error))
        raise
    finally:
        state["finished_at"] = now()
        save(target, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("probe", "formal", "stage"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--variant", choices=tuple(OLD_RUNS))
    parser.add_argument("--phase", choices=("probe", "formal"))
    args = parser.parse_args()
    if args.action == "stage":
        if not args.variant or not args.phase:
            parser.error("stage requires --variant and --phase")
        stage(args.root, args.variant, args.phase)
    else:
        run(args.root, args.action)
