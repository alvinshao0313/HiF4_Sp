"""One finite GPU0/1 supplement; wait only when explicitly requested."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import time

from ..ban.evaluate import REPO, file_sha256
from .config import INFERENCE, model_view, now, read, save
from .processes import finish_stage, gpu_snapshot, require_free
from .remaining_data import prepare, verify
from .remaining_report import TASKS, summarize

MODULE = "Native_NVFP4_HiF4_Linear_Puncture.experiments.ban_trace_calibration.remaining"


def command(root, manifest, arm, name, *, short=False):
    target = root / ("qualification" if short else "formal") / arm / name
    view = model_view(manifest["model"], target / "model")
    result = [sys.executable, "-u", "-m", "Native_NVFP4_HiF4_Linear_Puncture.experiments.ban.additional_main",
              "--dataset_manifest", str(root / "datasets.json"), "--model_path", str(view),
              "--hif4_runtime_spec", manifest["spec"], "--output_dir", str(target / "evaluation"),
              "--datasets", TASKS[name]["task"], "--max_new_tokens", str(64 if short else TASKS[name]["max_new_tokens"])]
    for key, value in INFERENCE.items():
        result.append("--" + key)
        if not isinstance(value, bool):
            result.append(str(value))
    if short:
        result += ["--max_samples", "1"]
    if arm == "top8":
        result += ["--expert_pruning_method", "none"]
    else:
        settings = manifest["arms"][arm]
        result += ["--expert_pruning_method", "Ban", "--ban_artifact_path", settings["artifact"],
                   "--ban_lambda", str(settings["lambda"]), "--ban_k_min", "5", "--ban_execution", "compact",
                   "--ban_stats_dir", str(target / "routing")]
    return result, target


def wait_for_pair(root, manifest, state):
    deadline = time.monotonic() + manifest["wait_timeout_hours"] * 3600
    state.update(status="waiting_for_gpu", wait_started_at=now(),
                 wait_timeout_hours=manifest["wait_timeout_hours"], observed_gpu_state=gpu_snapshot("0,1"))
    save(root / "status.json", state)
    print("WAITING_FOR_GPU_0_1", flush=True)
    while True:
        snapshot = gpu_snapshot("0,1")
        if len(snapshot) == 2 and all(not d["pids"] and d["memory_mib"] <= 1024 for d in snapshot.values()):
            verify(root)
            state.update(status="running", gpu_available_at=now(), observed_gpu_state=snapshot)
            save(root / "status.json", state)
            print("GPU_0_1_AVAILABLE", flush=True)
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("GPU0/1 did not become free within the authorized finite 24-hour wait; no evaluation started")
        # This is a single user's finite experiment queue, never a GPU holder,
        # process terminator or retry daemon. Both cards must be empty together.
        time.sleep(30)


def run(root, wait):
    root = Path(root).resolve()
    manifest, state = verify(root), read(root / "status.json")
    if state["status"] != "prepared":
        raise ValueError("only a newly prepared supplement may start; no automatic retries")
    state.update(controller_pid=os.getpid(), started_at=now(), controller_command=sys.argv)
    save(root / "status.json", state)

    def phase(arm, name, short):
        verify(root)
        require_free("0,1")
        label = f"{'qualification' if short else 'formal'}/{arm}/{name}"
        argv, target = command(root, manifest, arm, name, short=short)
        logfile = root / "logs" / (label.replace("/", "_") + ".log")
        record = {"status": "running", "command": argv, "gpu_pair": "0,1", "started_at": now(), "log": str(logfile)}
        state["phases"][label] = record
        save(root / "status.json", state)
        try:
            with logfile.open("x") as log:
                child = subprocess.Popen(argv, cwd=REPO, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0,1",
                    "OMP_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"},
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                record["pid"] = child.pid
                save(root / "status.json", state)
                record["exit_code"] = child.wait()
            record["cleanup"] = finish_stage(child.pid, "0,1")
            if record["exit_code"]:
                raise RuntimeError(f"{label} failed: {logfile}")
            result = summarize(target, name, read(root / f"{name}_input_audit.json"), arm=arm, short=short)
            record.update(status="complete", score_pct=result["score_pct"], scoring_status=result["status"],
                          report=str(target / "report.json"))
        except Exception as error:
            record.update(status="failed", error=str(error))
            raise
        finally:
            record["ended_at"] = now()
            save(root / "status.json", state)

    try:
        if wait:
            wait_for_pair(root, manifest, state)
        else:
            require_free("0,1")
            state["status"] = "running"
        # New task/cache/report path: formal engine geometry and five actual
        # answers, only decode steps shortened. Old kernel evidence is reused.
        for arm in ("top8", "c4", "trace"):
            phase(arm, "aime24", True)
        for arm in ("c4", "trace"):
            for name in TASKS:
                phase(arm, name, False)
        phase("top8", "aime24", False)
        report = build_report(root)
        state["status"] = report["status"]
    except Exception as error:
        state.update(status="failed", errors=[*state["errors"], str(error)])
        raise
    finally:
        state["ended_at"] = now()
        save(root / "status.json", state)


def build_report(root):
    root, tasks, pending = Path(root), {}, []
    for name in TASKS:
        groups = {}
        for arm in ("top8", "c4", "trace"):
            target = root / ("top8" if arm == "top8" and name != "aime24" else f"formal/{arm}") / name
            result = read(target / "report.json")
            groups[arm] = result
            if result["status"] != "complete":
                pending.append(f"{arm}/{name}")
        if all(g["status"] == "complete" for g in groups.values()):
            baseline = groups["top8"]
            for arm in ("c4", "trace"):
                group = groups[arm]
                assert group["protocol"] == baseline["protocol"]
                assert group["actual_input_tokens_hash"] == baseline["actual_input_tokens_hash"]
                group["retention_pct"] = 100 * group["score_pct"] / baseline["score_pct"]
        tasks[name] = groups
    report = {"status": "awaiting_aime_review" if pending else "complete", "tasks": tasks,
              "pending_aime_reviews": pending, "created_at": now(),
              "scope": "Remaining tasks only; fixed 6.4-budget lambdas, no test-set tuning or speed-conversion experiment."}
    save(root / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--source", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--data_source", required=True)
    start = sub.add_parser("run")
    start.add_argument("--run", required=True)
    start.add_argument("--wait_for_gpu", action="store_true")
    summarize = sub.add_parser("report")
    summarize.add_argument("--run", required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.source, args.output, args.data_source)
    elif args.action == "run":
        run(args.run, args.wait_for_gpu)
    else:
        build_report(args.run)


if __name__ == "__main__":
    main()
