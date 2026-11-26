"""Fixed ten-task TP2 continuation, preserving the running CE development process."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys

import psutil

from ..reasoning_trace_full_kl_ce.artifacts import now, read_json, sha256, verify, write_json
from ..reasoning_trace_full_kl_ce.config import CANDIDATES, MODULE, REPO

PARALLEL_MODULE = __package__
PAIRS = ("0,1", "4,5", "6,7")
TASKS = ("mmlu_pro", "lcb", "gsm8k", "aime25", "arc")


def identity(pid):
    path = Path(f"/proc/{pid}")
    try:
        stat = (path / "stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": pid, "start_ticks": stat[19],
                "command": (path / "cmdline").read_bytes().rstrip(b"\0").decode().split("\0")}
    except FileNotFoundError:
        return None


def sources():
    return {str(p.relative_to(REPO)): sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def locked(root, objective):
    selected = read_json(root / "development" / objective / "selection.json")
    export = root / "models" / CANDIDATES[objective] / "export.json"
    if (not selected["complete"] or selected["formal_tests_used_for_selection"]
            or selected["export_sha256"] != sha256(export)):
        raise ValueError(f"{objective} requires independently locked checkpoint/export")
    return sha256(root / "development" / objective / "selection.json")


def prepare(root, output, controller, development):
    root, original = verify(root)
    parent, child = identity(controller), identity(development)
    if not parent or not child or str(root) not in parent["command"] or str(root) not in child["command"]:
        raise ValueError("expected live original controller and development command")
    if (parent["command"][3] != MODULE + ".run" or child["command"][3] != MODULE + ".development"
            or child["command"][-2:] != ["--objective", "hard_ce"]):
        raise ValueError("legacy PID command identity mismatch")
    locked(root, "full_kl")
    if (root / "evaluation").exists():
        raise ValueError("formal outputs already exist; refusing a competing evaluation schedule")
    output.mkdir(parents=True, exist_ok=False)
    write_json(original, output / "original_manifest.json")
    write_json({"created_at": now(), "run": str(root), "status": "prepared", "pairs": PAIRS,
                "legacy_controller": parent, "adopted_development": child,
                "tasks": [[o, t] for o in CANDIDATES for t in TASKS],
                "source_hashes": sources(), "original_source_hashes_sha256": sha256(root / "source_hashes.json"),
                "KL_selection_sha256": locked(root, "full_kl"),
                "selection_isolation": "each checkpoint is locked before its tests; existing CE selector reads development data only"},
               output / "manifest.json")
    for path in sources():
        target = output / "source" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPO / path).read_bytes())


def validate(root, output):
    root, _ = verify(root)
    m = read_json(output / "manifest.json")
    if (m["run"] != str(root) or m["source_hashes"] != sources()
            or m["original_source_hashes_sha256"] != sha256(root / "source_hashes.json")
            or m["KL_selection_sha256"] != locked(root, "full_kl")):
        raise ValueError("fixed continuation bindings changed")
    return root, m


def resources(pair):
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.total,memory.free", "--format=csv,noheader,nounits"], text=True)
    observed = {int(a): (int(b), int(c)) for a, b, c in (line.split(",") for line in raw.splitlines())}
    for gpu in map(int, pair.split(",")):
        total, free = observed[gpu]
        if free < total * .9 + 1024:
            raise RuntimeError(f"GPU {gpu}: {free} MiB free; existing work remains untouched")
    return {str(g): observed[g] for g in map(int, pair.split(","))}


def record(output, item):
    with (output / "commands.jsonl").open("a") as stream:
        stream.write(json.dumps({"time": now(), **item}) + "\n")


async def command(root, output, module, name, pair, *arguments):
    observed = resources(pair) if pair else {}
    argv = [sys.executable, "-u", "-m", module, "--run", str(root), *arguments]
    logfile = output / "logs" / (name + ".log")
    logfile.parent.mkdir(exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": pair, "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "TOKENIZERS_PARALLELISM": "false"}
    with logfile.open("x") as log:
        child = await asyncio.create_subprocess_exec(*argv, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                                     stdout=log, stderr=subprocess.STDOUT)
    item = {"name": name, "pid": child.pid, "command": argv, "gpus": pair, "log": str(logfile), "resources": observed}
    record(output, {**item, "status": "started"})
    print(json.dumps({**item, "status": "started"}), flush=True)
    code = await child.wait()
    record(output, {**item, "status": "exited", "exit_code": code})
    return code


async def qualify(root, output):
    root, m = validate(root, output)
    # Exercise waiting on an actual grandchild, as for the adopted CE process.
    program = "import subprocess,sys; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(1)']); print(p.pid,flush=True); sys.exit(p.wait())"
    helper = await asyncio.create_subprocess_exec(sys.executable, "-c", program,
                                                 stdin=subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE)
    expected = identity(int(await helper.stdout.readline()))
    if expected is None:
        raise RuntimeError("real foreign-process qualification child was not observed")
    await wait_pid(expected)
    if await helper.wait() != 0 or identity(expected["pid"]) is not None:
        raise RuntimeError("real foreign-process wait failed")
    codes = await asyncio.gather(*(command(root, output, PARALLEL_MODULE + ".qualify", "qualify_" + pair.replace(",", "_"), pair,
                                           "--output", str(output / ("qualification_" + pair.replace(",", "_")))) for pair in PAIRS[:2]))
    if codes != [0, 0]:
        raise RuntimeError(f"concurrent real TP2 qualification failed: {codes}")
    write_json({"passed": True, "source_hashes": sources(), "pairs": PAIRS[:2],
                "real_foreign_process_wait_passed": True, "time": now()}, output / "qualification_complete.json")


async def wait_pid(expected):
    observed = identity(expected["pid"])
    if observed is not None:
        if observed != expected:
            raise ValueError("adopted development PID was reused or its command changed")
        try:
            process = psutil.Process(expected["pid"])
        except psutil.NoSuchProcess:
            return
        await asyncio.to_thread(process.wait)


async def adopted_development(root, output, expected):
    await wait_pid(expected)
    selection = locked(root, "hard_ce")
    record(output, {"name": "adopted_hard_ce_development", "pid": expected["pid"],
                    "status": "finished_artifacts_verified", "exit_code": None,
                    "exit_code_note": "adopted process is not this controller's child; no exit code is inferred",
                    "selection_sha256": selection})
    return selection


async def execute(root, output):
    root, m = validate(root, output)
    if identity(m["legacy_controller"]["pid"]) is not None:
        raise RuntimeError("original controller must be explicitly stopped before takeover; this entry never stops processes")
    q = read_json(output / "qualification_complete.json")
    if not q["passed"] or q["source_hashes"] != sources():
        raise ValueError("applicable concurrent real TP2 qualification required")
    if (root / "evaluation").exists():
        raise ValueError("formal evaluation has already started; refusing duplicate task ownership")
    for pair in PAIRS[:2]:
        resources(pair)
    original = read_json(root / "manifest.json")
    original.update(status="running", stage="parallel_evaluation", pid=os.getpid(), parallel_continuation=str(output), inference_gpu_pairs=PAIRS)
    write_json(original, root / "manifest.json")
    m.update(status="running", pid=os.getpid(), started_at=now())
    write_json(m, output / "manifest.json")
    pending = [("full_kl", task) for task in TASKS]
    idle, active, errors, review = list(PAIRS[:2]), {}, [], []
    adopt = asyncio.create_task(adopted_development(root, output, m["adopted_development"]))
    ce_waiting = True
    while pending or active or ce_waiting:
        pending.sort(key=lambda row: (TASKS.index(row[1]), list(CANDIDATES).index(row[0])))
        while pending and idle:
            objective, task = pending.pop(0)
            locked(root, objective)
            pair = idle.pop(0)
            job = asyncio.create_task(command(root, output, MODULE + ".evaluation", objective + "_" + task, pair,
                                              "--objective", objective, "--task", task))
            active[job] = (pair, objective, task)
        waiting = set(active) | ({adopt} if ce_waiting else set())
        done, _ = await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
        for job in done:
            if job is adopt:
                ce_waiting = False
                try:
                    m["CE_selection_sha256"] = job.result()
                    write_json(m, output / "manifest.json")
                    qualification = asyncio.create_task(command(root, output, MODULE + ".qualify", "hard_ce_qualify", "6,7", "--objective", "hard_ce"))
                    active[qualification] = ("6,7", "hard_ce", "qualify")
                except Exception as error:
                    errors.append(f"CE development: {error}")
                continue
            pair, objective, task = active.pop(job)
            try:
                code = job.result()
            except Exception as error:
                errors.append(f"{objective}/{task}: {error}")
                continue
            if code == 0 or (task == "aime25" and code == 21):
                if task == "qualify":
                    pending.extend(("hard_ce", name) for name in TASKS)
                elif code == 21:
                    review.append(str(root / "evaluation" / CANDIDATES[objective] / task / "corrected_responses.json"))
                idle.append(pair)
            else:
                errors.append(f"{objective}/{task}: exit {code}")
                if task != "qualify":
                    idle.append(pair)
        if pending and not idle and not active and not ce_waiting:
            errors.append("no available pair remains for pending jobs")
            break
    if errors:
        m.update(status="stopped", errors=errors, finished_at=now())
        original.update(status="stopped", reason=errors, stopped_at=now())
    elif review:
        m.update(status="awaiting_aime_review", pending_reviews=review, finished_at=now())
        original.update(status="awaiting_aime_review", pending_reviews=review, finished_at=now())
    else:
        code = await command(root, output, MODULE + ".report", "report", "")
        m.update(status="complete" if code == 0 else "stopped", report_exit_code=code, finished_at=now())
        original.update(status=m["status"], finished_at=now())
    write_json(m, output / "manifest.json")
    write_json(original, root / "manifest.json")
    if errors or m["status"] == "stopped":
        raise RuntimeError(f"parallel continuation incomplete: {m}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("prepare", "qualify", "execute"), required=True)
    parser.add_argument("--legacy-controller", type=int)
    parser.add_argument("--development-pid", type=int)
    args = parser.parse_args()
    root, output = args.run.resolve(), args.output.resolve()
    if not output.is_relative_to(root):
        parser.error("continuation output must be inside the original RUN")
    if args.mode == "prepare":
        if not args.legacy_controller or not args.development_pid:
            parser.error("prepare requires both exact legacy PID identities")
        prepare(root, output, args.legacy_controller, args.development_pid)
    else:
        asyncio.run(qualify(root, output) if args.mode == "qualify" else execute(root, output))
