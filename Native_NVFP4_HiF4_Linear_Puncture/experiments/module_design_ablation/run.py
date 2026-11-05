"""Finite authorized cohort: equal-budget LR screening, training, six-task evaluation."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from queue import Queue, Empty
import subprocess
import sys
from threading import Lock
import time

from ..non_equivalent_reconstruction.artifact import sha256, write_json
from ..downstream_accuracy_recovery.protocol import TASK_DIRS, protocol_for
from ..downstream_accuracy_recovery.reporting import validate_metrics
from .aime import aime_protocol, validate_result
from .config import NATIVE_SNAPSHOT, LR_MULTIPLIERS
from .variants import VARIANTS

REPO = Path(__file__).resolve().parents[3]
MODULE = "Native_NVFP4_HiF4_Linear_Puncture.experiments.module_design_ablation"
HISTORICAL = REPO / "Native_NVFP4_HiF4_Linear_Puncture/results/non_equivalent_reconstruction/full48_20260915T084300Z/group_top_mass/model"
GPUS = (0, 1, 6, 7)
TASKS = ("arc", "gsm8k", "lcb", "aime25", "mmlu_pro")


def now():
    return datetime.now(timezone.utc).isoformat()


def check_gpus_free(gpus):
    requested = {str(value) for value in str(gpus).split(",")}
    inventory = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True)
    selected = {uuid.strip() for line in inventory.splitlines()
                for index, uuid in [line.split(",", 1)] if index.strip() in requested}
    if len(selected) != len(requested):
        raise ValueError(f"GPU indices not found: {gpus}")
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True)
    occupied = [line for line in processes.splitlines() if line.split(",", 1)[0].strip() in selected]
    if occupied:
        raise RuntimeError(f"authorized GPUs have an existing compute process; no job launched: {occupied}")


def source_manifest():
    package = Path(__file__).resolve().parent
    paths = list(package.rglob("*.py"))
    for name in ("non_equivalent_reconstruction", "e2e_diag_reconstruction/core", "e2e_diag_reconstruction/data",
                 "e2e_diag_reconstruction/training", "downstream_accuracy_recovery"):
        paths.extend((package.parent / name).glob("*.py"))
    paths.extend((package.parents[1] / "src").glob("*.py"))
    paths.extend([REPO / name for name in (
        "main.py", "tasks/aime.py", "tasks/custom_tasks.py",
        "3rdparty/lighteval/src/lighteval/models/vllm/vllm_model.py",
        "3rdparty/vllm/vllm/model_executor/models/qwen3_moe.py",
        "3rdparty/vllm/vllm/model_executor/layers/linear.py",
        "3rdparty/vllm/vllm/model_executor/layers/quantization/hif4_runtime.py",
        "3rdparty/vllm/vllm/model_executor/layers/quantization/hif4_triton.py",
        "3rdparty/vllm/vllm/model_executor/layers/quantization/hif4_transform_triton.py",
        "3rdparty/vllm/vllm/model_executor/layers/fused_moe/experts/hif4_emulation_moe.py",
    )])
    return {str(path.relative_to(REPO)): sha256(path) for path in sorted(set(paths))}


def scores_for(root, name):
    scores = {}
    for task in TASKS:
        directory = root / "evaluation" / name / task
        if task == "aime25":
            metrics = json.loads((directory / "metrics.json").read_text())
            audit = json.loads((directory / "prompt_audit.json").read_text())
            checked = validate_result(metrics, audit=audit, model_kind=metrics["model_kind"], smoke=False)
            scores["AIME25 avg5"] = checked["score"]
        else:
            metrics = json.loads((directory / "eval" / TASK_DIRS[task] / "metrics.json").read_text())
            reference = None if task == "arc" else json.loads((directory / "prompt_audit.json").read_text())["models"]["native_nvfp4"]["tasks"][task]
            scores.update(validate_metrics(metrics, task, smoke=False, model_kind=metrics["model_kind"], prompt_reference=reference))
    return scores


def remove_export_weights(root, name):
    """Only this cohort's indexed BF16 shards, after all six scores validate."""
    scores_for(root, name)
    model = root / "exports" / name
    if model.is_symlink() or model.resolve().parent != (root / "exports").resolve():
        raise ValueError("cleanup must remain in this run's exports directory")
    marker = json.loads((model / "ablation_export.json").read_text())
    if not marker.get("complete") or marker.get("smoke_only") or marker["variant"] != name:
        raise ValueError("cleanup requires this variant's completed formal export")
    index = json.loads((model / "model.safetensors.index.json").read_text())
    files = []
    for filename in sorted(set(index["weight_map"].values())):
        path = model / filename
        if (Path(filename).name != filename or not filename.startswith("model-")
                or path.is_symlink() or path.resolve().parent != model.resolve()):
            raise ValueError(f"unexpected BF16 export asset: {filename}")
        files.append({"path": str(path), "bytes": path.stat().st_size,
                      "allocated_bytes": path.stat().st_blocks * 512, "sha256": sha256(path)})
    record = {"authorization": "user confirmed on 2026-10-08: remove this cohort's BF16 exports after six-task validation",
              "reason": "rebuildable BF16 shards; preserve learned parameters, rotation files, metadata and evaluation evidence",
              "files": files, "deleted": False}
    write_json(record, model / "weight_cleanup.json")
    for row in files:
        Path(row["path"]).unlink()
    if any(Path(row["path"]).exists() for row in files):
        raise RuntimeError("BF16 cleanup incomplete")
    record.update(deleted=True, completed_at=now(), allocated_bytes_removed=sum(row["allocated_bytes"] for row in files))
    write_json(record, model / "weight_cleanup.json")


def summarize(root):
    results = {name: scores_for(root, name) for name in VARIANTS}
    tasks = ("MMLU-Pro", "LCB", "GSM8K", "ARC-E", "ARC-C", "AIME25 avg5")
    if len(tasks) != 6 or any(set(row) != set(tasks) for row in results.values()):
        raise ValueError("summary requires exactly six validated scores for every one of the 15 variants")
    native = results["native_nvfp4"]
    differences = {name: {task: 100 * (value - native[task]) for task, value in row.items()}
                   for name, row in results.items()}
    pairs = (("norm", "fixed_r64"), ("diag", "fixed_r64"), ("rotation", "fixed_r64"),
             ("rotation_norm", "rotation"), ("matrix_norm", "matrix"), ("current", "matrix_norm"),
             ("norm_kl", "norm"), ("rotation_norm_kl", "rotation_norm"),
             ("matrix_norm_kl", "matrix_norm"), ("current_kl", "current"),
             ("matrix_norm", "rotation_norm"), ("current_kl", "existing_two_stage"))
    paired = {f"{a} - {b}": {task: 100 * (results[a][task] - results[b][task]) for task in tasks} for a, b in pairs}
    costs = {}
    for name, variant in VARIANTS.items():
        if not variant.needs_training:
            continue
        training = json.loads((root / "training" / name / "summary.json").read_text())
        selection = json.loads((root / "screen" / name / "selection.json").read_text())
        initial = json.loads((root / "training" / name / "layers/000/initial_metrics.json").read_text())
        screening_seconds = sum(json.loads((Path(trial["path"]) / "summary.json").read_text())["seconds"] for trial in selection["trials"])
        costs[name] = {"training_seconds": training["seconds"], "screening_seconds": screening_seconds,
                       "selected_multiplier": selection["selected"]["multiplier"],
                       "learning_rates": initial["learning_rates"], "trainable_parameters": initial["parameters"] * 48}
    nondominated = [name for name, row in results.items() if not any(
        all(other[task] >= row[task] for task in tasks) and any(other[task] > row[task] for task in tasks)
        for other_name, other in results.items() if other_name != name)]
    write_json({"complete": True, "scores": results, "delta_native_pp": differences, "paired_delta_pp": paired,
                "training_costs": costs, "not_dominated_on_observed_six_scores": nondominated}, root / "comparison.json")
    lines = ["# 模块消融全量结果", "", "所有分数为百分数；配对差值为百分点。AIME25 为 30 题×5 次生成的均值。", "",
             "| 模型 | " + " | ".join(tasks) + " |", "| --- | " + " | ".join("---:" for _ in tasks) + " |"]
    lines += ["| " + name + " | " + " | ".join(f"{100 * row[task]:.3f}" for task in tasks) + " |" for name, row in results.items()]
    lines += ["", "## 配对差值", "", "| 配对 | " + " | ".join(tasks) + " |", "| --- | " + " | ".join("---:" for _ in tasks) + " |"]
    lines += ["| " + name + " | " + " | ".join(f"{row[task]:+.3f}" for task in tasks) + " |" for name, row in paired.items()]
    lines += ["", "## 训练成本", "", "| 配置 | 正式训练小时 | LR 筛选小时 | 可训练参数 | 选中 multiplier |",
              "| --- | ---: | ---: | ---: | ---: |"]
    lines += [f"| {name} | {row['training_seconds'] / 3600:.2f} | {row['screening_seconds'] / 3600:.2f} | {row['trainable_parameters']:,} | {row['selected_multiplier']:g} |"
              for name, row in costs.items()]
    lines += ["", "六项观测分数上未被其他候选全面超过的配置：" + "、".join(nondominated) + "。此描述不等同于统计显著。",
              "", "HiF4 权重/激活采用 4-bit 数值量化；推理导出存储为 QDQ 后的 BF16 权重，不能当作实测 4-bit packed 存储或原生 4-bit 计算性能。"]
    lines += ["", "这些结果限定于本轮数据、学习率搜索和训练预算；不证明全局最优。已有两阶段参考包含额外 E4 训练，不能把其差值单独归因于初始化。", ""]
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def process_identity(pid):
    """Read the Linux process identity; zombies have already released GPU work."""
    directory = Path(f"/proc/{pid}")
    try:
        fields = (directory / "stat").read_text().rsplit(") ", 1)[1].split()
        command = [word.decode() for word in (directory / "cmdline").read_bytes().split(b"\0") if word]
    except FileNotFoundError:
        return None
    if fields[0] == "Z":
        return None
    return {"pid": pid, "start_ticks": int(fields[19]), "command": command}


def wait_existing_process(pid, command, *, interval=5.0):
    identity = process_identity(pid)
    if identity is None:
        return
    if identity["command"] != command:
        raise RuntimeError(f"PID {pid} no longer identifies the recorded job")
    while True:
        current = process_identity(pid)
        if current is None:
            return
        if current != identity:
            raise RuntimeError(f"PID {pid} identity changed while awaiting completion")
        time.sleep(interval)


def validate_training_artifacts(command):
    from .config import Config
    from .train import source_hashes

    def argument(name):
        return command[command.index(name) + 1]

    if command[1:3] != ["-m", MODULE + ".train"]:
        raise ValueError("only existing training jobs can be adopted")
    directory = Path(argument("--output_dir"))
    cfg = Config(output_dir=str(directory), variant=argument("--variant"),
                 phase=argument("--phase"), lr_multiplier=float(argument("--lr_multiplier")))
    cfg.validate()
    manifest = json.loads((directory / "manifest.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    if (manifest.get("complete") is not True or summary.get("complete") is not True
            or manifest["config"] != cfg.to_dict()
            or manifest["completed_layers"] != list(range(cfg.layers))
            or manifest["source_hashes"] != source_hashes()
            or summary["phase"] != cfg.phase or len(summary["layers"]) != cfg.layers):
        raise ValueError(f"existing training artifacts are incomplete or mismatched: {directory}")
    for layer, metrics in enumerate(summary["layers"]):
        if (not (directory / f"layers/{layer:03d}/selected.pt").is_file()
                or not 0 <= metrics["selected_epoch"] < cfg.epochs
                or not 0 <= metrics["objective"] < float("inf")):
            raise ValueError(f"invalid completed layer: {directory}, {layer}")
    return {"manifest_sha256": sha256(directory / "manifest.json"),
            "summary_sha256": sha256(directory / "summary.json")}


def run(output_dir, qualification, resume_driver=False):
    root, evidence = Path(output_dir).resolve(), Path(qualification).resolve()
    checked = json.loads((evidence / "qualification.json").read_text())
    sources = source_manifest()
    if not checked.get("passed") or checked["source_sha256"] != sources:
        raise ValueError("qualification must cover the current source files")
    # Require the completed prerequisite before allocating worker threads.
    # This avoids relying on Python builds exposing the optional pidfd API.
    native_check = Path(checked["evidence"]["aime"]["aime25_native"]["path"])
    native_metrics = json.loads(native_check.read_text())
    native_audit = json.loads((native_check.parent / "prompt_audit.json").read_text())
    validate_result(native_metrics, audit=native_audit, model_kind="native_nvfp4", smoke=True)
    active_variants = {}
    if resume_driver:
        state = json.loads((root / "status.json").read_text())
        manifest = json.loads((root / "manifest.json").read_text())
        if process_identity(state["pid"]) is not None:
            raise RuntimeError("the previous driver is still alive; do not run two drivers")
        if state["errors"] or any(not key.startswith(("screen/", "train/")) for key in state["jobs"]):
            raise ValueError("driver recovery requires a training-only run without failed jobs")
        changed = {key for key in manifest["source_sha256"].keys() | sources.keys()
                   if manifest["source_sha256"].get(key) != sources.get(key)}
        if changed - {str(Path(__file__).resolve().relative_to(REPO))}:
            raise ValueError("driver recovery cannot change training or evaluation sources")
        for key, row in state["jobs"].items():
            if row["status"] == "running":
                if not key.startswith("train/") or int(row["gpus"]) not in GPUS:
                    raise ValueError("only an existing formal single-GPU training job can be adopted")
                gpu, name = int(row["gpus"]), key.split("/", 1)[1]
                if gpu in active_variants:
                    raise ValueError("multiple recorded active jobs on the same GPU")
                identity = process_identity(row["pid"])
                if identity is not None and identity["command"] != row["command"]:
                    raise ValueError("existing training PID does not match its recorded command")
                if identity is None:
                    validate_training_artifacts(row["command"])
                active_variants[gpu] = name
            elif row["status"] != "complete":
                raise ValueError("failed jobs are not retried by driver recovery")
        revision = {"previous_driver_pid": state["pid"], "pid": os.getpid(), "started_at": now(),
                    "qualification": str(evidence), "source_sha256": sources}
        manifest.setdefault("driver_revisions", []).append(revision)
        write_json(manifest, root / "manifest.json")
        state.update(pid=os.getpid(), status="running", driver_restarted_at=now())
    else:
        root.mkdir(parents=True, exist_ok=False)
        (root / "logs").mkdir()
        write_json({"created_at": now(), "variants": list(VARIANTS), "gpus": GPUS,
                    "qualification": str(evidence), "source_sha256": sources,
                    "protocols": {**{task: protocol_for(task) for task in TASKS if task != "aime25"}, "aime25": aime_protocol()},
                    "lr_multipliers": LR_MULTIPLIERS, "screen_epochs": 3, "formal_epochs": 20,
                    "remove_bf16_exports_after_validated_six_tasks": True}, root / "manifest.json")
        state = {"status": "running", "pid": os.getpid(), "started_at": now(), "jobs": {}, "errors": []}
    lock = Lock()

    def save():
        write_json(state, root / "status.json")

    def job(key, gpus, module, arguments):
        if source_manifest() != sources:
            raise RuntimeError("execution sources changed after qualification")
        command = [sys.executable, "-m", MODULE + "." + module, *map(str, arguments)]
        previous = state["jobs"].get(key)
        if previous is not None:
            if not resume_driver or module != "train" or previous["command"] != command or previous["gpus"] != str(gpus):
                raise ValueError(f"existing job does not match the recovery command: {key}")
            if previous["status"] == "running":
                with lock:
                    previous["adopted_at"] = now()
                    save()
                wait_existing_process(previous["pid"], command)
                completion = validate_training_artifacts(command)
                with lock:
                    previous.update(status="complete", exit_code=None, ended_at=now(),
                                    exit_code_observed=False, completion_evidence=completion,
                                    completion_basis="process exited and all training artifacts validated")
                    save()
            elif previous["status"] == "complete" and previous.get("exit_code", 0) == 0:
                validate_training_artifacts(command)
            else:
                raise ValueError(f"failed jobs are not automatically retried: {key}")
            return
        check_gpus_free(gpus)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpus), "OMP_NUM_THREADS": "4"}
        log_path = root / "logs" / (key.replace("/", "__") + ".log")
        started = time.monotonic()
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            with lock:
                state["jobs"][key] = {"status": "running", "pid": process.pid, "gpus": str(gpus), "command": command,
                                      "log": str(log_path), "started_at": now()}
                save()
            rc = process.wait()
        with lock:
            state["jobs"][key].update(status="complete" if rc == 0 else "failed", exit_code=rc,
                                     seconds=time.monotonic() - started, ended_at=now())
            save()
        if rc:
            raise RuntimeError(f"{key} failed rc={rc}; {log_path}")

    def train_variant(name, gpu):
        trials = []
        for multiplier in LR_MULTIPLIERS:
            trial = root / "screen" / name / f"lr_{multiplier:g}"
            job(f"screen/{name}/{multiplier:g}", gpu, "train",
                ["--variant", name, "--phase", "screen", "--lr_multiplier", multiplier, "--output_dir", trial])
            summary = json.loads((trial / "summary.json").read_text())
            if not summary.get("complete") or summary["phase"] != "screen":
                raise ValueError("incomplete LR screening result")
            trials.append({"multiplier": multiplier, "objective": summary["layers"][0]["objective"], "path": str(trial)})
        winner = min(trials, key=lambda trial: trial["objective"])
        selection = {"trials": trials, "selected": winner, "formal_initialization": "fresh_native"}
        selection_path = root / "screen" / name / "selection.json"
        if selection_path.is_file():
            if json.loads(selection_path.read_text()) != selection:
                raise ValueError(f"existing LR selection changed: {name}")
        else:
            write_json(selection, selection_path)
        job(f"train/{name}", gpu, "train", ["--variant", name, "--phase", "formal", "--lr_multiplier", winner["multiplier"],
                                          "--output_dir", root / "training" / name])

    def evaluate_variant(name, gpus):
        definition = VARIANTS[name]
        kind = "native_nvfp4" if name == "native_nvfp4" else "hif4"
        if definition.is_reference:
            model = NATIVE_SNAPSHOT if name == "native_nvfp4" else HISTORICAL
        else:
            model = root / "exports" / name
            arguments = ["--variant", name, "--output_dir", model]
            if definition.needs_training:
                arguments.extend(["--run_dir", root / "training" / name])
            job(f"export/{name}", str(gpus).split(",")[0], "materialize", arguments)
        for task in TASKS:
            job(f"eval/{name}/{task}", gpus, "evaluate", ["--model_dir", model, "--model_kind", kind,
                "--task", task, "--output_dir", root / "evaluation" / name / task])
        scores_for(root, name)
        if not definition.is_reference:
            remove_export_weights(root, name)

    def process_queue(queue, resource, function):
        first = active_variants.get(resource) if function is train_variant else None
        while True:
            from_queue = first is None
            if first is not None:
                name, first = first, None
            else:
                try:
                    name = queue.get_nowait()
                except Empty:
                    return
            try:
                function(name, resource)
            except Exception as error:
                with lock:
                    state["errors"].append({"variant": name, "error": str(error), "time": now()})
                    state["status"] = "running_with_errors"
                    save()
            finally:
                if from_queue:
                    queue.task_done()

    def worker_finished(future):
        error = future.exception()
        if error is not None:
            with lock:
                state["errors"].append({"scope": "worker", "error": repr(error), "time": now()})
                state["status"] = "running_with_errors"
                save()

    save()
    train_queue = Queue()
    for name in ("rotation_norm_kl", "current_kl", "rotation_norm", "matrix_norm_kl", "rotation", "current", "matrix_norm", "matrix", "diag", "norm_kl", "norm"):
        if name not in active_variants.values():
            train_queue.put(name)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(process_queue, train_queue, gpu, train_variant) for gpu in GPUS]
        for future in futures:
            future.add_done_callback(worker_finished)
        for future in futures:
            future.result()
    if state["errors"]:
        state.update(status="failed", ended_at=now())
        save()
        raise RuntimeError("one or more training variants failed; see status.json")
    native_check = Path(checked["evidence"]["aime"]["aime25_native"]["path"])
    if not native_check.is_file():
        state.update(status="failed", ended_at=now())
        state["errors"].append({"error": "native AIME prerequisite did not produce validated metrics", "path": str(native_check)})
        save()
        raise RuntimeError("native AIME prerequisite failed; formal evaluation has not started")
    native_metrics = json.loads(native_check.read_text())
    native_audit = json.loads((native_check.parent / "prompt_audit.json").read_text())
    validate_result(native_metrics, audit=native_audit, model_kind="native_nvfp4", smoke=True)
    eval_queue = Queue()
    for name in VARIANTS:
        eval_queue.put(name)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(process_queue, eval_queue, pair, evaluate_variant) for pair in ("0,1", "6,7")]
        for future in futures:
            future.add_done_callback(worker_finished)
        for future in futures:
            future.result()
    if state["errors"]:
        state.update(status="failed", ended_at=now())
        save()
        raise RuntimeError("one or more evaluations failed; see status.json")
    summarize(root)
    state.update(status="complete", ended_at=now())
    save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--qualification", required=True)
    parser.add_argument("--resume_driver", action="store_true",
                        help="adopt an interrupted driver's running training jobs; never restart their training")
    run(**vars(parser.parse_args()))
