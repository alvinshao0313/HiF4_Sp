"""Run the finite approved experiment after a real TP2 preflight."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import threading

from ..ban.evaluate import REPO, file_sha256, gpu_pair
from .config import ARMS, INFERENCE, MODULE, TASKS, model_view, now, read, save, verify
from .report import build_report, summarize
from .processes import finish_stage, require_free


def check_pairs(pairs):
    if not 1 <= len(pairs) <= 3:
        raise ValueError("provide one to three TP2 pairs")
    devices = [item for pair in pairs for item in pair.split(",")]
    if len(set(devices)) != len(devices):
        raise ValueError("TP2 pairs must be disjoint")
    for pair in pairs:
        gpu_pair(pair)


def check_free(pair):
    require_free(pair)


def evaluation_command(root, manifest, arm, name, target, artifact=None, lam=None, *, short=False):
    view = model_view(manifest["model"], target / "model")
    command = [sys.executable, "-u", "-m", "Native_NVFP4_HiF4_Linear_Puncture.experiments.ban.additional_main",
               "--dataset_manifest", str(root / "datasets.json"), "--model_path", str(view),
               "--hif4_runtime_spec", manifest["spec"], "--output_dir", str(target / "evaluation"),
               "--datasets", TASKS[name]["task"], "--max_new_tokens", str(TASKS[name]["max_new_tokens"])]
    for key, value in INFERENCE.items():
        command.append("--" + key)
        if not isinstance(value, bool):
            command.append(str(value))
    if short:
        command += ["--max_samples", "2"]
    if arm == "top8":
        command += ["--expert_pruning_method", "none"]
    else:
        if artifact is None or lam is None:
            raise ValueError("Ban evaluation requires a matched artifact and lambda")
        command += ["--expert_pruning_method", "Ban", "--ban_artifact_path", str(artifact),
                    "--ban_lambda", str(lam), "--ban_k_min", "5", "--ban_execution", "compact",
                    "--ban_stats_dir", str(target / "routing")]
    return command


class Workflow:
    def __init__(self, args):
        self.args, self.root = args, Path(args.run).resolve()
        check_pairs(args.gpu_pairs)
        self.manifest = verify(self.root)
        self.state = read(self.root / "status.json")
        expected = "prepared" if args.preflight else "preflight_complete"
        if self.state["status"] != expected:
            raise ValueError(f"expected {expected}, found {self.state['status']}; no automatic restart")
        for pair in args.gpu_pairs:
            check_free(pair)
        if not args.preflight:
            proof = read(self.root / "preflight.json")
            if not proof["passed"]:
                raise ValueError("preflight did not pass")
            for path, digest in proof["artifact_sha256"].items():
                if file_sha256(path) != digest:
                    raise ValueError("preflight artifact changed")
            if self.manifest.get("continuation_source") and not args.verify_restart:
                proof = read(self.root / "restart_verification.json")
                if not proof["passed"]:
                    raise ValueError("restart process-handoff verification did not pass")
                for path, digest in proof["artifact_sha256"].items():
                    if file_sha256(path) != digest:
                        raise ValueError("restart verification artifact changed")
        if args.verify_restart and not self.manifest.get("continuation_source"):
            raise ValueError("--verify_restart is only for an explicitly prepared continuation")
        self.lock = threading.Lock()
        mode = "preflight_running" if args.preflight else "restart_verification_running" if args.verify_restart else "running"
        self.state.update(status=mode, controller_pid=os.getpid(),
                          started_at=now(), gpu_pairs=args.gpu_pairs, controller_command=sys.argv)
        save(self.root / "status.json", self.state)

    def phase(self, label, command, pair, validate=None):
        path = self.root / "logs" / (label.replace("/", "_") + ".log")
        record = {"status": "starting", "command": command, "gpu_pair": pair,
                  "log": str(path), "started_at": now()}
        with self.lock:
            self.state["phases"][label] = record
            save(self.root / "status.json", self.state)
        try:
            verify(self.root)
            check_free(pair)
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=pair, OMP_NUM_THREADS="4",
                               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
            with path.open("w", encoding="utf-8") as log:
                child = subprocess.Popen(command, cwd=REPO, env=environment, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                with self.lock:
                    record.update(status="running", pid=child.pid)
                    save(self.root / "status.json", self.state)
                code = child.wait()
            record["exit_code"] = code
            record["cleanup"] = finish_stage(child.pid, pair)
            if code:
                raise RuntimeError(f"{label} exited {code}; see {path}")
            if validate is not None:
                record.update(validate())
            record["status"] = "complete"
        except Exception as error:
            record.update(status="failed", error=str(error))
            with self.lock:
                self.state["errors"].append({"phase": label, "error": str(error)})
            raise
        finally:
            with self.lock:
                record["ended_at"] = now()
                save(self.root / "status.json", self.state)

    def stage(self, name, pair, arm="trace"):
        label = name if name in ("diagnostic", "preflight_budget") else f"{arm}/{name}"
        if label in self.manifest.get("reused_phases", {}):
            return
        self.phase(label, [sys.executable, "-u", "-m", MODULE + ".stages", "--run", str(self.root),
                           "--stage", name, "--arm", arm], pair)

    def evaluate(self, arm, name, index, pair, artifact, lam, *, short=False):
        label = f"qualification/{arm}/{name}" if short else f"{arm}/{name}/round_{index:02d}"
        if label in self.manifest.get("reused_phases", {}):
            return
        target = self.root / label
        command = evaluation_command(self.root, self.manifest, arm, name, target, artifact, lam, short=short)
        audit = read(self.root / f"{name}_input_audit.json")

        def validate():
            result = summarize(target, name, arm, audit, short=short,
                               target_experts=self.manifest["budget"]["target"],
                               budget_tolerance=self.manifest["budget"]["tolerance"])
            save(target / "report.json", result)
            return {"report": str(target / "report.json"), "score_pct": result["score_pct"]}

        self.phase(label, command, pair, validate)

    def arm(self, arm, pair):
        try:
            artifact, lam = None, None
            if arm != "top8":
                if arm == "trace":
                    self.stage("calibrate", pair)
                self.stage("budget", pair, arm)
                budget = read(self.root / f"budget/{arm}/budget.json")
                if budget["status"] != "complete" or not budget["budget_matched"]:
                    raise RuntimeError("budget did not match; no formal evaluation allowed")
                artifact, lam = budget["artifact"], budget["chosen_lambda"]
            for name in TASKS:
                self.evaluate(arm, name, 0, pair, artifact, lam, short=True)
            for name, definition in TASKS.items():
                for index in range(1, definition["rounds"] + 1):
                    self.evaluate(arm, name, index, pair, artifact, lam)
        except Exception as error:
            with self.lock:
                if not any(row.get("error") == str(error) for row in self.state["errors"]):
                    self.state["errors"].append({"arm": arm, "error": str(error)})
                save(self.root / "status.json", self.state)

    def lane(self, index):
        # With fewer pairs, finish each arm before loading the next; no GPU holder.
        order = ("trace", "c4", "top8") if len(self.args.gpu_pairs) == 1 else ARMS
        for arm in order[index::len(self.args.gpu_pairs)]:
            self.arm(arm, self.args.gpu_pairs[index])

    def run(self):
        if self.args.verify_restart:
            try:
                files, labels = [], []
                for index in (1, 2):
                    label = f"restart_verification/round_{index:02d}"
                    path = self.root / (label + ".json")
                    self.phase(label, [sys.executable, "-u", "-m", MODULE + ".release_probe",
                               "--run", str(self.root), "--output", str(path)], self.args.gpu_pairs[0])
                    if read(path)["status"] != "complete":
                        raise RuntimeError("real restart verification did not complete")
                    files.append(path)
                    labels.append(label)
                save(self.root / "restart_verification.json", {"passed": True, "created_at": now(),
                     "artifact_sha256": {str(path): file_sha256(path) for path in files},
                     "cleanup": {label: self.state["phases"][label]["cleanup"] for label in labels},
                     "coverage": "Two consecutive real TP2 engines, 16 held-out prompts, formal batch/context/BF16 KV, 64 generated tokens; own-session exit and GPU release joined before the second start. No formal score."})
                self.state["status"] = "preflight_complete"
            except Exception:
                self.state["status"] = "failed"
                raise
            finally:
                self.state["ended_at"] = now()
                save(self.root / "status.json", self.state)
            return
        if self.args.preflight:
            try:
                self.stage("diagnostic", self.args.gpu_pairs[0])
                self.stage("preflight_budget", self.args.gpu_pairs[0])
                files = [self.root / "diagnostic_verified.json", self.root / "preflight_budget/budget.json",
                         self.root / "calibration/diagnostic/c4_ban.pt"]
                if not read(files[0])["passed"] or read(files[1])["status"] != "complete":
                    raise RuntimeError("real preflight proof is incomplete")
                save(self.root / "preflight.json", {"passed": True, "created_at": now(),
                     "artifact_sha256": {str(p): file_sha256(p) for p in files},
                     "coverage": "Two-domain 48-layer calibration and TP2 prefill/64-step decode; formal engine/batch/context and 16 held-out prompts. Not a budget or accuracy result."})
                self.state["status"] = "preflight_complete"
            except Exception:
                self.state["status"] = "failed"
                raise
            finally:
                self.state["ended_at"] = now()
                save(self.root / "status.json", self.state)
            return
        with ThreadPoolExecutor(max_workers=len(self.args.gpu_pairs)) as pool:
            list(pool.map(self.lane, range(len(self.args.gpu_pairs))))
        try:
            report = build_report(self.root)
            self.state["status"] = "complete" if report["status"] == "complete" and not self.state["errors"] else "failed"
        except Exception as error:
            self.state["errors"].append({"phase": "report", "error": str(error)})
            self.state["status"] = "failed"
        self.state["ended_at"] = now()
        save(self.root / "status.json", self.state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--gpu_pairs", type=gpu_pair, nargs="+", required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--verify_restart", action="store_true")
    workflow = Workflow(parser.parse_args())
    workflow.run()
    if workflow.state["status"] == "failed":
        raise SystemExit(1)
