"""R16 completion -> TP2 operator validation -> probe -> one formal G512 run."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from ..qwen3_aisbench_gpqa.common import REPO, check_hashes, now, read, save

PACKAGE = "Native_NVFP4_HiF4_Linear_Puncture.experiments.qwen3_mxfp4_dualscale_tp2"


def stage(root, phase):
    import torch
    from ..nvfp4_gpqa.run import run as evaluate
    from ..qwen3_mxfp4.evidence import summarize_phase
    from ..qwen3_mxfp4.workflow import verify_worker_release
    from ..qwen3_aisbench_gpqa.report import final_submission_summary
    from .report import validate_runtime
    root = Path(root).resolve()
    torch.set_num_threads(8)
    check_hashes(read(root / "preparation.json")["source_sha256"])
    if phase == "formal":
        for name in ("communication.json", "operators.json", "dualscale_g512/probe/runtime_validation.json"):
            if read(root / name)["status"] != "passed":
                raise ValueError(f"Required TP2 validation is missing: {name}")
    os.environ["QWEN3_MXFP4_AUDIT_DIR"] = str(root / "dualscale_g512" / phase / "audit")
    evaluate(root / "dualscale_g512", phase)
    summarize_phase(root / "dualscale_g512", phase, root / "protocol_inputs.json",
                    root / "dualscale_g512" / phase / "summary.json")
    validate_runtime(root, phase)
    verify_worker_release(root / "dualscale_g512" / phase / "audit")
    if phase == "formal":
        final_submission_summary(root, "dualscale_g512")


def run(root):
    from ..qwen3_aisbench_gpqa.continue_gpu03 import wait_existing, completed
    from ..qwen3_mxfp4.workflow import verify_worker_release
    from ..qwen3_mxfp4_rotation_eval.workflow import available
    root = Path(root).resolve()
    parent = root.parent
    target = root / "workflow_status.json"
    if target.exists():
        raise FileExistsError("This finite TP2 workflow already started; no automatic retry")
    preparation = read(root / "preparation.json")
    check_hashes(preparation["source_sha256"])
    if read(root / "cpu_validation.json")["status"] != "passed":
        raise ValueError("CPU validation must pass first")
    auth = read(root / "authorization.json")
    if auth["status"] != "old_controller_stopped" or auth["new_devices"] != "0,1":
        raise ValueError("Unexpected TP2 authorization")
    candidate = read(root / "dualscale_g512/manifest.json")
    if candidate["environment"]["CUDA_VISIBLE_DEVICES"] != "0,1":
        raise ValueError("Only GPU0-1 is selected for TP2")
    state = {"status": "waiting_for_r16", "pid": os.getpid(), "started_at": now(),
             "devices": "0,1", "stages": {}, "formal_started": False}
    save(target, state)
    parent_path = parent / "formal_workflow_status.json"
    parent_state = read(parent_path)
    parent_state.update(status="waiting_for_r16_then_tp2_validation", pid=os.getpid(), tp2_replacement=str(root))
    save(parent_path, parent_state)
    current = "r16x2"
    try:
        print(f"Waiting for R16 PID {auth['protected_r16']['pid']}; G512 requires all TP2 checks", flush=True)
        identity = {"pid": auth["protected_r16"]["pid"], "create_time": auth["protected_r16"]["created"],
                    "cmd": auth["protected_r16"]["command"]}
        exit_code = wait_existing(identity)
        if exit_code not in (None, 0):
            raise RuntimeError(f"Existing R16 stage exited {exit_code}")
        completed(parent, "r16x2")
        verify_worker_release(parent / "r16x2/formal/audit")
        parent_state["stages"]["r16x2"].update(status="completed", exit_code=exit_code,
                                                      completion_verified_at=now())
        save(parent_path, parent_state)
        torchrun = [sys.executable, "-u", "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2", "-m"]
        direct = read(parent / "direct/manifest.json")["source_snapshot"]
        steps = [
            ("communication", torchrun + [PACKAGE + ".validate_communication", "--output", str(root / "communication.json")]),
            ("operators", torchrun + [PACKAGE + ".validate_gpu", "--model", preparation["checkpoint"],
                                     "--direct-model", direct, "--output", str(root / "operators.json")]),
            ("probe", [sys.executable, "-u", "-m", PACKAGE + ".workflow", "stage", "--root", str(root), "--phase", "probe"]),
            ("formal", [sys.executable, "-u", "-m", PACKAGE + ".workflow", "stage", "--root", str(root), "--phase", "formal"]),
        ]
        for current, command in steps:
            available("0,1")
            check_hashes(preparation["source_sha256"])
            log = root / f"{current}_controller.log"
            record = {"status": "starting", "command": command, "started_at": now(), "log": str(log)}
            state["stages"][current] = record
            state["status"] = current
            parent_state.update(status="running", updated_at=now())
            parent_state["stages"]["dualscale_g512"].update(
                status="running" if current == "formal" else "validating_tp2", phase=current,
                devices="0,1", tensor_parallel_size=2, run_root=str(root / "dualscale_g512"))
            save(target, state)
            save(parent_path, parent_state)
            with log.open("x") as stream:
                proc = subprocess.Popen(command, cwd=REPO, stdin=subprocess.DEVNULL, stdout=stream,
                                        stderr=subprocess.STDOUT, start_new_session=True,
                                        env=dict(os.environ, **candidate["environment"]))
                record.update(pid=proc.pid, status="running")
                if current == "formal":
                    state["formal_started"] = True
                save(target, state)
                code = proc.wait()
            record.update(exit_code=code, finished_at=now())
            if code:
                raise RuntimeError(f"{current} exited {code}; see {log}")
            if current in ("communication", "operators"):
                if read(root / f"{current}.json")["status"] != "passed":
                    raise ValueError(f"{current} did not pass")
            elif current == "probe":
                if read(root / "dualscale_g512/probe/runtime_validation.json")["status"] != "passed":
                    raise ValueError("TP2 probe runtime validation failed")
            else:
                completed(root, "dualscale_g512")
            record["status"] = "completed"
            save(target, state)
            print(f"Completed TP2 {current}", flush=True)
        results = {v: completed(parent, v) for v in ("native", "direct", "r64", "fht32", "r16x2")}
        results["dualscale_g512"] = completed(root, "dualscale_g512")
        save(parent / "comparison.json", {
            "status": "awaiting_blind_review" if any(v["pending_review_count"] for v in results.values())
                      else "completed_with_cancellations",
            "results": results, "cancelled": ["dualscale_g256"], "stopped": ["direct_uos_c7p25"],
            "tensor_parallel_size": {v: 2 if v == "dualscale_g512" else 4 for v in results},
            "protocol": "AISBench GPQA x10, BF16 KV; authorized G512 TP2; other candidates TP4",
        })
        state["status"] = "completed"
        parent_state["status"] = "completed_with_cancellations"
        parent_state["stages"]["dualscale_g512"].update(status="completed", finished_at=now())
    except Exception as error:
        state.update(status="failed", error=str(error), failed_stage=current)
        if current in state["stages"]:
            state["stages"][current]["status"] = "failed"
        parent_state.update(status="failed", error=str(error))
        parent_state["stages"]["dualscale_g512"].update(status="failed", error=str(error))
        raise
    finally:
        state["finished_at"] = now()
        parent_state["updated_at"] = now()
        save(target, state)
        save(parent_path, parent_state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "stage"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--phase", choices=("probe", "formal"))
    args = parser.parse_args()
    if args.action == "stage":
        if not args.phase:
            parser.error("stage requires --phase")
        stage(args.root, args.phase)
    else:
        run(args.root)
