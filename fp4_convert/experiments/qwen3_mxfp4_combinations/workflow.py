"""Two finite TP2 lanes: qualify all candidates before starting formal queues."""
import argparse
import os
from pathlib import Path

from ..qwen3_aisbench_gpqa.common import check_hashes, now, read, save
from .format import VARIANTS


def run(root, lane, phase):
    import torch
    from ..nvfp4_gpqa.run import run as evaluate
    from ..qwen3_mxfp4.evidence import summarize_phase
    from ..qwen3_mxfp4.workflow import verify_worker_release
    from ..qwen3_mxfp4_rotation_eval.workflow import available
    from ..qwen3_aisbench_gpqa.report import final_submission_summary
    from .report import validate_runtime
    root = Path(root).resolve()
    target = root / f"{phase}_lane{lane}.json"
    if target.exists():
        raise FileExistsError("This finite phase already started; no automatic retry")
    prep = read(root / "preparation.json")
    check_hashes(prep["source_sha256"])
    if phase == "formal":
        for i in (0, 1):
            if read(root / f"qualify_lane{i}.json")["status"] != "completed":
                raise ValueError("All four candidates must qualify before formal evaluation")
    torch.set_num_threads(8)
    state = {"status": "running", "pid": os.getpid(), "started_at": now(), "stages": {}}
    save(target, state)
    try:
        for variant in VARIANTS[lane::2]:
            candidate = root / variant
            manifest = read(candidate / "manifest.json")
            devices = manifest["environment"]["CUDA_VISIBLE_DEVICES"]
            state.update(current_variant=variant, devices=devices)
            state["stages"][variant] = {"status": "running", "started_at": now()}
            save(target, state)
            available(devices)
            check_hashes(prep["source_sha256"])
            operators = read(candidate / "operators.json")
            if operators["status"] != "passed":
                raise ValueError(f"Operator validation did not pass: {variant}")
            check_hashes(operators["source_sha256"])
            if phase == "formal" and read(candidate / "probe/runtime_validation.json")["status"] != "passed":
                raise ValueError("Probe runtime validation missing")
            eval_phase = "probe" if phase == "qualify" else "formal"
            available(devices)
            os.environ["QWEN3_MXFP4_AUDIT_DIR"] = str(candidate / eval_phase / "audit")
            evaluate(candidate, eval_phase)
            summarize_phase(candidate, eval_phase, root / "protocol_inputs.json", candidate / eval_phase / "summary.json")
            validate_runtime(root, variant, eval_phase)
            verify_worker_release(candidate / eval_phase / "audit")
            if phase == "formal":
                final_submission_summary(root, variant)
            state["stages"][variant].update(status="completed", finished_at=now())
            save(target, state)
        state["status"] = "completed"
    except Exception as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        state["finished_at"] = now()
        save(target, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--lane", type=int, choices=(0, 1), required=True)
    parser.add_argument("--phase", choices=("qualify", "formal"), required=True)
    args = parser.parse_args()
    run(args.root, args.lane, args.phase)
