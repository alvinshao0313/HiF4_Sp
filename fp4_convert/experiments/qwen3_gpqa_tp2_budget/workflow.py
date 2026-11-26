"""One candidate: formal-configuration short check followed by GPQA x10."""
import argparse
import os
from pathlib import Path

from ..qwen3_aisbench_gpqa.common import check_hashes, now, read, save


def run(root, variant):
    import torch
    from ..nvfp4_gpqa.run import run as evaluate
    from ..qwen3_mxfp4.evidence import summarize_phase
    from ..qwen3_mxfp4.workflow import verify_worker_release
    from ..qwen3_mxfp4_rotation_eval.workflow import available
    from ..qwen3_aisbench_gpqa.report import final_submission_summary
    from .report import validate_runtime
    root = Path(root).resolve()
    target = root / variant / "workflow_status.json"
    if target.exists():
        raise FileExistsError("Candidate already started; no automatic retries")
    prep = read(root / "preparation.json")
    check_hashes(prep["source_sha256"])
    required = root / "uos_operators.json" if variant == "direct_uos_c7p25" else Path(prep["previous_g512"]) / "operators.json"
    if read(required)["status"] != "passed":
        raise ValueError("Required operator validation did not pass")
    manifest = read(root / variant / "manifest.json")
    torch.set_num_threads(8)
    state = {"status": "running", "pid": os.getpid(), "variant": variant, "started_at": now(),
             "devices": manifest["environment"]["CUDA_VISIBLE_DEVICES"], "stages": {}}
    save(target, state)
    try:
        for phase in ("probe", "formal"):
            if phase == "formal" and read(root / variant / "probe/runtime_validation.json")["status"] != "passed":
                raise ValueError("Formal TP2 probe did not pass")
            available(state["devices"])
            check_hashes(prep["source_sha256"])
            state.update(status=phase)
            state["stages"][phase] = {"status": "running", "started_at": now()}
            save(target, state)
            os.environ["QWEN3_MXFP4_AUDIT_DIR"] = str(root / variant / phase / "audit")
            evaluate(root / variant, phase)
            summarize_phase(root / variant, phase, root / "protocol_inputs.json", root / variant / phase / "summary.json")
            validate_runtime(root, variant, phase)
            verify_worker_release(root / variant / phase / "audit")
            if phase == "formal":
                final_submission_summary(root, variant)
            state["stages"][phase].update(status="completed", finished_at=now())
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
    parser.add_argument("--variant", required=True, choices=("direct_uos_c7p25", "dualscale_g512"))
    args = parser.parse_args()
    run(args.root, args.variant)
