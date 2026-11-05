"""Full TP2 downstream metrics, pinned to B prompts and the corrected final-answer scorer."""
import argparse
from pathlib import Path
import sys

from ..downstream_accuracy_recovery import evaluate as shared
from ..downstream_accuracy_recovery.protocol import protocol_for, reasoning_details
from ..downstream_accuracy_recovery.reporting import audit_prompts, validate_metrics
from ..module_design_ablation.aime import aime_protocol, audit_aime, validate_result
from ..module_design_ablation.evaluate_aime import model_view
from ..reasoning_trace_calibration.evaluation import AimeReviewRequired, correct_aime, run_reasoning
from ..reasoning_trace_calibration.pinned import install
from .artifacts import read_json, verify, write_json
from .config import CANDIDATE, NATIVE, ORIGIN


def evaluate(root, task):
    root, manifest = verify(root)
    source = root / "models" / CANDIDATE
    if not read_json(source / "export.json")["complete"]:
        raise ValueError("complete formal export is required")
    target = root / "evaluation" / CANDIDATE / task
    install()
    model = model_view(source, target / "model")
    reference = Path(manifest["source_run"]) / "evaluation" / ORIGIN / task
    if task == "aime25":
        audit = audit_aime(model)
        if audit != read_json(reference / "prompt_audit.json"):
            raise ValueError("AIME prompts/tokens/golds differ from B")
        write_json(audit, target / "prompt_audit.json")
        spec, metadata = shared.check_model(model, "hif4")
        result = run_reasoning(model, target / "vllm_run", aime_protocol(smoke=False), "hif4", spec)
        result.update(protocol=aime_protocol(smoke=False), model_kind="hif4", model=metadata)
        validate_result(result, audit=audit, model_kind="hif4", smoke=False)
        write_json(result, target / "metrics.json")
        correct_aime(target, result)
        return
    audit = audit_prompts({"native_nvfp4": NATIVE, CANDIDATE: model}, target / "prompt_audit.json")
    prior = read_json(reference / "prompt_audit.json")["models"][ORIGIN]
    if audit["models"][CANDIDATE] != prior:
        raise ValueError("formal prompt/token audit differs from B")
    previous = shared.run_reasoning
    shared.run_reasoning = run_reasoning
    try:
        result = shared.evaluate(model, target, task, model_kind="hif4", smoke=False, tensor_parallel_size=2)
    finally:
        shared.run_reasoning = previous
    scores = validate_metrics(result, task, smoke=False, model_kind="hif4",
                              prompt_reference=None if task == "arc" else prior["tasks"][task])
    if task != "arc":
        _, details = reasoning_details(result, task)
        metric = protocol_for(task, smoke=False)["metric"]
        rows = {str(r["doc"]["id"]): r["metric"][metric] for r in details}
        write_json(rows, target / "paired_scores.json")
    write_json(scores, target / "verified_scores.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--task", choices=("arc", "mmlu_pro", "lcb", "gsm8k", "aime25"), required=True)
    parser.add_argument("--rescore-aime", action="store_true")
    args = parser.parse_args()
    try:
        if args.rescore_aime:
            root, _ = verify(args.run)
            target = root / "evaluation" / CANDIDATE / "aime25"
            correct_aime(target, read_json(target / "metrics.json"))
        else:
            evaluate(args.run, args.task)
    except AimeReviewRequired as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(21) from error
