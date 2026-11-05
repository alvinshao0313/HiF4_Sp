"""Coverage/aggregation regression tests; the fixtures are unit-test data only."""
import copy
import json
import math

import pytest

from ..aime import COUNT, GENERATIONS, METRIC, TASK_NAME, digest, validate_details, validate_sample_limit
from ..evaluate_aime import model_view


def example():
    scores = [0, 0.2, 0.4, 0.6, 0.8, 1] * 5
    rows = [{"doc": {"id": str(index)}, "gold": [str(index)],
             "model_response": {"input": f"question {index}",
                                "text_post_processed": [f"answer {index}"] * GENERATIONS},
             "metric": {METRIC: score}} for index, score in enumerate(scores)]
    documents = {row["doc"]["id"]: {"prompt_sha256": digest(row["model_response"]["input"]),
                                      "gold_sha256": digest(row["gold"])} for row in rows}
    audit = {"task_name": TASK_NAME, "metric": METRIC, "count": COUNT,
             "generations_per_doc": GENERATIONS, "documents": documents,
             "documents_sha256": digest(documents)}
    return rows, audit


def test_complete_coverage_and_question_mean():
    rows, audit = example()
    result = validate_details(rows, 0.5, audit=audit)
    assert result["question_count"] == 30
    assert result["generation_count"] == 150
    assert result["score"] == 0.5


def test_smoke_still_requires_five_generations():
    rows, audit = example()
    result = validate_details(rows[:1], 0, audit=audit, smoke=True)
    assert result["generation_count"] == 5
    rows[0]["model_response"]["text_post_processed"].pop()
    with pytest.raises(ValueError, match="exactly five"):
        validate_details(rows[:1], 0, audit=audit, smoke=True)


def test_full_aime_coverage_allows_larger_shared_sample_cap():
    for limit in (None, 30, 300):
        validate_sample_limit(limit, smoke=False)
    for limit in (1, 29, 0.5, True, float("inf")):
        with pytest.raises(ValueError, match="sample limit"):
            validate_sample_limit(limit, smoke=False)
    validate_sample_limit(1, smoke=True)
    with pytest.raises(ValueError, match="sample limit"):
        validate_sample_limit(None, smoke=True)


@pytest.mark.parametrize("corruption,match", [
    ("missing", "expected 30"), ("duplicate", "duplicate"), ("unknown_id", "IDs differ"),
    ("prompt", "prompt differs"), ("gold", "gold differs"),
    ("metric", "requires avg"), ("mean", "aggregate differs"),
    ("nan", "finite fraction"), ("boolean", "finite fraction"),
    ("not_five_binary", "five binary"), ("one_generation", "exactly five"),
])
def test_invalid_results_rejected(corruption, match):
    rows, audit = example()
    score = 0.5
    if corruption == "missing":
        rows.pop()
    elif corruption == "duplicate":
        rows[-1] = copy.deepcopy(rows[0])
    elif corruption == "unknown_id":
        rows[0]["doc"]["id"] = "999"
    elif corruption == "prompt":
        rows[0]["model_response"]["input"] += " changed template"
    elif corruption == "gold":
        rows[0]["gold"] = ["different dataset revision"]
    elif corruption == "metric":
        rows[0]["metric"] = {"pass@5": 0}
    elif corruption == "mean":
        score = 0.6
    elif corruption == "nan":
        rows[0]["metric"][METRIC] = float("nan")
    elif corruption == "boolean":
        rows[0]["metric"][METRIC] = False
    elif corruption == "not_five_binary":
        rows[0]["metric"][METRIC] = 0.3
    elif corruption == "one_generation":
        rows[0]["model_response"]["text_post_processed"] = ["one answer"]
    with pytest.raises(ValueError, match=match):
        validate_details(rows, score, audit=audit)


def test_official_math_metric_is_average_not_pass_or_majority():
    from lighteval.models.model_output import ModelResponse
    from lighteval.tasks.requests import Doc
    from tasks.aime import aime25_avg5

    doc = Doc(query="2 = ?", choices=["2"], gold_index=0)
    response = ModelResponse(text=[r"\boxed{2}", r"\boxed{2}", r"\boxed{3}",
                                   r"\boxed{3}", r"\boxed{2}"])
    metric = aime25_avg5.metrics[0]
    assert metric.metric_name == METRIC
    assert metric.sample_level_fn.num_samples() == 5
    assert math.isclose(metric.sample_level_fn.compute(doc, response), 0.6)


def test_model_view_preserves_model_assets_but_not_prediction_cache(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "weights.safetensors"}}))
    for name in ("weights.safetensors", "config.json", "tokenizer.json", "hif4_runtime_spec.pt"):
        (source / name).write_text("unit-test asset")
    (source / "prediction_cache").mkdir()
    (source / "prediction_cache/old.json").write_text("unit-test cache")
    (source / "old_results.json").write_text("unit-test result")
    target = model_view(source, tmp_path / "view")
    assert (target / "weights.safetensors").resolve() == source / "weights.safetensors"
    assert (target / "hif4_runtime_spec.pt").is_symlink()
    assert not (target / "prediction_cache").exists()
    assert not (target / "old_results.json").exists()
    with pytest.raises(FileExistsError):
        model_view(source, target)
