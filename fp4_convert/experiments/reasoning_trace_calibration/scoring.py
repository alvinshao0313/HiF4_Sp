"""Final-submission scoring, shared by AIME correction and math development."""
from __future__ import annotations

import json
import re


def boxes(text):
    values = []
    for match in re.finditer(r"\\(?:boxed|fbox)\s*\{", text):
        start = end = match.end()
        depth = 1
        while end < len(text) and depth:
            depth += (text[end] == "{") - (text[end] == "}")
            end += 1
        if depth == 0:
            values.append(text[start:end - 1].strip())
    return values


def final_text(raw):
    # Qwen's prompt has already opened <think>, so raw need not contain it.
    return raw.rsplit("</think>", 1)[1].strip() if "</think>" in raw else ""


def equivalent(gold, expression):
    from lighteval.metrics.metrics import Metrics
    from lighteval.models.model_output import ModelResponse
    from lighteval.tasks.requests import Doc
    metric = Metrics.avg_at_n_math(sample_params={"n": 5}).sample_level_fn.compute_score
    # MATH golds may be radicals/tuples, not just AIME integers. Without
    # delimiters the extractor can read only the leading coefficient or nothing.
    doc = Doc(query="", choices=["$\\boxed{" + str(gold) + "}$"], gold_index=0)
    result = float(metric(doc, ModelResponse(text=["$\\boxed{" + expression + "}$"])))
    if result not in (0., 1.):
        raise ValueError("math equivalence must produce a binary result")
    return int(result)


def aime_submission(raw, gold):
    final = final_text(raw)
    values = boxes(final)
    if not final:
        return {"score": 0, "decision": "no_final_submission", "answer": None, "review": False}
    if not values:
        return {"score": None, "decision": "review_missing_box", "answer": None, "review": True}
    answer = values[-1]
    normalized = answer.replace(r"\,", "").replace(" ", "")
    integer = bool(re.fullmatch(r"[+-]?\d+(?:\.0+)?", normalized))
    score = int(int(float(normalized)) == int(gold)) if integer else equivalent(gold, answer)
    # Multiple different submissions and expression-only submissions are retained
    # for explicit review. The preliminary score cannot enter aggregation.
    conflict = len(set(values)) > 1
    return {"score": score, "decision": "last_final_boxed_integer" if integer else "final_boxed_expression",
            "answer": answer, "review": conflict or not integer, "boxes": values}


def math_score(raw, solution):
    submitted = boxes(final_text(raw))
    markers = list(re.finditer(r"\\(?:boxed|fbox)\b", solution))
    if not markers:
        raise ValueError("source math reference has no boxed answer")
    tail = solution[markers[-1].end():].lstrip()
    if tail.startswith("{"):
        values = boxes(solution[markers[-1].start():])
        if not values:
            raise ValueError("source math reference has an unclosed boxed argument")
        gold = values[0]
    else:
        # TeX permits a single token as an unbraced macro argument. The pinned
        # MATH train rows 888/1011 use exactly this syntax (boxed 2 / boxed 9).
        token = re.match(r"\\[A-Za-z]+|[^\\{}\s]", tail)
        if token is None:
            raise ValueError("source math reference has no valid boxed argument")
        gold = token.group()
    return equivalent(gold, submitted[-1]) if submitted else 0


def code_score(raw, tests):
    from lighteval.tasks.tasks.lcb.codegen_metrics import check_correctness, extract_code
    code = extract_code(final_text(raw))
    inputs, outputs = [], []
    for kind in ("public_tests", "private_tests", "generated_tests"):
        values = tests[kind]
        if len(values["input"]) != len(values["output"]):
            raise ValueError("source code tests have inconsistent coverage")
        inputs.extend(values["input"])
        outputs.extend(values["output"])
    if not inputs:
        raise ValueError("source code problem has no judging tests")
    if not code:
        return 0
    result = check_correctness({"input_output": json.dumps({"inputs": inputs, "outputs": outputs, "fn_name": None})},
                               code, timeout=6)
    if not result or len(result) > len(inputs):
        raise ValueError("invalid code judging result coverage")
    # The official checker stops on the first wrong answer/compile error/timeout.
    # A short failing result is valid evidence for pass@1=0.
    if any(value != 1 for value in result):
        return 0
    if len(result) != len(inputs):
        raise ValueError("checker reported success without running all tests")
    return 1
