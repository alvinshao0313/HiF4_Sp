"""Verify actual sampling and unchanged single-response scoring semantics."""
from copy import deepcopy
import unittest

from lighteval.metrics.metrics import Metrics
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.lighteval_task import LightevalTask
from lighteval.tasks.requests import Doc

from .tasks import RecordedAvg10, TASKS_TABLE


class ScoringTest(unittest.TestCase):
    def test_actual_generation_count(self):
        for config in TASKS_TABLE:
            self.assertEqual(max(LightevalTask(config).num_samples), 10)

    def test_mmlu_preserves_single_response_final_text(self):
        doc = Doc(query="Pick A", choices=list("ABCD"), gold_index=0)
        text = ["Answer: A"] * 7 + ["Answer: B"] * 3
        response = ModelResponse(text=["<think>Answer: B</think>" + x for x in text], text_post_processed=text)
        expected = [float(Metrics.gpqa_instruct_metric.value.sample_level_fn.compute(
            doc=deepcopy(doc), model_response=ModelResponse(text=[raw], text_post_processed=[final])))
                    for raw, final in zip(response.text, text)]
        value = RecordedAvg10("mmlu_pro").compute(doc, response)
        self.assertEqual(doc.specific["rotation_eval_scoring"]["scores"], expected)
        self.assertAlmostEqual(value, 0.7)

    def test_lcb_real_code_execution(self):
        doc = Doc(query="Add two integers", choices=[""], gold_index=0,
                  specific={"inputs": ["1 2\n", "5 -2\n"], "outputs": ["3\n", "3\n"], "fn_name": None})
        correct = "```python\na, b = map(int, input().split())\nprint(a + b)\n```"
        wrong = "```python\nprint(0)\n```"
        response = ModelResponse(text=[correct] * 6 + [wrong] * 4)
        self.assertAlmostEqual(RecordedAvg10("lcb").compute(doc, response), 0.6)
        self.assertEqual(doc.specific["rotation_eval_scoring"]["scores"], [1] * 6 + [0] * 4)

    def test_missing_responses_fail(self):
        with self.assertRaises(ValueError):
            RecordedAvg10("mmlu_pro").compute(Doc(query="", choices=["A"], gold_index=0), ModelResponse(text=["A"]))


if __name__ == "__main__":
    unittest.main()
