"""Check actual repetition, final-text scoring, and executable LCB correctness."""
from copy import deepcopy
import unittest

from lighteval.metrics.metrics import Metrics
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.lighteval_task import LightevalTask
from lighteval.tasks.requests import Doc

from .tasks import RecordedAvg5, TASKS_TABLE, mmlu_prompt


class ScoringTest(unittest.TestCase):
    def test_actual_generation_count(self):
        for config in TASKS_TABLE:
            self.assertEqual(max(LightevalTask(config).num_samples), 5)

    def test_mmlu_final_text_and_high_letter(self):
        doc = Doc(query="Pick J", choices=list("ABCDEFGHIJ"), gold_index=9)
        text = ["ANSWER: J"] * 3 + ["ANSWER: B"] * 2
        response = ModelResponse(text=["<think>Answer: B</think>" + x for x in text], text_post_processed=text)
        expected = [float(Metrics.gpqa_instruct_metric.value.sample_level_fn.compute(
            doc=deepcopy(doc), model_response=ModelResponse(text=[raw], text_post_processed=[final])))
            for raw, final in zip(response.text, text)]
        self.assertAlmostEqual(RecordedAvg5("mmlu_pro").compute(doc, response), 0.6)
        self.assertEqual(doc.specific["combination_eval_scoring"]["scores"], expected)

    def test_lcb_real_code_execution(self):
        doc = Doc(query="Add integers", choices=[""], gold_index=0,
                  specific={"inputs": ["1 2\n", "5 -2\n"], "outputs": ["3\n", "3\n"], "fn_name": None})
        correct = "```python\na, b = map(int, input().split())\nprint(a + b)\n```"
        wrong = "```python\nprint(0)\n```"
        response = ModelResponse(text=[correct] * 3 + [wrong] * 2)
        self.assertAlmostEqual(RecordedAvg5("lcb").compute(doc, response), 0.6)
        self.assertEqual(doc.specific["combination_eval_scoring"]["scores"], [1, 1, 1, 0, 0])

    def test_missing_responses_fail(self):
        with self.assertRaises(ValueError):
            RecordedAvg5("mmlu_pro").compute(Doc(query="", choices=["A"], gold_index=0), ModelResponse(text=["A"]))

    def test_na_does_not_shift_option_letters(self):
        doc = mmlu_prompt({"question": "Pick C", "options": ["one", "N/A", "three"],
                           "answer_index": 2, "category": "other", "question_id": 1})
        self.assertIn("A. one\nC. three", doc.query)
        self.assertEqual(doc.get_golds(), ["C"])


if __name__ == "__main__":
    unittest.main()
