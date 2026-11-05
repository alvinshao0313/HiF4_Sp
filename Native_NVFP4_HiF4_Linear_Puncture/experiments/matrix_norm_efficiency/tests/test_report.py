import unittest

from ..config import TASKS
from ..report import accuracy_summary


class AccuracyContract(unittest.TestCase):
    def test_gains_cannot_cancel_aime_loss(self):
        native = {task: {"0": .5} for task in TASKS}
        candidate = {task: {"0": .6} for task in TASKS}
        candidate["aime24"] = {"0": .4}
        result = accuracy_summary(native, candidate, repeats=20)
        self.assertAlmostEqual(result["mean_positive_loss_pp"], 2.)
        self.assertFalse(result["accuracy_target_met"])

    def test_years_have_separate_equal_weight(self):
        native = {task: {"0": .5} for task in TASKS}
        candidate = {task: {"0": .5} for task in TASKS}
        candidate["aime25"] = {"0": .48}
        result = accuracy_summary(native, candidate, repeats=20)
        self.assertAlmostEqual(result["mean_positive_loss_pp"], .4)
        self.assertTrue(result["accuracy_target_met"])

    def test_missing_task_or_pair_cannot_pass(self):
        scores = {task: {"0": .5} for task in TASKS}
        with self.assertRaises(ValueError):
            accuracy_summary(scores, {k: v for k, v in scores.items() if k != "aime24"})
        with self.assertRaises(ValueError):
            accuracy_summary(scores, {**scores, "gsm8k": {"1": .5}})
