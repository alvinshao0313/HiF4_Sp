"""The development decision uses complete evidence and a fixed tie rule."""
from __future__ import annotations

import unittest

from ..config import CANDIDATES
from ..development import choose_candidate, score_summary


class DevelopmentContractTests(unittest.TestCase):
    def test_score_has_priority_then_time_then_declared_order(self):
        names = list(CANDIDATES)
        scores = dict.fromkeys(names, .5)
        seconds = dict.fromkeys(names, 100.)
        self.assertEqual(choose_candidate(scores, seconds), names[0])
        seconds[names[-1]] = 50.
        self.assertEqual(choose_candidate(scores, seconds), names[-1])
        scores[names[0]] = .6
        self.assertEqual(choose_candidate(scores, seconds), names[0])

    def test_missing_scores_or_unmeasured_times_cannot_choose_winner(self):
        scores = dict.fromkeys(CANDIDATES, .5)
        seconds = dict.fromkeys(CANDIDATES, 100.)
        incomplete = dict(scores)
        incomplete.pop(next(iter(CANDIDATES)))
        with self.assertRaises(ValueError):
            choose_candidate(incomplete, seconds)
        seconds[next(iter(CANDIDATES))] = float("nan")
        with self.assertRaises(ValueError):
            choose_candidate(scores, seconds)

    def test_domain_weighting_and_exact_response_membership(self):
        questions = [{"id": "m0", "domain": "math"}, {"id": "m1", "domain": "math"},
                     {"id": "c0", "domain": "code"}]
        rows = [{**question, "score": int(question["domain"] == "math")} for question in questions]
        self.assertEqual(score_summary(rows, questions), ({"math": 1., "code": 0.}, .5))
        for invalid in (rows[:-1], rows[:2] + [rows[0]], rows[:2] + [{**rows[2], "score": .5}]):
            with self.assertRaises(ValueError):
                score_summary(invalid, questions)


if __name__ == "__main__":
    unittest.main()
