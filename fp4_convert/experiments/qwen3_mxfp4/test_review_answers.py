"""Final-answer review regressions: formatting, conflicts, and thinking isolation."""
import unittest

from .review_answers import parse_final


class ReviewTests(unittest.TestCase):
    def test_final_formats(self):
        for text, answer in [
            ('{"answer": "A"}', 'A'), ('**Answer**: "B"', 'B'), ('`answer`: "C"', 'C'),
            (r'$$\text{answer}: \text{D}$$', 'D'), ('**answer**: "a"', 'A'),
            (r'\boxed{\text{B}}', 'B'), ('```\n"C"\n```', 'C'),
            ('The correct answer is **D**.', 'D'),
        ]:
            with self.subTest(text=text):
                self.assertEqual(parse_final(text)['answer'], answer)

    def test_only_completed_final_submission(self):
        self.assertEqual(parse_final('<think>"answer": "A"</think>**answer**: "B"')['answer'], 'B')
        self.assertIsNone(parse_final('<think>"answer": "A"')['answer'])
        self.assertIsNone(parse_final('<think>"answer": "A"</think>No answer.')['answer'])

    def test_conflicts_require_review(self):
        for text in ['"answer":"A"\n"answer":"C"', 'The correct option is D.\n' + r'\boxed{B}']:
            self.assertEqual(parse_final(text)['status'], 'review_conflict')

    def test_no_answer_from_math_or_option_list(self):
        for text in [r'\boxed{4\pi}', 'A) first\nB) second\nC) third\nD) fourth', 'The answer is a reaction product.']:
            self.assertIsNone(parse_final(text)['answer'])


if __name__ == '__main__':
    unittest.main()
