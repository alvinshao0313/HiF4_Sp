"""Focused protocol checks plus replay of the 600 previously reviewed AIME answers."""
from __future__ import annotations

import argparse
from pathlib import Path
import random
import unittest

import numpy as np
import torch

from .artifacts import digest, read_json, sha256, write_json
from .config import COHORT, PRIMARY, REPO
from .isolation import cluster_questions, grams, near_edges, normalize, split_pools
from .report import loss, paired
from .scoring import aime_submission


class ProtocolTests(unittest.TestCase):
    def test_clean_s1k_retains_slots_and_refills_fixed_unused_order(self):
        from .clean_s1k import choose
        order = random.Random(42).sample(range(1000), k=1000)
        original = {"train": order[:128], "val": order[128:160]}
        clusters = dict(enumerate(range(1000)))
        # Skip a blocked replacement and another from an already retained cluster.
        blocked = {order[3], order[133], order[160]}
        clusters[order[161]] = clusters[order[0]]
        selected, dropped, replaced = choose(original, clusters, blocked, 1000)
        self.assertEqual([len(selected[s]) for s in ("train", "val")], [128, 32])
        self.assertEqual(selected["train"][3], order[162])
        self.assertEqual(selected["val"][5], order[163])
        self.assertEqual(len(dropped), 2)
        for split in original:
            for slot, index in enumerate(original[split]):
                if index not in blocked:
                    self.assertEqual(selected[split][slot], index)
        a = {clusters[i] for i in selected["train"]}
        b = {clusters[i] for i in selected["val"]}
        self.assertFalse(a & b or a & blocked or b & blocked)
        self.assertEqual((selected, dropped, replaced), choose(original, clusters, blocked, 1000))

    def test_clean_s1k_removes_cross_split_source_cluster(self):
        from .clean_s1k import choose
        order = random.Random(42).sample(range(1000), k=1000)
        original = {"train": order[:128], "val": order[128:160]}
        clusters = dict(enumerate(range(1000)))
        clusters[order[128]] = clusters[order[0]]
        selected, dropped, _ = choose(original, clusters, set(), 1000)
        self.assertEqual(selected["train"], original["train"])
        self.assertEqual(selected["val"][0], order[160])
        self.assertEqual(dropped[0]["reason"], "cross_split_cluster")

    def test_native_prompt_token_ids_and_reference_whitelist(self):
        from transformers import AutoTokenizer
        from .config import NATIVE
        from .generate import prompt
        tokenizer = AutoTokenizer.from_pretrained(NATIVE, local_files_only=True)
        tokens = prompt(tokenizer, {"domain": "math", "question": "Compute 2 + 2.",
                                    "solution": "REFERENCE_MUST_NOT_ENTER", "hidden_tests": "PRIVATE"})
        self.assertIsInstance(tokens, list)
        self.assertTrue(tokens and all(isinstance(t, int) for t in tokens))
        text = tokenizer.decode(tokens)
        self.assertIn("Compute 2 + 2.", text)
        self.assertNotIn("REFERENCE_MUST_NOT_ENTER", text)
        self.assertNotIn("PRIVATE", text)

    def test_prefix_join_matches_brute_force(self):
        rng = random.Random(42)
        texts = ["".join(rng.choices("abcdefg0123 ", k=80)) for _ in range(24)]
        texts += [s[:-2] + "zz" for s in texts] + ["x", "xy", "abcde"]
        observed = {(a, b) for a, b, _ in near_edges(texts)}
        expected = set()
        for b, right in enumerate(texts):
            for a, left in enumerate(texts[:b]):
                l, r = grams(left), grams(right)
                if 5 * len(l & r) >= 4 * len(l | r):
                    expected.add((a, b))
        self.assertEqual(observed, expected)

    def test_numeric_and_source_id_clusters(self):
        questions = ["Find the total when 123 people each buy 45 apples.", "Find the total when 789 people each buy 16 apples."]
        self.assertEqual(normalize(questions[0], mask_numbers=True), normalize(questions[1], mask_numbers=True))
        records = [{"id": "a", "role": "candidate", "domain": "math", "question": questions[0], "identities": []},
                   {"id": "t", "role": "test", "domain": "math", "question": questions[1], "identities": []},
                   {"id": "z", "role": "candidate", "domain": "code", "question": "A completely renamed problem", "identities": ["cf:1:A"]},
                   {"id": "u", "role": "test", "domain": "code", "question": "Different statement", "identities": ["cf:1:A"]}]
        groups, edges = cluster_questions(records)
        self.assertEqual(sorted(sorted(r["id"] for r in group) for group in groups), [["a", "t"], ["u", "z"]])

    def test_cluster_split_independence(self):
        groups = [[{"id": f"{domain}:{i}", "domain": domain, "role": "candidate"}] for domain in ("math", "code") for i in range(200)]
        pools, _, leaks = split_pools(groups)
        self.assertFalse(leaks)
        for domain in pools:
            sets = [set(r["id"] for r in pools[domain][split]) for split in ("train", "val", "dev")]
            self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
            self.assertEqual(len(sets[2]), 64)

    def test_router_full_softmax_tail_and_gradient(self):
        from ..non_equivalent_reconstruction.losses import masked_router_loss
        torch.manual_seed(42)
        s = torch.randn(2, 7, 128, requires_grad=True)
        t = torch.randn_like(s) * 2
        mask = torch.ones(2, 7, dtype=torch.bool)
        mask[0, -2:] = False
        actual = masked_router_loss(s, t, mask, kind="top_mass", top_k=8, temperature=1.)
        p, q = s[mask].softmax(-1), t[mask].softmax(-1)
        ids = q.topk(8, dim=-1).indices
        pi, qi = p.gather(-1, ids), q.gather(-1, ids)
        reference = ((qi * (qi.log() - pi.log())).sum(-1)
                     + (1 - qi.sum(-1)) * ((1 - qi.sum(-1)).log() - (1 - pi.sum(-1)).log())).mean()
        torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-6)
        a = torch.autograd.grad(actual, s, retain_graph=True)[0]
        b = torch.autograd.grad(reference, s)[0]
        torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-5)
        self.assertEqual(int(torch.count_nonzero(a[~mask])), 0)

    def test_pinned_mmlu_reconstructs_canonical_full_prompts(self):
        from transformers import AutoTokenizer
        from lighteval.tasks.registry import Registry
        from lighteval.tasks.lighteval_task import LightevalTask
        from lighteval.tasks.prompt_manager import PromptManager
        from ..downstream_accuracy_recovery.reporting import _digest
        from .config import NATIVE
        from .pinned import install
        original = LightevalTask.download_dataset_worker
        try:
            install()
            task = Registry(tasks="mmlu_pro|0").load_tasks()["mmlu_pro|0"]
            task.dataset = LightevalTask.download_dataset_worker(task)
            docs = task.get_docs(max_samples=None)
            self.assertEqual(len(docs), 12032)
            tokenizer = AutoTokenizer.from_pretrained(NATIVE, local_files_only=True)
            manager = PromptManager(True, tokenizer, enable_thinking=True)
            prompts = [manager.prepare_prompt(doc) for doc in docs]
            tokens = [tokenizer.encode(text, add_special_tokens=False) for text in prompts]
            actual = {"count": len(docs), "generations_per_doc": 1, "max_prompt_tokens": max(map(len, tokens)),
                      "document_ids_sha256": _digest([str(doc.id) for doc in docs]),
                      "prompt_set_sha256": _digest(sorted(prompts)), "prompt_sha256": _digest(prompts),
                      "token_sha256": _digest(tokens)}
            reference = read_json(COHORT / "evaluation/native_nvfp4/mmlu_pro/prompt_audit.json")
            self.assertEqual(actual, reference["models"]["native_nvfp4"]["tasks"]["mmlu_pro"])
        finally:
            LightevalTask.download_dataset_worker = staticmethod(original)

    def test_scoring_uses_final_submission(self):
        self.assertEqual(aime_submission("earlier \\boxed{42}</think>Final: \\boxed{19}", "19")["score"], 1)
        self.assertEqual(aime_submission("reasoning \\boxed{19}", "19")["score"], 0)
        self.assertTrue(aime_submission("</think>Maybe \\boxed{19}, or \\boxed{20}", "19")["review"])

    def test_math_gold_radicals_tuples_and_equivalence(self):
        from .scoring import math_score
        for answer in ("4", r"\frac{1}{2}", r"2\sqrt{3}", r"\left(2,3\right)"):
            self.assertEqual(math_score("</think>$\\boxed{" + answer + "}$", "$\\boxed{" + answer + "}$"), 1, answer)
        self.assertEqual(math_score(r"</think>$\boxed{\sqrt{12}}$", r"$\boxed{2\sqrt{3}}$"), 1)
        self.assertEqual(math_score(r"</think>$\boxed{\frac{2}{3}}$", r"$\boxed{\frac{1}{2}}$"), 0)

    def test_pinned_math_unbraced_reference_tokens(self):
        from .config import ROOT
        from .scoring import math_score
        rows = read_json(ROOT / "results/reasoning_trace_calibration/run_20261016T132000Z/data_sources/math/train-00000-of-00001.parquet.json")
        self.assertEqual(math_score(r"</think>$\boxed{2}$", rows[888]["solution"]), 1)
        self.assertEqual(math_score(r"</think>$\boxed{9}$", rows[1011]["solution"]), 1)

    def test_real_code_checker_success_and_early_failure(self):
        from .scoring import code_score
        tests = {"public_tests": {"input": ["1 2\n", "3 4\n"], "output": ["3\n", "7\n"]},
                 "private_tests": {"input": [], "output": []}, "generated_tests": {"input": [], "output": []}}
        self.assertEqual(code_score("</think>```python\na,b=map(int,input().split());print(a+b)\n```", tests), 1)
        self.assertEqual(code_score("</think>```python\nprint(0)\n```", tests), 0)
        self.assertEqual(code_score("</think>```python\nthis is not valid python!!!\n```", tests), 0)

    def test_paired_loss_bootstrap(self):
        left = {k: {str(i): float(i % 2) for i in range(30)} for k in PRIMARY}
        same = paired(left, left, repeats=100)
        self.assertEqual(same["mean_positive_loss_change_pp"], 0.)
        self.assertEqual(same["paired_95pct_pp"], [0., 0.])
        self.assertEqual(loss({k: 1. for k in PRIMARY})["mean_positive_loss_pp"], 0.)

    def test_replay_reviewed_aime_600_answers(self):
        import pyarrow.parquet as pq
        from loguru import logger
        logger.remove()
        corrected = read_json(COHORT / "aime_scoring_recomputed_20261015.json")
        original = read_json(COHORT / "aime_scoring_audit_20261015.json")["runs"]
        original["fixed_r64/formal"] = read_json(COHORT / "aime_scoring_fixed_r64_20261015.json")
        expected = {"native_nvfp4/formal": 106, "fixed_r64/formal": 105, "matrix_norm/formal": 99, "existing_two_stage/formal": 92}
        for name, count in expected.items():
            run = original[name]
            paths = [(REPO / key, value) for key, value in run["artifacts_sha256"].items() if key.endswith("GENERATIVE.parquet")]
            self.assertEqual(len(paths), 1)
            path, checksum = paths[0]
            self.assertEqual(sha256(path), checksum)
            rows = pq.read_table(path).to_pylist()
            text = {(str(r["sample_id"]), i + 1): raw for r in rows for i, raw in enumerate(r["sample"]["text"])}
            total = 0
            for reviewed in corrected["runs"][name]["per_response"]:
                raw = text[(reviewed["question_id"], reviewed["response_index"])]
                self.assertEqual(digest(raw), reviewed["raw_sha256"])
                actual = aime_submission(raw, reviewed["gold"])
                # The historical audit supplies explicit reviews for unfinished
                # final bodies, expressions and conflicting submissions.
                value = actual["score"]
                if value is None:
                    self.assertEqual(reviewed["decision"], "manually_reviewed_no_final_submission")
                    value = 0
                self.assertEqual(value, reviewed["recomputed_score"], (name, reviewed["question_id"], reviewed["response_index"]))
                total += value
            self.assertEqual(total, count)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ProtocolTests))
    write_json({"passed": result.wasSuccessful(), "tests": result.testsRun, "failures": len(result.failures),
                "errors": len(result.errors)}, args.run / "qualification/cpu_tests.json")
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
