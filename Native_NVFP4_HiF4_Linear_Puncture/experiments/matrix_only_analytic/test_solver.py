"""CPU algebra/quantization contracts; separate from the real-model diagnostic."""
import tempfile
from pathlib import Path
import unittest

import torch

from ..non_equivalent_reconstruction.artifact import atomic_save, load
from ..non_equivalent_reconstruction.transforms import block_right
from ...src.formats import qdq_hif4_direct
from .solver import (ProjectionData, collect_statistics, delta_loss, fit_projection,
                     quantized_weight, quadratic_mse, solve_increment)


class SolverTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def test_quantization_axis_and_nonsymmetric_h(self):
        weight = torch.randint(-16, 17, (96, 128)).float() / 32
        h = torch.randint(-8, 9, (2, 64, 64)).float() / 16
        folded = block_right(weight, h.transpose(-1, -2))
        reference = torch.cat([weight[:, i * 64:(i + 1) * 64] @ h[i].T for i in range(2)], dim=1)
        self.assertTrue(torch.equal(folded, reference))
        quantized = torch.cat([qdq_hif4_direct(reference[:, i * 64:(i + 1) * 64], output_dtype=torch.bfloat16)
                               for i in range(2)], dim=1)
        self.assertTrue(torch.equal(quantized_weight(weight, h), quantized))
        self.assertFalse(torch.equal(folded, block_right(weight, h)))

    def test_cached_loss_and_cross_block_residual(self):
        a = qdq_hif4_direct(torch.randn(137, 128), output_dtype=torch.bfloat16)
        y = torch.randn(137, 96).to(torch.bfloat16)
        stats = collect_statistics(ProjectionData(a, y), chunk_tokens=31)
        v = qdq_hif4_direct(torch.randn(96, 128), output_dtype=torch.bfloat16).T.double()
        direct = (a.double() @ v - y.double()).square().mean()
        self.assertAlmostEqual(quadratic_mse(stats, v), float(direct), places=11)
        residual = stats.gram @ v - stats.cross
        candidate = qdq_hif4_direct(torch.randn(96, 64), output_dtype=torch.bfloat16).T.double()
        delta = candidate - v[:64]
        cached_change = delta_loss(stats.gram[:64, :64], residual[:64], delta)
        new_v = v.clone()
        new_v[:64] = candidate
        actual_change = ((a.double() @ new_v - y.double()).square().sum()
                         - (a.double() @ v - y.double()).square().sum()) / a.shape[0]
        torch.testing.assert_close(cached_change, actual_change, rtol=1e-11, atol=1e-10)
        updated = residual + stats.gram[:, :64] @ delta
        torch.testing.assert_close(updated, stats.gram @ new_v - stats.cross, rtol=1e-11, atol=1e-11)
        self.assertGreater(float((updated[64:] - residual[64:]).norm()), 0.)

    def test_regularized_sylvester_with_rank_deficient_grams(self):
        aa, bb = torch.randn(64, 17).double(), torch.randn(64, 23).double()
        g, k = aa @ aa.T, bb @ bb.T
        rhs = torch.randn(64, 64).double()
        damping = torch.tensor(.01, dtype=torch.float64) * g.diag().mean() * k.diag().mean()
        delta = solve_increment(g, k, rhs, damping)
        error = (g @ delta @ k + damping * delta - rhs).norm() / rhs.norm()
        self.assertLess(float(error), 1e-10)

    def test_real_qdq_fit_freeze_selection_and_reload(self):
        weight = torch.randn(96, 128)
        reference = weight.clone()
        x = torch.randn(80, 128)
        from ..e2e_diag_reconstruction.core.moe_transforms import apply_r64_no_cross_head
        a = qdq_hif4_direct(apply_r64_no_cross_head(x), output_dtype=torch.bfloat16)
        y = (x @ weight.T).to(torch.bfloat16)
        train = ProjectionData(a[:48], y[:48])
        validation = ProjectionData(a[48:], y[48:])
        for method in ("analytic", "adam"):
            result = fit_projection(weight, train, validation, method)
            self.assertTrue(torch.equal(weight, reference))
            self.assertIsNone(weight.grad)
            self.assertFalse(result.selected_h.requires_grad)
            curve = result.report["curve"]
            best = curve[result.report["selected_iteration"]]["validation"]["mse"]
            self.assertLessEqual(best, curve[0]["validation"]["mse"])
            if method == "analytic":
                self.assertGreater(sum(entry["accepted_blocks"] for entry in curve[1:]), 0)
                scores = [entry["cached_real_product_mse"] for entry in curve[1:]]
                self.assertLessEqual(scores[1], scores[0])
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "h.pt"
                atomic_save(result.selected_h, path)
                self.assertTrue(torch.equal(load(path), result.selected_h))


if __name__ == "__main__":
    unittest.main()
