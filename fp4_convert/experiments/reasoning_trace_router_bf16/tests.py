"""Small loss/precision contracts; the real-layer contracts run on GPU."""
import argparse
from pathlib import Path
import unittest

import torch

from ..non_equivalent_reconstruction.losses import masked_router_loss
from .artifacts import verify
from .student import RouterStudent, load_layer
from .training import optimizer_contract, optimizer_for


class Contracts(unittest.TestCase):
    def test_top_mass_is_teacher_to_student_nine_bucket_kl(self):
        torch.manual_seed(42)
        s = torch.randn(2, 3, 128, requires_grad=True)
        t = torch.randn_like(s)
        mask = torch.tensor([[True, True, False], [True, False, False]])
        q, p = t.softmax(-1), s.softmax(-1)
        ids = t.topk(8, -1).indices
        outside = torch.ones_like(t, dtype=torch.bool).scatter_(-1, ids, False)
        q = torch.cat((q.gather(-1, ids), q.masked_fill(~outside, 0).sum(-1, keepdim=True)), -1)
        p = torch.cat((p.gather(-1, ids), p.masked_fill(~outside, 0).sum(-1, keepdim=True)), -1)
        expected = (q * (q.log() - p.log())).sum(-1)[mask].mean()
        actual = masked_router_loss(s, t, mask, kind="top_mass", top_k=8, temperature=1.)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        actual.backward()
        self.assertEqual(float(s.grad[~mask].abs().sum()), 0.)
        self.assertGreater(float(s.grad[mask].abs().sum()), 0.)

    def test_actual_B_gate_has_only_bf16_optimizer_state(self):
        student = RouterStudent(load_layer(self.model, 0, "cpu"))
        parameter = student.router_weight
        initial = parameter.detach().clone()
        optimizer = optimizer_for(student, 1e-4)
        torch.manual_seed(42)
        x = torch.randn(1, 16, parameter.shape[1], dtype=torch.bfloat16)
        teacher = torch.randn(1, 16, parameter.shape[0], dtype=torch.bfloat16)
        loss = masked_router_loss(student.router(x), teacher, torch.ones(1, 16, dtype=torch.bool), kind="top_mass")
        loss.backward()
        optimizer.step()
        self.assertFalse(optimizer_contract(student, optimizer)["fp32_master"])
        self.assertFalse(torch.equal(initial, parameter))
        self.assertEqual([n for n, p in student.named_parameters() if p.requires_grad], ["router_weight"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    root, manifest = verify(parser.parse_args().run)
    Contracts.model = manifest["initialization"]
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    raise SystemExit(0 if result.wasSuccessful() else 1)
