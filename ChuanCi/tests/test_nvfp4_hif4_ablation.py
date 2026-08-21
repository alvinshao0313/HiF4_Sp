import importlib
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

module = importlib.import_module("nvfp4_hif4_torch")


class PayloadFormatTests(unittest.TestCase):
    def test_e2m1_native_recomputes_s0_from_its_own_range(self) -> None:
        x = torch.ones(64)
        result = module.quantize_hif4(
            x,
            config=module.HiF4Config(
                scale_mode="continuous",
                payload_format="e2m1",
                hierarchy_format="e2m1",
            ),
        )

        torch.testing.assert_close(
            result.top_scale,
            torch.tensor([1.0 / 24.0], dtype=torch.float32),
            rtol=1e-6,
            atol=0,
        )
        self.assertTrue(torch.equal(result.e1_per_8, torch.ones_like(result.e1_per_8)))
        self.assertTrue(torch.equal(result.e1_per_4, torch.ones_like(result.e1_per_4)))
        self.assertTrue(torch.equal(result.payload_magnitude, torch.full_like(x, 6.0)))
        torch.testing.assert_close(result.values, x, rtol=1e-6, atol=1e-7)

    def test_e2m1_fixed_hierarchy_reuses_s1p2_s0(self) -> None:
        x = torch.ones(64)
        result = module.quantize_hif4(
            x,
            config=module.HiF4Config(
                scale_mode="continuous",
                payload_format="e2m1",
                hierarchy_format="s1p2",
            ),
        )

        torch.testing.assert_close(
            result.top_scale,
            torch.tensor([1.0 / 7.0], dtype=torch.float32),
            rtol=1e-6,
            atol=0,
        )
        self.assertTrue(torch.all(result.payload_magnitude <= 2.0).item())
        self.assertFalse(torch.equal(result.payload_magnitude, torch.full_like(x, 6.0)))

    def test_bf16_range_matched_keeps_s1p2_hierarchy_and_clips(self) -> None:
        x = torch.cat((torch.full((63,), 1.0), torch.tensor([8.0])))
        result = module.quantize_hif4(
            x,
            config=module.HiF4Config(
                scale_mode="continuous",
                payload_format="bf16_range_matched",
                hierarchy_format="s1p2",
            ),
        )

        self.assertTrue(torch.all(result.payload_magnitude >= 0).item())
        self.assertTrue(torch.all(result.payload_magnitude <= 1.75).item())
        expected_s0 = torch.tensor([8.0 / 7.0], dtype=torch.float32)
        torch.testing.assert_close(result.top_scale, expected_s0, rtol=1e-6, atol=0)

    def test_bf16_unclipped_can_exceed_s1p2_payload_range(self) -> None:
        x = torch.cat((torch.full((63,), 1.0), torch.tensor([8.0])))
        result = module.quantize_hif4(
            x,
            config=module.HiF4Config(
                scale_mode="continuous",
                payload_format="bf16_unclipped",
                hierarchy_format="s1p2",
                enable_exp8=False,
                enable_exp4=False,
            ),
        )

        self.assertGreater(result.payload_magnitude.max().item(), 1.75)
        torch.testing.assert_close(result.values, x, rtol=5e-3, atol=5e-3)

    def test_default_config_remains_standard_s1p2(self) -> None:
        x = torch.randn(4, 64, generator=torch.Generator().manual_seed(13))
        default = module.quantize_hif4(x)
        explicit = module.quantize_hif4(
            x,
            config=module.HiF4Config(
                payload_format="s1p2",
                hierarchy_format="s1p2",
                enable_exp8=True,
                enable_exp4=True,
            ),
        )
        torch.testing.assert_close(default.values, explicit.values, rtol=0, atol=0)
        torch.testing.assert_close(default.top_scale, explicit.top_scale, rtol=0, atol=0)
        torch.testing.assert_close(default.e1_per_8, explicit.e1_per_8, rtol=0, atol=0)
        torch.testing.assert_close(default.e1_per_4, explicit.e1_per_4, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
