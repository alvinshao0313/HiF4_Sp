import importlib
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

module = importlib.import_module("nvfp4_hif4_torch")


class GreenfieldModuleTests(unittest.TestCase):
    def test_imports_greenfield_module(self) -> None:
        self.assertEqual(module.__name__, "nvfp4_hif4_torch")
        self.assertTrue(hasattr(module, "HiF4Config"))
        self.assertTrue(hasattr(module, "quantize_hif4"))


class CodebookAndRoundingTests(unittest.TestCase):
    def test_e4m3fn_positive_codebook(self) -> None:
        values, codes = module.build_e4m3fn_codebook()
        self.assertEqual(values.dtype, torch.float32)
        self.assertEqual(codes.dtype, torch.int16)
        self.assertEqual(values.numel(), 127)
        self.assertEqual(values[0].item(), 0.0)
        self.assertEqual(values[1].item(), 2.0**-9)
        self.assertEqual(values[-1].item(), 448.0)
        self.assertTrue(torch.all(values[1:] > values[:-1]).item())
        torch.testing.assert_close(
            codes,
            torch.arange(127, dtype=torch.int16),
            rtol=0,
            atol=0,
        )

    def test_e6m2_unsigned_scale_codebook(self) -> None:
        values, codes = module.build_e6m2_codebook()
        self.assertEqual(values.numel(), 255)
        self.assertEqual(values[0].item(), 2.0**-48)
        self.assertEqual(values[-1].item(), 1.5 * 2.0**15)
        self.assertEqual(codes[0].item(), 0)
        self.assertEqual(codes[-1].item(), 254)
        self.assertTrue(torch.all(values[1:] > values[:-1]).item())

    def test_round_positive_to_codebook_uses_even_code_on_ties(self) -> None:
        values = torch.tensor([0.0, 1.0, 2.0, 4.0])
        codes = torch.tensor([0, 1, 2, 3], dtype=torch.int16)
        x = torch.tensor([-1.0, 0.5, 1.5, 3.0, 8.0])
        expected = torch.tensor([0.0, 0.0, 2.0, 2.0, 4.0])
        actual = module.round_positive_to_codebook(x, values, codes)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_e2m1_midpoints(self) -> None:
        x = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
        expected = torch.tensor([0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0])
        actual = module.quantize_e2m1_magnitude(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_round_bfloat16_matches_native_cast(self) -> None:
        values = torch.tensor(
            [0.0, -0.0, 1.0, 1.001, -3.1415926, 2.0**-120, 2.0**120],
            dtype=torch.float32,
        )
        expected = values.to(torch.bfloat16).to(torch.float32)
        actual = module.round_bfloat16(values)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class HiF4CoreTests(unittest.TestCase):
    def test_hif4_groups_last_dimension_without_crossing_rows(self) -> None:
        x = torch.randn(3, 128, generator=torch.Generator().manual_seed(1))
        result = module.quantize_hif4(
            x,
            config=module.HiF4Config(group_size=64, group_dim=-1),
        )
        self.assertEqual(result.values.shape, x.shape)
        self.assertEqual(result.top_scale.shape, (3, 2))
        self.assertEqual(result.e1_per_8.shape, (3, 2, 8))
        self.assertEqual(result.e1_per_4.shape, (3, 2, 16))

    def test_hif4_groups_dimension_zero(self) -> None:
        x = torch.randn(128, 3, generator=torch.Generator().manual_seed(2))
        result = module.quantize_hif4(
            x,
            config=module.HiF4Config(group_size=64, group_dim=0),
        )
        self.assertEqual(result.values.shape, x.shape)

    def test_hif4_rejects_illegal_inputs(self) -> None:
        valid = torch.randn(64)

        with self.assertRaises(TypeError):
            module.quantize_hif4(torch.ones(64, dtype=torch.int32))

        with self.assertRaises((ValueError, RuntimeError)):
            module.quantize_hif4(torch.tensor([float("nan"), 1.0] * 32))

        with self.assertRaises((ValueError, RuntimeError)):
            module.quantize_hif4(torch.tensor([float("inf"), 1.0] * 32))

        with self.assertRaises(ValueError):
            module.quantize_hif4(
                valid,
                config=module.HiF4Config(group_size=4),
            )

        with self.assertRaises(ValueError):
            module.quantize_hif4(
                valid,
                config=module.HiF4Config(group_size=12),
            )

        with self.assertRaises(ValueError):
            module.quantize_hif4(torch.randn(63))

        with self.assertRaises(ValueError):
            module.quantize_hif4(
                torch.randn(64, 64),
                config=module.HiF4Config(group_dim=3),
            )

        with self.assertRaises(ValueError):
            module.quantize_hif4(
                valid,
                config=module.HiF4Config(compute_dtype=torch.float64),
            )

        with self.assertRaises(ValueError):
            module.quantize_hif4(
                valid,
                config=module.HiF4Config(scale_mode="invalid_mode"),
            )

    def test_hif4_zero_and_payload_domain(self) -> None:
        zero = torch.zeros(2, 64)
        zero_result = module.quantize_hif4(zero)
        self.assertTrue(torch.equal(zero_result.values, zero))
        self.assertTrue(torch.isfinite(zero_result.local_scale).all().item())

        x = torch.linspace(-7, 7, 128).reshape(2, 64)
        result = module.quantize_hif4(x)
        payload = result.payload_magnitude
        self.assertTrue(torch.all(payload >= 0).item())
        self.assertTrue(torch.all(payload <= 1.75).item())
        self.assertTrue(torch.equal(payload * 4, torch.round(payload * 4)))
        self.assertTrue(torch.all((result.e1_per_8 == 0) | (result.e1_per_8 == 1)).item())
        self.assertTrue(torch.all((result.e1_per_4 == 0) | (result.e1_per_4 == 1)).item())

    def test_hif4_constant_one_group(self) -> None:
        x = torch.ones(64)
        result = module.quantize_hif4(
            x,
            config=module.HiF4Config(scale_mode="continuous"),
        )
        torch.testing.assert_close(
            result.top_scale,
            torch.tensor([1.0 / 7.0], dtype=torch.float32),
            rtol=1e-6,
            atol=0,
        )
        self.assertTrue(torch.equal(result.e1_per_8, torch.ones_like(result.e1_per_8)))
        self.assertTrue(torch.equal(result.e1_per_4, torch.ones_like(result.e1_per_4)))
        torch.testing.assert_close(
            result.values,
            torch.ones_like(x),
            rtol=1e-6,
            atol=1e-7,
        )

    def test_continuous_mode_is_scale_equivariant(self) -> None:
        x = torch.randn(4, 64, generator=torch.Generator().manual_seed(3))
        factor = torch.tensor(1.371)
        base = module.quantize_hif4(
            x,
            config=module.HiF4Config(scale_mode="continuous"),
        )
        scaled = module.quantize_hif4(
            x * factor,
            config=module.HiF4Config(scale_mode="continuous"),
        )
        torch.testing.assert_close(
            scaled.values,
            base.values * factor,
            rtol=2e-6,
            atol=1e-7,
        )


class NVFP4SimulationTests(unittest.TestCase):
    def test_simulate_nvfp4_returns_legal_fake_quant_values(self) -> None:
        x = torch.randn(3, 32, generator=torch.Generator().manual_seed(4))
        result = module.simulate_nvfp4(x)
        self.assertEqual(result.values.shape, x.shape)
        self.assertEqual(result.payload.shape, x.shape)
        self.assertEqual(result.block_scales.shape, (3, 2))
        self.assertEqual(result.global_scale.ndim, 0)
        legal = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
        for value in result.payload.abs().unique():
            self.assertTrue(torch.any(legal == value).item())

    def test_nvfp4_values_equal_scale_times_payload(self) -> None:
        x = torch.randn(2, 32, generator=torch.Generator().manual_seed(5))
        result = module.simulate_nvfp4(x)
        effective = (
            result.block_scales.unsqueeze(-1)
            * result.global_scale
        ).expand(2, 2, 16).reshape_as(x)
        expected = effective * result.payload
        torch.testing.assert_close(result.values, expected, rtol=0, atol=0)

    def test_nvfp4_zero_and_rejects_illegal(self) -> None:
        zero = torch.zeros(2, 32)
        result = module.simulate_nvfp4(zero)
        self.assertEqual(result.global_scale.item(), 1.0)
        self.assertTrue(torch.equal(result.block_scales, torch.zeros_like(result.block_scales)))
        self.assertTrue(torch.equal(result.payload, torch.zeros_like(result.payload)))
        self.assertTrue(torch.equal(result.values, torch.zeros_like(result.values)))

        with self.assertRaises(TypeError):
            module.simulate_nvfp4(torch.ones(32, dtype=torch.int32))

        with self.assertRaises((ValueError, RuntimeError)):
            module.simulate_nvfp4(torch.tensor([float("nan")] * 32))

        with self.assertRaises((ValueError, RuntimeError)):
            module.simulate_nvfp4(torch.tensor([float("inf")] * 32))

        with self.assertRaises(ValueError):
            module.simulate_nvfp4(torch.randn(31))

    def test_nvfp4_non_last_block_dim(self) -> None:
        # block_dim=0：沿第 0 维分 block；block_scales 保持 moved 布局 (3, 2)。
        # 注意：s_T 是整 tensor 共享，不能与“逐列重新 simulate”逐值对齐。
        x = torch.randn(32, 3, generator=torch.Generator().manual_seed(11))
        result = module.simulate_nvfp4(x, block_dim=0)
        self.assertEqual(result.values.shape, x.shape)
        self.assertEqual(result.payload.shape, x.shape)
        self.assertEqual(result.block_scales.shape, (3, 2))
        legal = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
        for value in result.payload.abs().unique():
            self.assertTrue(torch.any(legal == value).item())
        # 用同一 s_T 验证：每列在 block 维上独立成 block。
        expanded = (
            result.block_scales.unsqueeze(-1).expand(3, 2, 16).reshape(3, 32).transpose(0, 1)
            * result.global_scale
        )
        expected = expanded * result.payload
        torch.testing.assert_close(result.values, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
