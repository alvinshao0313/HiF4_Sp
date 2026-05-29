import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import NVFP4.torch_fake as torch_fake  # noqa: E402
from NVFP4.torch_fake import (  # noqa: E402
    NvFp4FakeLinear,
    cast_to_fp4_e2m1,
    cast_to_fp8_e4m3fn,
    fake_quant_nvfp4_activation,
)


CUDA_AVAILABLE = torch.cuda.is_available()


def ref_nvfp4_activation_qdq(
    x: torch.Tensor,
    input_global_scale: torch.Tensor,
    group_size: int = 16,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if output_dtype is None:
        output_dtype = x.dtype
    original_shape = x.shape
    hidden_size = original_shape[-1]
    x_2d = x.reshape(-1, hidden_size).to(torch.float32)
    grouped = x_2d.reshape(x_2d.shape[0], hidden_size // group_size, group_size)
    global_scale = input_global_scale.reshape(()).to(device=x.device, dtype=torch.float32)
    if global_scale.item() == 0.0:
        return torch.zeros_like(x, dtype=output_dtype)

    vec_max = grouped.abs().amax(dim=-1, keepdim=True)
    block_scale = global_scale * (vec_max / 6.0)
    block_scale = torch.clamp(block_scale, max=448, min=-448)
    block_scale = block_scale.to(torch.float8_e4m3fn).to(torch.float32)
    output_scale = 1.0 / (block_scale / global_scale)

    scaled = grouped * output_scale
    clipped = torch.clamp(scaled, -6.0, 6.0)
    fp4 = cast_to_fp4_e2m1(clipped)
    return (fp4 * (block_scale / global_scale)).reshape(original_shape).to(output_dtype)


class Nvfp4TorchFakeTest(unittest.TestCase):
    def setUp(self):
        torch_fake.USE_TRITON_NVFP4_KERNEL = False

    def tearDown(self):
        torch_fake.USE_TRITON_NVFP4_KERNEL = False

    def test_cast_to_fp8_e4m3fn_matches_torch_float8_cast(self):
        values = torch.tensor(
            [
                -448.0,
                -432.0,
                -400.0,
                -1.6875,
                -1.0625,
                -0.003,
                -0.001,
                0.0,
                0.001,
                0.001953125,
                0.003,
                0.015,
                0.016,
                0.1,
                1.0625,
                1.1875,
                1.3125,
                1.4375,
                1.6875,
                1.8125,
                2.125,
                2.375,
                240.0,
                248.0,
                272.0,
                304.0,
                400.0,
                432.0,
                448.0,
            ],
            dtype=torch.float32,
        )
        expected = values.to(torch.float8_e4m3fn).to(torch.float32)
        actual = cast_to_fp8_e4m3fn(values)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_cast_to_fp4_e2m1_thresholds(self):
        values = torch.tensor(
            [
                -6.1,
                -4.2,
                -3.2,
                -2.1,
                -1.6,
                -1.0,
                -0.4,
                -0.1,
                0.0,
                0.25,
                0.26,
                0.74,
                0.75,
                1.25,
                1.26,
                1.74,
                1.75,
                2.5,
                2.51,
                3.49,
                3.5,
                5.0,
                5.01,
            ],
            dtype=torch.float32,
        )
        expected = torch.tensor(
            [
                -6.0,
                -4.0,
                -3.0,
                -2.0,
                -1.5,
                -1.0,
                -0.5,
                -0.0,
                0.0,
                0.0,
                0.5,
                0.5,
                1.0,
                1.0,
                1.5,
                1.5,
                2.0,
                2.0,
                3.0,
                3.0,
                4.0,
                4.0,
                6.0,
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(cast_to_fp4_e2m1(values), expected)

    def test_triton_kernel_is_disabled_by_default(self):
        self.assertFalse(torch_fake.USE_TRITON_NVFP4_KERNEL)

    def test_fake_quant_nvfp4_activation_matches_reference_random(self):
        torch.manual_seed(123)
        x = torch.randn(4, 32, dtype=torch.float32) * 8
        input_global_scale = torch.tensor([1872.0], dtype=torch.float32)

        actual = fake_quant_nvfp4_activation(
            x,
            input_global_scale,
            group_size=16,
            output_dtype=torch.float32,
        )
        expected = ref_nvfp4_activation_qdq(
            x,
            input_global_scale,
            group_size=16,
            output_dtype=torch.float32,
        )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_fake_quant_nvfp4_activation_matches_reference_boundaries(self):
        values = torch.tensor(
            [
                -8.0,
                -6.0,
                -5.01,
                -5.0,
                -3.5,
                -2.5,
                -1.75,
                -1.25,
                -0.75,
                -0.25,
                0.0,
                0.25,
                0.75,
                1.25,
                1.75,
                2.5,
                3.5,
                5.0,
                5.01,
                6.0,
                8.0,
                12.0,
                -12.0,
                448.0,
                -448.0,
                512.0,
                -512.0,
                1.0,
                -1.0,
                2.0,
                -2.0,
                4.0,
            ],
            dtype=torch.float32,
        ).reshape(2, 16)
        input_global_scale = torch.tensor([1024.0], dtype=torch.float32)

        actual = fake_quant_nvfp4_activation(
            values,
            input_global_scale,
            group_size=16,
            output_dtype=torch.float32,
        )
        expected = ref_nvfp4_activation_qdq(
            values,
            input_global_scale,
            group_size=16,
            output_dtype=torch.float32,
        )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_fake_quant_nvfp4_activation_clamps_scale_to_448(self):
        x = torch.full((2, 16), 6.0, dtype=torch.float32)
        input_global_scale = torch.tensor([1872.0], dtype=torch.float32)

        actual = fake_quant_nvfp4_activation(
            x,
            input_global_scale,
            group_size=16,
            output_dtype=torch.float32,
        )
        expected = ref_nvfp4_activation_qdq(
            x,
            input_global_scale,
            group_size=16,
            output_dtype=torch.float32,
        )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_zero_global_scale_returns_zero(self):
        x = torch.randn(2, 16, dtype=torch.float32)
        actual = fake_quant_nvfp4_activation(
            x,
            torch.tensor([0.0], dtype=torch.float32),
            output_dtype=torch.float32,
        )
        torch.testing.assert_close(actual, torch.zeros_like(x), rtol=0.0, atol=0.0)

    def test_zero_block_stays_zero(self):
        x = torch.zeros(2, 16, dtype=torch.float32)
        actual = fake_quant_nvfp4_activation(
            x,
            torch.tensor([1872.0], dtype=torch.float32),
            output_dtype=torch.float32,
        )
        self.assertFalse(torch.isnan(actual).any())
        torch.testing.assert_close(actual, x, rtol=0.0, atol=0.0)

    def test_fake_linear_matches_explicit_functional_linear(self):
        torch.manual_seed(123)
        x = torch.randn(2, 3, 16, dtype=torch.bfloat16)
        weight = torch.randn(5, 16, dtype=torch.bfloat16)
        bias = torch.randn(5, dtype=torch.bfloat16)
        input_global_scale = torch.tensor([1024.0], dtype=torch.float32)

        layer = NvFp4FakeLinear(weight, bias, input_global_scale)
        actual = layer(x)

        x_qdq = fake_quant_nvfp4_activation(
            x,
            input_global_scale,
            output_dtype=x.dtype,
        )
        expected = F.linear(x_qdq, weight, bias)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_kernel_path_requires_cuda(self):
        torch_fake.USE_TRITON_NVFP4_KERNEL = True
        x = torch.randn(2, 16, dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "CUDA"):
            fake_quant_nvfp4_activation(x, torch.tensor([1.0]))

    @unittest.skipUnless(CUDA_AVAILABLE, "CUDA is required for Triton kernel test")
    def test_kernel_path_matches_reference_random(self):
        torch_fake.USE_TRITON_NVFP4_KERNEL = True
        torch.manual_seed(123)
        x = torch.randn(4, 32, device="cuda", dtype=torch.float32) * 8
        input_global_scale = torch.tensor([1872.0], device="cuda", dtype=torch.float32)

        actual = fake_quant_nvfp4_activation(
            x,
            input_global_scale,
            group_size=16,
            output_dtype=torch.float32,
        )
        expected = ref_nvfp4_activation_qdq(
            x,
            input_global_scale,
            group_size=16,
            output_dtype=torch.float32,
        )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    @unittest.skipUnless(CUDA_AVAILABLE, "CUDA is required for Triton kernel test")
    def test_kernel_path_zero_global_scale_returns_zero(self):
        torch_fake.USE_TRITON_NVFP4_KERNEL = True
        x = torch.randn(2, 16, device="cuda", dtype=torch.float32)
        actual = fake_quant_nvfp4_activation(
            x,
            torch.tensor([0.0], device="cuda", dtype=torch.float32),
            output_dtype=torch.float32,
        )
        torch.testing.assert_close(actual, torch.zeros_like(x), rtol=0.0, atol=0.0)

    def test_last_dimension_must_be_divisible_by_group_size(self):
        x = torch.randn(2, 15, dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "divisible"):
            fake_quant_nvfp4_activation(x, torch.tensor([1.0]))


if __name__ == "__main__":
    unittest.main()
