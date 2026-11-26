"""Direction, K32 independence, FP32 boundary and real expert conversion checks."""
import json
import math
import unittest

import torch
from safetensors import safe_open

from ..qwen3_mxfp4.test_format import scalar_reference
from .convert import SOURCE
from .format import convert_weight, nvfp4_to_bf16, unpack_codes
from .transform import apply_r16x2, r16x2_matrix, r16x2_reference


class R16x2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(8)

    def test_matrix_direction_identity_and_common_mode(self):
        matrix = r16x2_matrix(dtype=torch.float64)
        torch.testing.assert_close(matrix @ matrix.T, torch.eye(32, dtype=torch.float64), rtol=0, atol=1e-14)
        self.assertTrue(torch.equal(matrix, matrix.T))
        self.assertTrue(torch.equal(apply_r16x2(torch.eye(32)), r16x2_matrix()))
        self.assertTrue(torch.equal(matrix.abs(), torch.full_like(matrix, 1/math.sqrt(32))))
        constant = torch.zeros(1, 192)
        constant[:, 32:64] = 1
        expected = torch.zeros_like(constant)
        expected[:, 32:48] = math.sqrt(2)
        torch.testing.assert_close(apply_r16x2(constant), expected, rtol=1e-7, atol=0)
        generator = torch.Generator().manual_seed(1234)
        for width in (32, 192, 2048):
            x, b = (torch.randn(n, width, generator=generator).bfloat16() for n in (3, 9))
            rx, rb = apply_r16x2(x), apply_r16x2(b)
            self.assertEqual(rx.dtype, torch.float32)
            torch.testing.assert_close(rx, r16x2_reference(x), rtol=2e-6, atol=2e-6)
            torch.testing.assert_close(rx.double() @ rb.double().T, x.double() @ b.double().T, rtol=2e-6, atol=2e-5)
        self.assertFalse(torch.equal(rx, rx.bfloat16().float()))
        with self.assertRaises(ValueError):
            apply_r16x2(torch.zeros(1, 31))

    def test_real_experts_each_projection(self):
        mapping = json.loads((SOURCE / "model.safetensors.index.json").read_text())["weight_map"]
        for layer, expert in ((0, 0), (47, 127)):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                prefix = f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
                tensors = []
                for suffix in ("weight", "weight_scale", "weight_scale_2"):
                    name = prefix + "." + suffix
                    with safe_open(str(SOURCE / mapping[name]), framework="pt") as reader:
                        tensor = reader.get_tensor(name)
                        tensors.append(tensor[:4].contiguous() if tensor.ndim == 2 else tensor)
                bf16 = nvfp4_to_bf16(*tensors)
                rotated = apply_r16x2(bf16)
                torch.testing.assert_close(rotated, r16x2_reference(bf16), rtol=2e-6, atol=2e-6)
                packed, scales = convert_weight(*tensors, "r16x2")
                codes, expected_scales = scalar_reference(rotated)
                self.assertTrue(torch.equal(unpack_codes(packed), codes))
                self.assertTrue(torch.equal(scales, expected_scales))


if __name__ == "__main__":
    unittest.main()
