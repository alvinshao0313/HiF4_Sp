"""Format edge cases and real Qwen weights, independent of CUDA execution."""
import json
import math
import unittest

import torch
from safetensors import safe_open

from .convert import SOURCE
from .format import (VALUES, VARIANT, activation_reference, convert_weight, decode_scales,
                     dequantize, nvfp4_to_bf16, quantize, unpack_codes)


def reference_exponent(amax):
    """Independent threshold search, without frexp, log2 or bit manipulation."""
    if amax == 0:
        return 0
    return next((e for e in range(-127, 128) if amax <= math.ldexp(7.25, e)), 127)


def threshold_values():
    centers = torch.tensor([math.ldexp(7.25, e) for e in range(-148, 125)], dtype=torch.float32)
    return torch.cat((torch.nextafter(centers, torch.zeros_like(centers)), centers,
                      torch.nextafter(centers, torch.full_like(centers, float("inf")))))


def scalar_reference(x):
    """Deliberately scalar nearest-distance reference, independent of bucketize."""
    rows = x.float().reshape(-1, 32).tolist()
    all_codes, all_scales = [], []
    for row in rows:
        amax = max(abs(value) for value in row)
        exponent = reference_exponent(amax)
        scale = 2. ** exponent
        codes = []
        for value in row:
            code = min(range(8), key=lambda k: (abs(abs(value / scale) - VALUES[k]), k % 2, k))
            codes.append(code | (8 if math.copysign(1, value) < 0 else 0))
        all_codes.extend(codes)
        all_scales.append(exponent + 127)
    return (torch.tensor(all_codes, dtype=torch.uint8).reshape_as(x),
            torch.tensor(all_scales, dtype=torch.uint8).reshape(*x.shape[:-1], -1))


class FormatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(8)

    def check_reference(self, x):
        p, s = quantize(x)
        codes, expected_s = scalar_reference(x)
        self.assertTrue(torch.equal(unpack_codes(p), codes))
        self.assertTrue(torch.equal(s, expected_s))

    def test_zero_and_signed_zero(self):
        x = torch.zeros(2, 64)
        x[1] = -x[1]
        p, s = quantize(x)
        self.assertTrue(torch.all(s == 127))
        self.assertTrue(torch.all(p[1] == 0x88))
        self.check_reference(x)
        self.assertTrue(torch.equal(torch.signbit(dequantize(p, s)), torch.signbit(x)))

    def test_midpoints_saturation_and_pack(self):
        # amax=7 gives scale=1; include ALL midpoint ties and both signs.
        values = [0., .25, .75, 1.25, 1.75, 2.5, 3.5, 5., 6., 7., -7., -5., -.75, -.25, -.0, 1.]
        x = torch.tensor(values + [-v for v in values]).reshape(1, 32)
        self.check_reference(x)
        p, s = quantize(x)
        self.assertEqual(p[0, 0].item(), 0)
        self.assertEqual(s.item(), 127)

    def test_power_boundaries_and_scale_extremes(self):
        for exponent in (-126, -80, -1, 0, 1, 30, 126):
            center = torch.tensor(2. ** exponent)
            for amax in (torch.nextafter(center, torch.tensor(0.)), center,
                         torch.nextafter(center, torch.tensor(float("inf")))):
                x = torch.zeros(1, 32)
                x[0, 0] = amax
                x[0, 1] = -amax
                self.check_reference(x)
        self.check_reference(torch.full((1, 32), 2. ** -140))
        s = torch.tensor([0, 1, 127, 253, 254], dtype=torch.uint8)
        self.assertTrue(torch.equal(decode_scales(s), torch.tensor([2. ** -127, 2. ** -126, 1., 2. ** 126, 2. ** 127])))
        with self.assertRaises(ValueError):
            decode_scales(torch.tensor([255], dtype=torch.uint8))

    def test_random_reference(self):
        generator = torch.Generator().manual_seed(1234)
        self.check_reference(torch.randn(16, 64, generator=generator))
        self.check_reference(torch.randn(16, 64, generator=generator).bfloat16())

    def test_nvfp4_scale_order(self):
        p = torch.arange(256, dtype=torch.uint8).reshape(8, 32)
        s = torch.arange(1, 33).reshape(8, 4).to(torch.float8_e4m3fn)
        scale_2 = torch.tensor(.00371829)
        got = nvfp4_to_bf16(p, s, scale_2)
        reference = []
        for row in range(8):
            values = []
            for col in range(64):
                code = (int(p[row, col // 2]) >> (4 * (col % 2))) & 15
                effective = torch.tensor(float(s[row, col // 16]), dtype=torch.float32) * scale_2
                values.append(float(effective * (VALUES[code & 7] * (-1 if code & 8 else 1))))
            reference.append(values)
        self.assertTrue(torch.equal(got, torch.tensor(reference).bfloat16()))

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
                b = nvfp4_to_bf16(*tensors)
                # Decode the actual source independently, including multiplication
                # order and the explicit BF16 boundary before the UOS quantizer.
                p0, s0, global_scale = tensors
                codes0 = torch.stack((p0 & 15, p0 >> 4), -1).flatten(-2)
                values0 = torch.tensor(VALUES)[(codes0 & 7).long()]
                values0 = torch.where(codes0 >= 8, -values0, values0)
                effective = s0.float() * global_scale.float()
                independent_b = (values0.reshape(*effective.shape, 16) * effective[..., None]).reshape_as(b).bfloat16()
                self.assertTrue(torch.equal(b, independent_b))
                p, s = convert_weight(*tensors)
                codes, expected_s = scalar_reference(independent_b)
                self.assertTrue(torch.equal(unpack_codes(p), codes))
                self.assertTrue(torch.equal(s, expected_s))

    def test_exact_ceiling_thresholds_and_neighbors(self):
        generator = torch.Generator().manual_seed(1234)
        random = torch.randint(0, 0x7f800000, (10000,), generator=generator, dtype=torch.int32).view(torch.float32)
        maxima = torch.cat((torch.tensor([0., 2.**-149, torch.finfo(torch.float32).max]), threshold_values(), random))
        x = torch.zeros(len(maxima), 32)
        x[:, 0] = maxima
        _, actual = quantize(x)
        expected = torch.tensor([reference_exponent(float(v)) + 127 for v in maxima], dtype=torch.uint8)
        self.assertTrue(torch.equal(actual.flatten(), expected))
        # Above 7.25 the new scale doubles; at exactly 7.25 it remains 1.
        boundary = torch.tensor(7.25)
        p, s = quantize(torch.stack([boundary.expand(32), torch.nextafter(boundary, torch.tensor(float("inf"))).expand(32)]))
        self.assertEqual(s.flatten().tolist(), [127, 128])
        print(f"Verified {len(maxima)} FP32 exponent values against independent threshold search", flush=True)

    def test_invalid(self):
        for x in (torch.ones(1, 31), torch.full((1, 32), float("nan")), torch.ones(32, dtype=torch.float64)):
            with self.assertRaises(ValueError):
                quantize(x)


if __name__ == "__main__":
    unittest.main()
