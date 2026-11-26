"""Independent scalar scale/code decisions and real first/last-layer weights."""
import json
import math
import unittest
import torch
from safetensors import safe_open

from .convert import SOURCE
from .format import VARIANTS, options, quantize, convert_weight, unpack_codes, nvfp4_to_bf16
from ..qwen3_mxfp4.format import VALUES
from ..qwen3_mxfp4_r16x2.transform import apply_r16x2, r16x2_matrix


def reference(values, outer_enabled, uos):
    # Independent float64 logarithm/nearest-distance scalar reference;
    # FP32 outer normalization is explicit, matching the approved contract.
    z = values.clone().float()
    outer = torch.ones(z.shape[0], (z.shape[1]+511)//512) if outer_enabled else None
    if outer_enabled:
        for row in range(z.shape[0]):
            for start in range(0, z.shape[1], 512):
                part = z[row, start:start+512]
                maximum = max(abs(v) for v in part.tolist())
                t = torch.tensor(maximum / 6., dtype=torch.float32) if maximum else torch.tensor(1.)
                outer[row, start//512] = t
                z[row, start:start+512] = part / t
    codes = torch.empty_like(z, dtype=torch.uint8)
    scales = torch.empty((z.shape[0], z.shape[1]//32), dtype=torch.uint8)
    for row in range(z.shape[0]):
        for start in range(0, z.shape[1], 32):
            part = z[row, start:start+32].tolist()
            maximum = max(map(abs, part))
            e = 0 if maximum == 0 else (math.ceil(math.log2(maximum/7.25)) if uos else math.floor(math.log2(maximum))-2)
            e = max(-127, min(127, e))
            scales[row, start//32] = e+127
            for j, value in enumerate(part):
                code = min(range(8), key=lambda k: (abs(abs(value/(2.**e))-VALUES[k]), k%2, k))
                codes[row, start+j] = code | (8 if math.copysign(1., value)<0 else 0)
    return codes, scales, outer


class FormatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(8)

    def check(self, x, variant):
        rotation, outer, uos = options(variant)
        values = apply_r16x2(x) if rotation else x.float()
        packed, s, t = quantize(x, variant)
        c_ref, s_ref, t_ref = reference(values, outer, uos)
        self.assertTrue(torch.equal(unpack_codes(packed), c_ref))
        self.assertTrue(torch.equal(s, s_ref))
        if outer:
            self.assertTrue(torch.equal(t, t_ref))
        else:
            self.assertIsNone(t)

    def test_rotation_and_product(self):
        torch.manual_seed(1234)
        r = r16x2_matrix().double()
        torch.testing.assert_close(r @ r.T, torch.eye(32, dtype=torch.float64), atol=4e-8, rtol=0)
        x, b = torch.randn(3, 2048).bfloat16(), torch.randn(4, 2048).bfloat16()
        for value in (x, b):
            reference = (value.double().reshape(-1,32) @ r).reshape_as(value)
            torch.testing.assert_close(apply_r16x2(value).double(), reference, atol=2e-6, rtol=1e-6)
        torch.testing.assert_close(apply_r16x2(x).double() @ apply_r16x2(b).double().T,
                                   x.double() @ b.double().T, atol=2e-5, rtol=2e-6)

    def test_combination_codes_and_tail(self):
        torch.manual_seed(456)
        for variant in VARIANTS:
            for k in (768, 2048):
                x = torch.randn(3, k).bfloat16()
                x[0].zero_()
                self.check(x, variant)

    def test_thresholds_and_ties(self):
        # Rotation is checked separately; ensure inner rules remain exact.
        from ..qwen3_mxfp4_uos.test_format import scalar_reference
        from ..qwen3_mxfp4_uos.format import quantize as uos_quantize
        for e in (-120, -10, 0, 10, 120):
            boundary = torch.tensor(7.25*2.**e)
            for a in (torch.nextafter(boundary, torch.tensor(0.)), boundary,
                      torch.nextafter(boundary, torch.tensor(float('inf')))):
                x = torch.zeros(1,32); x[0,0] = a; x[0,1] = -a
                p, s = uos_quantize(x)
                expected_c, expected_s = scalar_reference(x)
                self.assertTrue(torch.equal(unpack_codes(p), expected_c))
                self.assertTrue(torch.equal(s, expected_s))
        for variant in VARIANTS:
            x = torch.tensor([0.,-.0,.25,.75,1.25,1.75,2.5,3.5,5.,6.,7.25,-6.,-.75,-.25,1.,-1.]*32).reshape(1,512)
            self.check(x, variant)

    def test_real_first_last_weights(self):
        mapping = json.loads((SOURCE/'model.safetensors.index.json').read_text())['weight_map']
        for layer in (0,47):
            for part in ('gate_proj','up_proj','down_proj'):
                prefix=f'model.layers.{layer}.mlp.experts.0.{part}'
                data={}
                for suffix in ('weight','weight_scale','weight_scale_2'):
                    name=prefix+'.'+suffix
                    with safe_open(str(SOURCE/mapping[name]),framework='pt') as f:
                        value=f.get_tensor(name)
                        data[suffix]=value[:2] if value.ndim==2 else value
                b=nvfp4_to_bf16(data['weight'],data['weight_scale'],data['weight_scale_2'])
                for variant in VARIANTS:
                    self.check(b,variant)
                    actual=convert_weight(data['weight'],data['weight_scale'],data['weight_scale_2'],variant)
                    expected=quantize(b,variant)
                    self.assertTrue(all(a is None and e is None or torch.equal(a,e) for a,e in zip(actual,expected)))


if __name__ == '__main__':
    unittest.main()
