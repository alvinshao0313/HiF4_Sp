"""Real numeric and serialized-format contracts, without engine substitutes."""
import json

import pytest
from safetensors import safe_open
from safetensors.torch import save_file
import torch

from .convert import convert, digest
from .format import dequantize, nvfp4_to_bf16, quantize, unpack


def test_all_e2m1_codes_and_rne_midpoints():
    packed = torch.tensor([0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe], dtype=torch.uint8)
    expected = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.])
    torch.testing.assert_close(unpack(packed), expected, rtol=0, atol=0)
    values = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5., 6.] * 4, dtype=torch.bfloat16)
    codes, scales = quantize(values)
    assert scales.item() == 127
    assert codes.tolist() == [0x20, 0x42, 0x64, 0x76] * 4
    torch.testing.assert_close(dequantize(codes, scales), torch.tensor([0., 1., 1., 2., 2., 4., 4., 6.] * 4).bfloat16(), rtol=0, atol=0)


@pytest.mark.parametrize('exponent', [-110, -10, 0, 10, 110])
def test_e8m0_block_scales_and_exact_representable_roundtrip(exponent):
    values = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.] * 4).mul(2. ** exponent).bfloat16()
    packed, scales = quantize(values)
    assert scales.item() == exponent + 127
    torch.testing.assert_close(dequantize(packed, scales), values, rtol=0, atol=0)
    zero, zero_scales = quantize(torch.zeros(2, 64, dtype=torch.bfloat16))
    assert torch.all(zero == 0) and torch.all(zero_scales == 127)


def test_nvfp4_decode_uses_global_multiplier_then_rounds_to_bf16():
    packed = torch.full((2, 32), 0x76, dtype=torch.uint8)
    scales = torch.full((2, 4), 1.5, dtype=torch.float8_e4m3fn)
    result = nvfp4_to_bf16(packed, scales, torch.tensor(7.3))
    expected = (torch.tensor([4., 6.] * 32).expand(2, -1) * (torch.tensor(1.5) * torch.tensor(7.3).reciprocal())).bfloat16()
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match='BF16'):
        quantize(expected.float())


def test_convert_actual_safetensors_and_preserve_unmodified_tensors(tmp_path):
    source, output = tmp_path / 'source', tmp_path / 'output'
    source.mkdir()
    config = {'text_config': {'moe_layer_freq': [1], 'num_local_experts': 1},
        'quantization_config': {'moe_quant_algo': 'NVFP4', 'group_size': 16,
                               'moe_quant_format': 'nvfp4-pack-quantized'}}
    (source / 'config.json').write_text(json.dumps(config))
    tensors = {'language_model.model.norm.weight': torch.arange(8).bfloat16()}
    for part in ['w1', 'w2', 'w3']:
        stem = 'language_model.model.layers.0.block_sparse_moe.experts.0.' + part
        tensors[stem + '.weight_packed'] = torch.arange(128, dtype=torch.uint8).reshape(4, 32)
        tensors[stem + '.weight_scale'] = torch.ones(4, 4, dtype=torch.float8_e4m3fn)
        tensors[stem + '.weight_global_scale'] = torch.tensor(2.)
        tensors[stem + '.input_global_scale'] = torch.tensor(1.)
    filename = 'model-00001-of-00001.safetensors'
    save_file(tensors, source / filename)
    (source / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {k:filename for k in tensors}}))
    before = digest(source / filename)
    convert(source, output, threads=2)
    assert digest(source / filename) == before
    status = json.loads((output / 'conversion_status.json').read_text())
    assert status['status'] == 'complete' and status['completed_matrices'] == 3
    index = json.loads((output / 'model.safetensors.index.json').read_text())['weight_map']
    assert not any('global_scale' in name for name in index)
    with safe_open(output / filename, framework='pt') as handle:
        torch.testing.assert_close(handle.get_tensor('language_model.model.norm.weight'), tensors['language_model.model.norm.weight'], rtol=0, atol=0)
        for part in ['w1', 'w2', 'w3']:
            stem = 'language_model.model.layers.0.block_sparse_moe.experts.0.' + part
            assert handle.get_tensor(stem + '.weight_scale').dtype == torch.uint8
            assert handle.get_tensor(stem + '.weight_scale').shape == (4, 2)
