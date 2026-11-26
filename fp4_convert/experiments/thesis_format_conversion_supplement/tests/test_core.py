"""CPU numerical contracts; full-model qualification remains separate."""
from types import SimpleNamespace

import torch

from HiFloat4.hif4_scale_threshold_optimization.src.weight_search import (
    FULL_OFFSETS, brute_force_group_search_reference, search_weight_groups)
from HiFloat4.hif4gptq.gptq.gptq_utils import GPTQ, WeightHiFxQuantizer


def test_full_search_matches_independent_enumeration_and_chunking():
    torch.manual_seed(42)
    weight = torch.randn(3, 128) * .1
    weight[0, :64] = 0
    small = search_weight_groups(weight, budget="full", device="cpu", group_chunk_size=1)
    large = search_weight_groups(weight, budget="full", device="cpu", group_chunk_size=6)
    assert torch.equal(small.reconstruction, large.reconstruction)
    for index, group in enumerate(weight.reshape(-1, 64)):
        expected, *_ = brute_force_group_search_reference(group, offsets=FULL_OFFSETS)
        assert torch.equal(small.reconstruction.reshape(-1, 64)[index], expected)


def test_variable_length_hessian_equals_direct_gram():
    torch.manual_seed(43)
    layer = SimpleNamespace(weight=torch.nn.Parameter(torch.randn(4, 128), requires_grad=False))
    block = GPTQ(layer)
    inputs = [torch.randn(n, 128) for n in (23, 31, 7)]
    for values in inputs:
        block.add_batch(values, None)
    expected = 2 / len(inputs) * sum(x.T @ x for x in inputs)
    torch.testing.assert_close(block.H, expected, rtol=2e-5, atol=2e-5)


def test_gptq_outputs_legal_payloads_under_its_selected_hierarchy():
    class RecordedQuantizer(WeightHiFxQuantizer):
        def __init__(self):
            super().__init__("hifx4")
            self.scales = []

        def find_params(self, weight):
            super().find_params(weight)
            self.scales.append(self._cached_scale.clone())

    torch.manual_seed(44)
    layer = SimpleNamespace(weight=torch.nn.Parameter(torch.randn(5, 128)*.1, requires_grad=False))
    block = GPTQ(layer)
    block.quantizer = RecordedQuantizer()
    block.add_batch(torch.randn(257, 128), None)
    block.fasterquant(groupsize=64, blocksize=128, percdamp=.01)
    scale = torch.cat(block.quantizer.scales, dim=-1)
    payload = layer.weight.abs()/scale
    assert torch.isfinite(payload).all()
    assert payload.max() <= 1.75
    torch.testing.assert_close(payload*4, (payload*4).round(), rtol=0, atol=2e-6)
    assert torch.equal(layer.weight, layer.weight.to(torch.bfloat16).float())


def test_expanded_pool_excludes_transitive_test_and_heldout_clusters():
    from fp4_convert.experiments.thesis_format_conversion_supplement.gptq_calibration import eligible_indices
    records = [
        {"id": "s1k_original_0", "source_index": 0, "role": "s1k", "split": "train"},
        {"id": "s1k_original_1", "source_index": 1, "role": "s1k_pool"},
        {"id": "s1k_original_2", "source_index": 2, "role": "s1k_pool"},
        {"id": "s1k_original_3", "source_index": 3, "role": "s1k", "split": "val"},
        {"id": "s1k_original_4", "source_index": 4, "role": "s1k_pool", "known_test_origin": True},
        {"id": "s1k_original_5", "source_index": 5, "role": "s1k_pool"},
        {"id": "intermediate", "role": "candidate"},
        {"id": "test", "role": "test"}, {"id": "dev", "role": "candidate"},
    ]
    edges = [{"left": a, "right": b} for a,b in (
        ("s1k_original_1", "intermediate"), ("intermediate", "test"), ("s1k_original_2", "dev"))]
    assert eligible_indices(records, edges, {"math": {"dev": [{"id": "dev"}]}}) == [0, 5]
