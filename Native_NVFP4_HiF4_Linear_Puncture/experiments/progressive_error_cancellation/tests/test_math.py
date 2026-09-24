import torch
from types import SimpleNamespace

from Native_NVFP4_HiF4_Linear_Puncture.experiments.progressive_error_cancellation.directions import (
    build_direction, compact_direction, load_direction,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.progressive_error_cancellation.jvp import toy_jvp
from Native_NVFP4_HiF4_Linear_Puncture.experiments.progressive_error_cancellation.losses import (
    absolute_mse, batch_cosine, decompose_errors,
)


def test_absolute_mse_and_direction_sign():
    value = torch.tensor([[[1., 2.], [3., 4.]]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    assert absolute_mse(value, torch.zeros_like(value), valid) == 7.5
    loss, active, *_ = batch_cosine(value, -value, valid)
    assert active and abs(float(loss)) < 1e-6


def test_zero_direction_is_skipped():
    value = torch.ones(1, 2, 3)
    valid = torch.ones(1, 2, dtype=torch.bool)
    loss, active, norm, direction_norm = batch_cosine(value, torch.zeros_like(value), valid)
    assert not active and loss.item() == 0 and direction_norm == 0 and norm > 0


def test_error_decomposition_is_exact():
    p = torch.randn(1, 3, 4)
    q = torch.randn(1, 3, 4)
    valid = torch.ones(1, 3, dtype=torch.bool)
    values = decompose_errors(p + q, p, torch.zeros_like(p), valid)
    assert torch.allclose(values["cumulative"], values["propagated"] + values["local"])


def test_toy_jvp():
    weight = torch.tensor([[2., 1.], [-1., 3.]])
    tangent = torch.tensor([[4., -2.]])
    x = torch.tensor([[1., 2.]])
    assert torch.allclose(toy_jvp(weight, x, tangent), tangent @ weight.T)


def test_direction_cache_hash_and_compaction(tmp_path):
    samples = [SimpleNamespace(sample_id=f"s{i}") for i in range(4)]
    native = {s.sample_id: torch.zeros(3, 2) for s in samples}
    student = {s.sample_id: torch.full((3, 2), float(i + 1)) for i, s in enumerate(samples)}
    cache = tmp_path / "direction"
    values, manifest = build_direction(
        method="direct", source_layer=1, runtime=None, native_hidden=native,
        student_hidden=student, samples=samples, snapshot=tmp_path,
        device=torch.device("cpu"), output_dir=cache,
    )
    loaded, loaded_manifest = load_direction(cache, expected_manifest=manifest)
    assert loaded_manifest == manifest
    assert all(torch.equal(loaded[s.sample_id], values[s.sample_id]) for s in samples)
    compact_direction(cache, values, manifest)
    assert not cache.exists()
    assert (tmp_path / "direction_summary.json").is_file()
    assert (tmp_path / "direction_probe.pt").is_file()


def test_shuffled_direction_is_fixed_within_batches(tmp_path):
    samples = [SimpleNamespace(sample_id=f"s{i}") for i in range(4)]
    native = {s.sample_id: torch.zeros(1, 1) for s in samples}
    student = {s.sample_id: torch.tensor([[float(i + 1)]]) for i, s in enumerate(samples)}
    batches = {"train": [samples[:2], samples[2:]]}
    values, manifest = build_direction(
        method="shuffled", source_layer=1, runtime=None, native_hidden=native,
        student_hidden=student, samples=samples, snapshot=tmp_path,
        device=torch.device("cpu"), batch_groups=batches,
    )
    assert manifest["shuffle_scope"] == "fixed_batch_permutation"
    for batch in batches["train"]:
        assert {values[s.sample_id].item() for s in batch} == {student[s.sample_id].item() for s in batch}
