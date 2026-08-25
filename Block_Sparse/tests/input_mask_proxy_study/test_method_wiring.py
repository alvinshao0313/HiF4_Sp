from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Block_Sparse.input_mask_proxy_study.config import (  # noqa: E402
    ExperimentConfig,
    MethodId,
)
from Block_Sparse.input_mask_proxy_study import hif4_proxy as hif4_mod  # noqa: E402
from Block_Sparse.input_mask_proxy_study.methods import (  # noqa: E402
    METHOD_SPECS,
    prepare_operands,
    run_method,
)


def _cfg(**overrides) -> ExperimentConfig:
    base = dict(
        model_path="Qwen/Qwen3.5-4B",
        dataset_hf_id="simplescaling/s1K-1.1_tokenized",
        seed=31,
        num_samples=1,
        max_seq_len=64,
        max_activation_blocks=2,  # 1 sample * (64/32) blocks
        layer_index=15,
        projection="up_proj",
        activation_block_rows=32,
        k_block_size=64,
        output_block_cols=32,
        output_keep_ratios=(0.5,),
        input_keep_ratios=(0.5,),
        model_dtype="bfloat16",
        compute_dtype="float32",
        warmup=1,
        fast_repeats=3,
        exact_repeats=1,
    )
    base.update(overrides)
    return ExperimentConfig(**base)


def test_method_specs_table():
    assert set(METHOD_SPECS) == set(MethodId)
    expected = {
        MethodId.FULL_EXACT_REF: ("ref", "full", "exact"),
        MethodId.XPROXY_EXACT_OWN_OUTPUT: ("xp", "xp_fullw", "exact"),
        MethodId.XPROXY_ENERGY_OWN_OUTPUT: ("xp", "xp_fullw", "energy"),
        MethodId.FULL_ENERGY_REF_OUTPUT: ("ref", "full", "energy"),
        MethodId.XWPROXY_EXACT_REF_OUTPUT: ("ref", "xp_wp", "exact"),
        MethodId.XWPROXY_EXACT_OWN_OUTPUT: ("xpwp", "xp_wp", "exact"),
        MethodId.XPROXY_S0MEAN_ENERGY_OWN_OUTPUT: ("xp", "xp_fullw", "s0mean_energy"),
        MethodId.XPROXY_ENERGY_UNCONDITIONED_OWN_OUTPUT: (
            "xp",
            "xp_fullw",
            "energy_unconditioned",
        ),
    }
    for mid, (o, c, r) in expected.items():
        spec = METHOD_SPECS[mid]
        assert spec.output_source == o
        assert spec.contribution_source == c
        assert spec.recovery_kind == r


def test_prepare_operands_reuses_single_activation_proxy(monkeypatch):
    calls = {"n": 0}
    original = hif4_mod.build_hif4_ternary_proxy

    def wrapped(x):
        calls["n"] += 1
        return original(x)

    monkeypatch.setattr(
        "Block_Sparse.input_mask_proxy_study.methods.build_hif4_ternary_proxy", wrapped
    )
    torch.manual_seed(0)
    x = torch.randn(64, 128)
    w = torch.randn(64, 128)
    cfg = _cfg(max_seq_len=64, max_activation_blocks=2)
    prepared = prepare_operands(x, w, cfg)
    # one for X, one for W
    assert calls["n"] == 2
    xp_result = original(x)
    assert torch.equal(prepared.xp, xp_result.proxy)
    assert torch.equal(prepared.xp_s0, xp_result.s0)
    assert prepared.xp_s0.shape == (64, 2)
    assert torch.allclose(
        prepared.w_energy, prepared.w_blocks.square().mean(dim=(-1, -2))
    )
    assert prepared.all_output_weight_energy.shape == (prepared.w_energy.shape[1],)
    assert torch.allclose(
        prepared.all_output_weight_energy, prepared.w_energy.sum(dim=0)
    )


def test_output_mask_wiring_m2_m3_m7_m8_share_my_xp():
    torch.manual_seed(0)
    cfg = _cfg(
        output_keep_ratios=(0.5,),
        input_keep_ratios=(0.25, 0.5, 0.75),
        max_seq_len=32,
        max_activation_blocks=1,
    )
    # Synthetic Kb=4 -> keep counts 1/2/3; full-run Kb=40 -> 10/20/30.
    x = torch.randn(32, 256)
    w = torch.randn(64, 256)
    prepared = prepare_operands(x, w, cfg)
    r2 = run_method(MethodId.XPROXY_EXACT_OWN_OUTPUT, prepared, cfg)
    r3 = run_method(MethodId.XPROXY_ENERGY_OWN_OUTPUT, prepared, cfg)
    r7 = run_method(MethodId.XPROXY_S0MEAN_ENERGY_OWN_OUTPUT, prepared, cfg)
    r8 = run_method(MethodId.XPROXY_ENERGY_UNCONDITIONED_OWN_OUTPUT, prepared, cfg)
    assert torch.equal(r2.output_masks_by_ratio[0.5], r3.output_masks_by_ratio[0.5])
    assert torch.equal(r3.output_masks_by_ratio[0.5], r7.output_masks_by_ratio[0.5])
    assert torch.equal(r3.output_masks_by_ratio[0.5], r8.output_masks_by_ratio[0.5])
    mx25 = r7.input_masks_by_ratio[(0.5, 0.25)]
    mx50 = r7.input_masks_by_ratio[(0.5, 0.5)]
    mx75 = r7.input_masks_by_ratio[(0.5, 0.75)]
    assert torch.all(mx25.sum(-1) == 1)
    assert torch.all(mx50.sum(-1) == 2)
    assert torch.all(mx75.sum(-1) == 3)
    assert torch.all(mx25 <= mx50)
    assert torch.all(mx50 <= mx75)


def test_m8_mx_independent_of_my_but_compute_uses_my():
    torch.manual_seed(0)
    cfg = _cfg(
        output_keep_ratios=(0.25, 0.5, 0.75),
        input_keep_ratios=(0.5,),
        max_seq_len=32,
        max_activation_blocks=1,
    )
    x = torch.randn(32, 256)
    w = torch.randn(96, 256)  # Jb=3
    prepared = prepare_operands(x, w, cfg)
    a, jb = prepared.my_xp_by_ratio[0.5].shape
    my_025 = torch.zeros(a, jb, dtype=torch.bool)
    my_050 = torch.zeros(a, jb, dtype=torch.bool)
    my_075 = torch.zeros(a, jb, dtype=torch.bool)
    my_025[:, 0] = True
    my_050[:, 1] = True
    my_075[:, 0] = True
    my_075[:, 1] = True
    # keep counts: 0.25->1, 0.5->2, 0.75->2 for jb=3
    my_050[:, 0] = True  # keep 2
    object.__setattr__(
        prepared,
        "my_xp_by_ratio",
        {0.25: my_025, 0.5: my_050, 0.75: my_075},
    )
    r8 = run_method(MethodId.XPROXY_ENERGY_UNCONDITIONED_OWN_OUTPUT, prepared, cfg)
    mx_025 = r8.input_masks_by_ratio[(0.25, 0.5)]
    mx_050 = r8.input_masks_by_ratio[(0.50, 0.5)]
    mx_075 = r8.input_masks_by_ratio[(0.75, 0.5)]
    assert torch.equal(mx_025, mx_050)
    assert torch.equal(mx_050, mx_075)
    assert torch.equal(
        r8.compute_masks_by_ratio[(0.25, 0.5)],
        my_025[:, :, None] & mx_025[:, None, :],
    )
    assert torch.equal(
        r8.compute_masks_by_ratio[(0.50, 0.5)],
        my_050[:, :, None] & mx_050[:, None, :],
    )
    assert torch.equal(
        r8.compute_masks_by_ratio[(0.75, 0.5)],
        my_075[:, :, None] & mx_075[:, None, :],
    )


def test_full_run_m7_keep_counts():
    from Block_Sparse.input_mask_proxy_study.config import ratio_to_keep_count

    kb = 40
    assert ratio_to_keep_count(0.25, kb) == 10
    assert ratio_to_keep_count(0.50, kb) == 20
    assert ratio_to_keep_count(0.75, kb) == 30


def test_output_mask_wiring_distinct_sources():
    torch.manual_seed(0)
    t, n, k = 32, 64, 128
    x = torch.zeros(t, k)
    w = torch.zeros(n, k)
    x[:, :64] = 2.0
    w[:32, :64] = 2.0
    cfg = _cfg(output_keep_ratios=(0.5,), input_keep_ratios=(0.5,))
    prepared = prepare_operands(x, w, cfg)
    a, jb = prepared.my_ref_by_ratio[0.5].shape
    my_ref = torch.zeros(a, jb, dtype=torch.bool)
    my_xp = torch.zeros(a, jb, dtype=torch.bool)
    my_xpwp = torch.zeros(a, jb, dtype=torch.bool)
    my_ref[:, 0] = True
    my_xp[:, 1] = True
    my_xpwp[:, 0] = True
    object.__setattr__(prepared, "my_ref_by_ratio", {0.5: my_ref})
    object.__setattr__(prepared, "my_xp_by_ratio", {0.5: my_xp})
    object.__setattr__(prepared, "my_xpwp_by_ratio", {0.5: my_xpwp})

    r1 = run_method(MethodId.FULL_EXACT_REF, prepared, cfg)
    r2 = run_method(MethodId.XPROXY_EXACT_OWN_OUTPUT, prepared, cfg)
    r3 = run_method(MethodId.XPROXY_ENERGY_OWN_OUTPUT, prepared, cfg)
    r4 = run_method(MethodId.FULL_ENERGY_REF_OUTPUT, prepared, cfg)
    r5 = run_method(MethodId.XWPROXY_EXACT_REF_OUTPUT, prepared, cfg)
    r6 = run_method(MethodId.XWPROXY_EXACT_OWN_OUTPUT, prepared, cfg)
    r7 = run_method(MethodId.XPROXY_S0MEAN_ENERGY_OWN_OUTPUT, prepared, cfg)
    r8 = run_method(MethodId.XPROXY_ENERGY_UNCONDITIONED_OWN_OUTPUT, prepared, cfg)

    assert torch.equal(r1.output_masks_by_ratio[0.5], my_ref)
    assert torch.equal(r4.output_masks_by_ratio[0.5], my_ref)
    assert torch.equal(r5.output_masks_by_ratio[0.5], my_ref)
    assert torch.equal(r2.output_masks_by_ratio[0.5], my_xp)
    assert torch.equal(r3.output_masks_by_ratio[0.5], my_xp)
    assert torch.equal(r7.output_masks_by_ratio[0.5], my_xp)
    assert torch.equal(r8.output_masks_by_ratio[0.5], my_xp)
    assert torch.equal(r6.output_masks_by_ratio[0.5], my_xpwp)
    assert torch.equal(r2.output_masks_by_ratio[0.5], r3.output_masks_by_ratio[0.5])
    assert torch.equal(r3.output_masks_by_ratio[0.5], r7.output_masks_by_ratio[0.5])
    assert torch.equal(r3.output_masks_by_ratio[0.5], r8.output_masks_by_ratio[0.5])
    assert torch.equal(r1.output_masks_by_ratio[0.5], r5.output_masks_by_ratio[0.5])


def test_unknown_method_errors():
    cfg = _cfg()
    x = torch.randn(32, 64)
    w = torch.randn(64, 64)
    prepared = prepare_operands(x, w, cfg)
    with pytest.raises(ValueError):
        run_method("not_a_method", prepared, cfg)  # type: ignore[arg-type]
