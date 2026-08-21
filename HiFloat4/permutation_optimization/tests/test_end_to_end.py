"""End-to-end smoke tests for the permutation package."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import pytest

from permutation_optimization.config import SearchConfig
from permutation_optimization.hierarchical_greedy import optimize_layer_permutation
from permutation_optimization.model_permutation import (
    apply_mlp_permutation_,
    discover_swiglu_mlps,
    get_mlp_modules,
)


class _TinyLM64(nn.Module):
    """Tiny model with one SwiGLU MLP; d_model/d_ff multiples of 64."""

    def __init__(self, vocab: int = 32, d_model: int = 64, d_ff: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.layers = nn.ModuleList([nn.Module()])
        self.layers[0].mlp = nn.Module()
        self.layers[0].mlp.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.layers[0].mlp.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.layers[0].mlp.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        x = self.embed(input_ids)
        m = self.layers[0].mlp
        return x + m.down_proj(torch.nn.functional.silu(m.gate_proj(x)) * m.up_proj(x))


def _tiny_pipeline_run(tmp_path: Path, apply_candidate_name: str = "selected"):
    from permutation_optimization.pipeline import reorder_model_mlps

    torch.manual_seed(0)
    model = _TinyLM64()
    g = torch.Generator().manual_seed(0)
    batches = [
        {
            "input_ids": torch.randint(0, 32, (2, 16), generator=g),
            "attention_mask": torch.ones(2, 16, dtype=torch.long),
        }
    ]
    cfg = SearchConfig(
        activation_rows=32,
        weight_rows=32,
        candidate_window=48,
        neighbor_k=16,
        beam_width_g4=2,
        exact_rerank_g4=4,
        beam_width_g64=2,
        refine_passes=0,
        refine_max_rounds=1,
        refine_candidates_per_round=16,
        seed=42,
    )
    metrics_path = tmp_path / "layer_metrics.jsonl"
    packed = reorder_model_mlps(
        model,
        batches,
        cfg,
        torch.device("cpu"),
        metrics_path=metrics_path,
        num_workers=1,
        apply_candidate_name=apply_candidate_name,
    )
    return packed, metrics_path, cfg


def test_pipeline_jsonl_contains_fixed_fields(tmp_path):
    _packed, metrics_path, cfg = _tiny_pipeline_run(tmp_path)
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    required = {
        "layer_name",
        "selected_candidate",
        "accepted",
        "rejection_reason",
        "search_split_seed",
        "validation_seeds",
        "candidate_metrics",
        "proxy_audit",
        "refinement",
        "elapsed_sec",
    }
    assert required <= set(row)
    assert row["search_split_seed"] == 42
    assert row["validation_seeds"] == [42, 43, 44]
    assert row["split_audit"]["overlap_rows"] == 0
    for cand in ("identity", "hierarchical", "q99_sort_desc"):
        assert cand in row["candidate_metrics"]


def test_pipeline_unaccepted_layer_saves_identity_and_auditable_candidate(tmp_path):
    packed, metrics_path, _cfg = _tiny_pipeline_run(tmp_path)
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    row = rows[0]
    name = row["layer_name"]
    perm = packed["permutations"][name]
    if not row["accepted"]:
        assert torch.equal(perm, torch.arange(perm.numel()))
    else:
        assert row["selected_candidate"] != "identity"
    # The chosen candidate name must be auditable in the metrics record.
    assert row["selected_candidate"] in row["candidate_metrics"]


def test_pipeline_named_candidate_is_applied_to_every_layer(tmp_path):
    packed, _metrics_path, _cfg = _tiny_pipeline_run(
        tmp_path, apply_candidate_name="q99_sort_desc"
    )
    assert packed["applied_candidate"] == "q99_sort_desc"
    assert "q99_sort_desc" in packed["candidate_permutations"]
    for layer_name, applied in packed["permutations"].items():
        expected = packed["candidate_permutations"]["q99_sort_desc"][layer_name]
        assert torch.equal(applied, expected)


def test_pipeline_exports_candidate_even_when_selected_is_identity(tmp_path):
    packed, metrics_path, _cfg = _tiny_pipeline_run(tmp_path)
    assert "candidate_permutations" in packed
    nested = packed["candidate_permutations"]
    assert "hierarchical" in nested
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    row = rows[0]
    name = row["layer_name"]
    for cand_name, perms in nested.items():
        assert name in perms
        candidate = perms[name]
        assert candidate.dtype == torch.long
        assert candidate.device.type == "cpu"
        assert torch.equal(
            torch.sort(candidate).values,
            torch.arange(candidate.numel()),
        )
    # Rejected layer: applied perm is identity, but candidates are still exported.
    if not row["accepted"]:
        applied = packed["permutations"][name]
        assert torch.equal(applied, torch.arange(applied.numel()))
        assert nested["hierarchical"][name].numel() == applied.numel()


def test_output_dir_overwrite_protection(tmp_path):
    from permutation_optimization.run_mlp_reorder import _ensure_output_dir

    target = tmp_path / "fresh"
    _ensure_output_dir(target, overwrite=False)
    (target / "config.json").write_text("{}")
    with pytest.raises(FileExistsError):
        _ensure_output_dir(target, overwrite=False)
    _ensure_output_dir(target, overwrite=True)


def test_accepted_false_returns_identity():
    # Random noise — hierarchical unlikely to beat identity on both metrics consistently;
    # Force reject by checking the accept logic path with a tiny config on pure noise
    # where we can't guarantee accept; instead verify the field semantics.
    torch.manual_seed(0)
    d_ff = 64
    act = torch.randn(40, d_ff)
    w = torch.randn(20, d_ff)
    cfg = SearchConfig(
        validation_fraction=0.2,
        candidate_window=16,
        neighbor_k=8,
        beam_width_g4=2,
        exact_rerank_g4=4,
        beam_width_g64=2,
        refine_passes=0,
        seed=0,
    )
    result = optimize_layer_permutation("noise", act, down_weight=w, config=cfg)
    if not result.accepted:
        assert torch.equal(result.permutation, torch.arange(d_ff))
    else:
        assert torch.equal(result.permutation, result.candidate_permutation)
    assert result.candidate_permutation.numel() == d_ff


def test_deployment_context_matches_module_style_w4a4():
    """DeploymentMLPContext W4A4 forward must match the module-style forward:
    perm-absorbed weights + real HiF4 fake quant + F.linear."""
    from permutation_optimization.hif4_reference import hif4_fake_quantize
    from permutation_optimization.objective import DeploymentMLPContext

    torch.manual_seed(29)
    d_model, d_ff, rows = 64, 128, 8
    g = torch.Generator().manual_seed(29)
    x = (torch.randn(rows, d_model, generator=g) * 0.5).to(torch.bfloat16)
    wu = (torch.randn(d_ff, d_model, generator=g) * 0.05).to(torch.bfloat16)
    wg = (torch.randn(d_ff, d_model, generator=g) * 0.05).to(torch.bfloat16)
    wd = (torch.randn(d_model, d_ff, generator=g) * 0.05).to(torch.bfloat16)
    perm = torch.randperm(d_ff, generator=torch.Generator().manual_seed(31))

    ctx = DeploymentMLPContext(x, wu, wg, wd, torch.device("cpu"))
    y_bf16_ctx, y_w4a4_ctx = ctx._debug_forward(perm)

    wu_p, wg_p, wd_p = wu[perm], wg[perm], wd[:, perm]
    x_q = hif4_fake_quantize(x)
    wg_q = hif4_fake_quantize(wg_p)
    wu_q = hif4_fake_quantize(wu_p)
    a = torch.nn.functional.silu(torch.nn.functional.linear(x_q, wg_q)) * (
        torch.nn.functional.linear(x_q, wu_q)
    )
    a_q = hif4_fake_quantize(a)
    wd_q = hif4_fake_quantize(wd_p)
    y_w4a4_mod = torch.nn.functional.linear(a_q, wd_q)
    y_bf16_mod = torch.nn.functional.linear(
        torch.nn.functional.silu(torch.nn.functional.linear(x, wg_p))
        * torch.nn.functional.linear(x, wu_p),
        wd_p,
    )
    assert torch.allclose(y_w4a4_ctx, y_w4a4_mod, rtol=1e-4, atol=1e-4)
    assert torch.allclose(y_bf16_ctx, y_bf16_mod, rtol=1e-4, atol=1e-4)


def test_fake_quant_keeps_model_dtype_for_rtn_writeback():
    """Quantized weights must re-enter linear ops in the model dtype (BF16),
    simulating an RTN BF16 checkpoint — not FP32."""
    from permutation_optimization.hif4_reference import hif4_fake_quantize

    w = torch.randn(32, 64, dtype=torch.bfloat16)
    w_q = hif4_fake_quantize(w)
    assert w_q.dtype == torch.bfloat16
    x = torch.randn(8, 64, dtype=torch.bfloat16)
    x_q = hif4_fake_quantize(x)
    assert x_q.dtype == torch.bfloat16


def test_apply_then_fp_match_on_tiny_mlp():
    torch.manual_seed(0)
    d_model, d_ff = 32, 64

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Module()])
            self.layers[0].mlp = nn.Module()
            self.layers[0].mlp.gate_proj = nn.Linear(d_model, d_ff, bias=False)
            self.layers[0].mlp.up_proj = nn.Linear(d_model, d_ff, bias=False)
            self.layers[0].mlp.down_proj = nn.Linear(d_ff, d_model, bias=False)

        def forward(self, x):
            m = self.layers[0].mlp
            return m.down_proj(torch.nn.functional.silu(m.gate_proj(x)) * m.up_proj(x))

    model = M()
    specs = discover_swiglu_mlps(model)
    gate, up, down = get_mlp_modules(model, specs[0])
    x = torch.randn(4, d_model)
    y0 = model(x).clone()
    # Build a hierarchical-like perm from optimize on synthetic acts
    act = torch.randn(32, d_ff)
    cfg = SearchConfig(
        candidate_window=16,
        neighbor_k=8,
        beam_width_g4=2,
        exact_rerank_g4=4,
        beam_width_g64=2,
        refine_passes=0,
        validation_fraction=0.25,
    )
    result = optimize_layer_permutation(specs[0].name, act, down_weight=down.weight.detach(), config=cfg)
    apply_mlp_permutation_(gate, up, down, result.permutation)
    y1 = model(x)
    assert torch.allclose(y0, y1, rtol=1e-5, atol=1e-5)
