from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from block_pruning.config import GradientBlockPruningConfig
from block_pruning.mlp_registry import MLPLinearTarget
from block_pruning.residual_permutation import (
    ResidualPermutationRecord,
    apply_residual_permutation,
    build_mlp_search_cache,
    compute_init_residual_permutation,
    compute_residual_channel_scores,
    discover_residual_mounts,
    evaluate_residual_permutation_loss,
    local_search_residual_permutation,
    release_mlp_search_cache,
    resolve_hidden_size,
    undo_residual_permutation,
)
from block_pruning.serialization import save_residual_permutation_artifacts
from block_pruning.wanda_scorer import InputRMSRecord


class _RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt()
        return x * rms * self.weight


def _fake_rms(targets: list[MLPLinearTarget]) -> dict[str, InputRMSRecord]:
    records: dict[str, InputRMSRecord] = {}
    for t in targets:
        d_in = int(t.module.weight.shape[1])
        rms = torch.linspace(0.5, 1.5, d_in)
        records[t.module_name] = InputRMSRecord(
            module_name=t.module_name,
            layer_index=t.layer_index,
            projection_type=t.projection_type,
            num_tokens=32,
            channel_square_sum=rms.double().square() * 32,
            input_rms=rms.double(),
        )
    return records


def test_discover_and_equivalence_logits():
    torch.manual_seed(0)

    class NamedTiny(nn.Module):
        def __init__(self):
            super().__init__()
            vocab, hidden, mid, ff = 16, 4, 6, 8
            self.config = type("Cfg", (), {"hidden_size": hidden})()
            self.embed_tokens = nn.Embedding(vocab, hidden)
            self.layers = nn.ModuleList([nn.Module()])
            layer = self.layers[0]
            layer.input_layernorm = _RMSNorm(hidden)
            layer.self_attn = nn.Module()
            # Non-square so intermediate is not residual.
            layer.self_attn.q_proj = nn.Linear(hidden, mid, bias=False)
            layer.self_attn.o_proj = nn.Linear(mid, hidden, bias=False)
            layer.post_attention_layernorm = _RMSNorm(hidden)
            layer.mlp = nn.Module()
            layer.mlp.gate_proj = nn.Linear(hidden, ff, bias=False)
            layer.mlp.up_proj = nn.Linear(hidden, ff, bias=False)
            layer.mlp.down_proj = nn.Linear(ff, hidden, bias=False)
            self.norm = _RMSNorm(hidden)
            self.lm_head = nn.Linear(hidden, vocab, bias=False)
            with torch.no_grad():
                for p in self.parameters():
                    p.uniform_(-0.5, 0.5)

        def forward(self, input_ids):
            x = self.embed_tokens(input_ids)
            layer = self.layers[0]
            h = layer.input_layernorm(x)
            h = layer.self_attn.o_proj(layer.self_attn.q_proj(h))
            x = x + h
            h = layer.post_attention_layernorm(x)
            h = layer.mlp.down_proj(
                F.silu(layer.mlp.gate_proj(h)) * layer.mlp.up_proj(h)
            )
            x = x + h
            return self.lm_head(self.norm(x))

    named = NamedTiny()
    targets = [
        MLPLinearTarget(
            "layers.0.mlp.gate_proj", named.layers[0].mlp.gate_proj, 0, "gate_proj"
        ),
        MLPLinearTarget(
            "layers.0.mlp.up_proj", named.layers[0].mlp.up_proj, 0, "up_proj"
        ),
        MLPLinearTarget(
            "layers.0.mlp.down_proj", named.layers[0].mlp.down_proj, 0, "down_proj"
        ),
    ]
    hidden = resolve_hidden_size(named, targets)
    assert hidden == 4
    mounts = discover_residual_mounts(named, hidden)
    names = {m.name for m in mounts}
    assert "embed_tokens.weight" in names
    assert "lm_head.weight" in names
    assert any("input_layernorm.weight" in n for n in names)
    assert any("q_proj.weight" in n for n in names)
    assert any("o_proj.weight" in n for n in names)
    assert any("gate_proj.weight" in n for n in names)
    assert any("down_proj.weight" in n for n in names)

    ids = torch.randint(0, 16, (2, 5))
    with torch.no_grad():
        logits0 = named(ids).clone()

    perm = torch.tensor([2, 0, 3, 1], dtype=torch.int64)
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(4, dtype=torch.int64)
    apply_residual_permutation(named, mounts, perm, hidden)
    with torch.no_grad():
        logits1 = named(ids).clone()
    torch.testing.assert_close(logits0, logits1, atol=1e-5, rtol=1e-5)

    undo_residual_permutation(named, mounts, inverse, hidden)
    with torch.no_grad():
        logits2 = named(ids).clone()
    torch.testing.assert_close(logits0, logits2, atol=1e-5, rtol=1e-5)


def test_negative_control_skip_lm_head():
    torch.manual_seed(1)

    class NamedTiny(nn.Module):
        def __init__(self):
            super().__init__()
            vocab, hidden, mid, ff = 16, 4, 6, 8
            self.config = type("Cfg", (), {"hidden_size": hidden})()
            self.embed_tokens = nn.Embedding(vocab, hidden)
            self.layers = nn.ModuleList([nn.Module()])
            layer = self.layers[0]
            layer.input_layernorm = _RMSNorm(hidden)
            layer.self_attn = nn.Module()
            layer.self_attn.q_proj = nn.Linear(hidden, mid, bias=False)
            layer.self_attn.o_proj = nn.Linear(mid, hidden, bias=False)
            layer.post_attention_layernorm = _RMSNorm(hidden)
            layer.mlp = nn.Module()
            layer.mlp.gate_proj = nn.Linear(hidden, ff, bias=False)
            layer.mlp.up_proj = nn.Linear(hidden, ff, bias=False)
            layer.mlp.down_proj = nn.Linear(ff, hidden, bias=False)
            self.norm = _RMSNorm(hidden)
            self.lm_head = nn.Linear(hidden, vocab, bias=False)
            with torch.no_grad():
                for p in self.parameters():
                    p.uniform_(-0.5, 0.5)

        def forward(self, input_ids):
            x = self.embed_tokens(input_ids)
            layer = self.layers[0]
            h = layer.input_layernorm(x)
            h = layer.self_attn.o_proj(layer.self_attn.q_proj(h))
            x = x + h
            h = layer.post_attention_layernorm(x)
            h = layer.mlp.down_proj(
                F.silu(layer.mlp.gate_proj(h)) * layer.mlp.up_proj(h)
            )
            x = x + h
            return self.lm_head(self.norm(x))

    named = NamedTiny()
    targets = [
        MLPLinearTarget(
            "layers.0.mlp.gate_proj", named.layers[0].mlp.gate_proj, 0, "gate_proj"
        ),
        MLPLinearTarget(
            "layers.0.mlp.up_proj", named.layers[0].mlp.up_proj, 0, "up_proj"
        ),
        MLPLinearTarget(
            "layers.0.mlp.down_proj", named.layers[0].mlp.down_proj, 0, "down_proj"
        ),
    ]
    hidden = resolve_hidden_size(named, targets)
    mounts = discover_residual_mounts(named, hidden)
    mounts_no_head = [m for m in mounts if not m.name.endswith("lm_head.weight")]
    assert len(mounts_no_head) < len(mounts)

    ids = torch.randint(0, 16, (2, 5))
    with torch.no_grad():
        logits0 = named(ids).float().clone()
    perm = torch.tensor([3, 1, 0, 2], dtype=torch.int64)
    apply_residual_permutation(named, mounts_no_head, perm, hidden)
    with torch.no_grad():
        logits1 = named(ids).float().clone()
    delta = (logits0 - logits1).abs().max().item()
    assert delta > 1e-3, f"expected mismatch when skipping lm_head, delta={delta}"


def test_search_loss_non_increasing():
    torch.manual_seed(2)
    hidden, ff = 8, 16
    gate = nn.Linear(hidden, ff, bias=False)
    up = nn.Linear(hidden, ff, bias=False)
    down = nn.Linear(ff, hidden, bias=False)
    with torch.no_grad():
        gate.weight.normal_(0, 1)
        up.weight.normal_(0, 1)
        down.weight.normal_(0, 1)
        # Make residual channel importance uneven so swaps can help.
        gate.weight[:, :2] *= 5
        up.weight[:, :2] *= 5
        down.weight[:2, :] *= 5

    targets = [
        MLPLinearTarget("layers.0.mlp.gate_proj", gate, 0, "gate_proj"),
        MLPLinearTarget("layers.0.mlp.up_proj", up, 0, "up_proj"),
        MLPLinearTarget("layers.0.mlp.down_proj", down, 0, "down_proj"),
    ]
    rms = _fake_rms(targets)
    config = GradientBlockPruningConfig(
        block_size="4",
        target_block_sparsity=0.25,
        score_type="magnitude",
        residual_permutation="block_loss",
        residual_perm_search_steps=200,
        max_prune_ratio_per_matrix=1.0,
        min_keep_blocks_per_matrix=1,
        seed=123,
    )
    config.validate()
    cache = build_mlp_search_cache(targets, rms, config, hidden)
    channel_score = compute_residual_channel_scores(cache)
    perm0, _ = compute_init_residual_permutation(channel_score)
    best, loss_init, loss_final, accepted = local_search_residual_permutation(
        perm0,
        cache,
        config,
        score_mode="magnitude",
        module_budgets=None,
    )
    assert loss_final <= loss_init + 1e-12
    # Identity should not beat a proper search when importance is uneven and
    # steps>0; at minimum final equals init when no improving swap exists.
    loss_id = evaluate_residual_permutation_loss(
        cache,
        torch.arange(hidden, dtype=torch.int64),
        config,
        score_mode="magnitude",
        module_budgets=None,
    )
    assert loss_final <= loss_id + 1e-12
    assert best.numel() == hidden
    assert accepted >= 0


def test_config_rejects_random_with_block_loss():
    try:
        cfg = GradientBlockPruningConfig(
            score_type="random",
            residual_permutation="block_loss",
            block_size="4",
            target_block_sparsity=0.2,
        )
        cfg.validate()
        assert False, "expected validation error"
    except ValueError as e:
        assert "random" in str(e)


def test_save_residual_artifacts(tmp_path: Path | None = None):
    out = Path(tmp_path) if tmp_path is not None else Path("/tmp/residual_perm_test")
    out.mkdir(parents=True, exist_ok=True)
    config = GradientBlockPruningConfig(
        residual_permutation="block_loss",
        residual_perm_search_steps=10,
        block_size="4",
        target_block_sparsity=0.2,
        score_type="magnitude",
    )
    config.validate()
    hidden = 4
    perm = torch.arange(hidden, dtype=torch.int64).flip(0)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(hidden, dtype=torch.int64)
    record = ResidualPermutationRecord(
        hidden_size=hidden,
        permutation=perm,
        inverse_permutation=inv,
        channel_score=torch.arange(hidden, dtype=torch.float64),
        loss_init=1.0,
        loss_final=0.5,
        search_steps=10,
        accepted_swaps=2,
        mount_names=["embed_tokens.weight", "lm_head.weight"],
        search_score_mode="magnitude",
    )
    save_residual_permutation_artifacts(out, record, config)
    assert (out / "residual_permutation.pt").is_file()
    assert (out / "residual_permutation_summary.json").is_file()
    payload = torch.load(out / "residual_permutation.pt", map_location="cpu", weights_only=False)
    assert payload["loss_final"] == 0.5
    torch.testing.assert_close(payload["permutation"], perm)


def test_gpu_search_smoke_matches_cpu_init_and_nonincreasing():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(2)
    hidden, ff = 8, 16
    device = torch.device("cuda:0")

    def _make_targets(dev: torch.device):
        gate = nn.Linear(hidden, ff, bias=False).to(dev)
        up = nn.Linear(hidden, ff, bias=False).to(dev)
        down = nn.Linear(ff, hidden, bias=False).to(dev)
        with torch.no_grad():
            gate.weight.normal_(0, 1)
            up.weight.normal_(0, 1)
            down.weight.normal_(0, 1)
            gate.weight[:, :2] *= 5
            up.weight[:, :2] *= 5
            down.weight[:2, :] *= 5
        targets = [
            MLPLinearTarget("layers.0.mlp.gate_proj", gate, 0, "gate_proj"),
            MLPLinearTarget("layers.0.mlp.up_proj", up, 0, "up_proj"),
            MLPLinearTarget("layers.0.mlp.down_proj", down, 0, "down_proj"),
        ]
        return targets

    config = GradientBlockPruningConfig(
        block_size="4",
        target_block_sparsity=0.25,
        score_type="magnitude",
        residual_permutation="block_loss",
        residual_perm_search_steps=50,
        max_prune_ratio_per_matrix=1.0,
        min_keep_blocks_per_matrix=1,
        seed=123,
    )
    config.validate()

    # CPU reference init permutation
    cpu_targets = _make_targets(torch.device("cpu"))
    # Sync identical weights onto CUDA copy
    cuda_targets = _make_targets(device)
    with torch.no_grad():
        for c, g in zip(cpu_targets, cuda_targets):
            g.module.weight.copy_(c.module.weight.to(device))

    rms_cpu = _fake_rms(cpu_targets)
    rms_cuda = {
        t.module_name: InputRMSRecord(
            module_name=t.module_name,
            layer_index=t.layer_index,
            projection_type=t.projection_type,
            num_tokens=rms_cpu[t.module_name].num_tokens,
            channel_square_sum=rms_cpu[t.module_name].channel_square_sum.clone(),
            input_rms=rms_cpu[t.module_name].input_rms.clone(),
        )
        for t in cuda_targets
    }

    cache_cpu = build_mlp_search_cache(cpu_targets, rms_cpu, config, hidden)
    cache_cuda = build_mlp_search_cache(cuda_targets, rms_cuda, config, hidden)
    assert all(w.is_cuda for w in cache_cuda.weights.values())

    score_cpu = compute_residual_channel_scores(cache_cpu)
    score_cuda = compute_residual_channel_scores(cache_cuda)
    torch.testing.assert_close(score_cpu, score_cuda, atol=1e-5, rtol=1e-5)
    perm0_cpu, _ = compute_init_residual_permutation(score_cpu)
    perm0_cuda, _ = compute_init_residual_permutation(score_cuda)
    torch.testing.assert_close(perm0_cpu, perm0_cuda)

    best, loss_init, loss_final, accepted = local_search_residual_permutation(
        perm0_cuda,
        cache_cuda,
        config,
        score_mode="magnitude",
        module_budgets=None,
    )
    assert loss_final <= loss_init + 1e-8
    assert best.numel() == hidden
    assert accepted >= 0
    release_mlp_search_cache(cache_cuda)
    assert len(cache_cuda.weights) == 0


def test_fast_loss_matches_allocator_paths():
    """Fast L paths must match full allocator pruned-score sums."""
    from block_pruning.mask_allocator import (
        allocate_block_masks,
        allocate_masks_by_module_budget,
    )
    from block_pruning.residual_permutation import (
        build_virtual_score_records,
        pruned_block_score_sum,
    )

    torch.manual_seed(0)
    hidden, ff = 8, 16
    gate = nn.Linear(hidden, ff, bias=False)
    up = nn.Linear(hidden, ff, bias=False)
    down = nn.Linear(ff, hidden, bias=False)
    with torch.no_grad():
        gate.weight.normal_()
        up.weight.normal_()
        down.weight.normal_()
    targets = [
        MLPLinearTarget("layers.0.mlp.gate_proj", gate, 0, "gate_proj"),
        MLPLinearTarget("layers.0.mlp.up_proj", up, 0, "up_proj"),
        MLPLinearTarget("layers.0.mlp.down_proj", down, 0, "down_proj"),
    ]
    rms = _fake_rms(targets)
    config = GradientBlockPruningConfig(
        block_size="4",
        target_block_sparsity=0.25,
        score_type="magnitude",
        residual_permutation="block_loss",
        residual_perm_search_steps=0,
        max_prune_ratio_per_matrix=1.0,
        min_keep_blocks_per_matrix=1,
        seed=0,
    )
    config.validate()
    cache = build_mlp_search_cache(targets, rms, config, hidden)
    perm = torch.arange(hidden, dtype=torch.int64)

    # Global magnitude path
    loss_fast = evaluate_residual_permutation_loss(
        cache, perm, config, score_mode="magnitude", module_budgets=None
    )
    records = build_virtual_score_records(cache, perm, config, "magnitude")
    alloc = allocate_block_masks(
        records, config, cache.masks, ranking_score_type="magnitude"
    )
    loss_ref = pruned_block_score_sum(records, alloc.masks, "magnitude")
    assert abs(loss_fast - loss_ref) < 1e-8

    # Per-module budget Wanda path
    budgets = {t.module_name: max(1, cache.masks[t.module_name].numel() // 4) for t in targets}
    loss_fast_b = evaluate_residual_permutation_loss(
        cache, perm, config, score_mode="wanda", module_budgets=budgets
    )
    records_w = build_virtual_score_records(cache, perm, config, "wanda")
    alloc_b = allocate_masks_by_module_budget(
        records_w, budgets, config, cache.masks, ranking_score_type="wanda"
    )
    loss_ref_b = pruned_block_score_sum(records_w, alloc_b.masks, "wanda")
    assert abs(loss_fast_b - loss_ref_b) < 1e-8


if __name__ == "__main__":
    test_discover_and_equivalence_logits()
    test_negative_control_skip_lm_head()
    test_search_loss_non_increasing()
    test_config_rejects_random_with_block_loss()
    test_save_residual_artifacts(Path("/tmp/residual_perm_art"))
    test_gpu_search_smoke_matches_cpu_init_and_nonincreasing()
    test_fast_loss_matches_allocator_paths()
    print("ok")
