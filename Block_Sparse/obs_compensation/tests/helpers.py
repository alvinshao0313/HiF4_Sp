from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from block_pruning.mlp_registry import MLPLinearTarget, collect_mlp_linears


def write_source_artifacts(
    root: Path,
    *,
    model_path: str = "tiny-model",
    block_height: int = 2,
    block_width: int = 2,
    mlp_permutation: str = "none",
    residual_permutation: str = "none",
    num_pruning_rounds: int = 1,
    masks: dict[str, torch.Tensor] | None = None,
    permutation_payload: dict[str, dict[str, Any]] | None = None,
    summary_overrides: dict[str, Any] | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    masks = masks or {}
    total = sum(int(mask.numel()) for mask in masks.values())
    pruned = sum(int((~mask).sum().item()) for mask in masks.values())
    summary = {
        "model_path": model_path,
        "block_size": f"{block_height}x{block_width}",
        "block_height": block_height,
        "block_width": block_width,
        "target_block_sparsity": pruned / total if total else 0.5,
        "actual_block_sparsity": pruned / total if total else 0.5,
        "score_type": "fisher_budget_wanda",
        "mlp_permutation": mlp_permutation,
        "residual_permutation": residual_permutation,
        "num_pruning_rounds": num_pruning_rounds,
    }
    if summary_overrides:
        summary.update(summary_overrides)
    (root / "pruning_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    torch.save(masks, root / "block_masks.pt")
    if permutation_payload is not None:
        torch.save(permutation_payload, root / "mlp_permutations.pt")
    return root


class TinyMLP(nn.Module):
    def __init__(self, d_model: int = 4, d_ff: int = 6):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TinyDecoderLayer(nn.Module):
    def __init__(self, d_model: int = 4, d_ff: int = 6):
        super().__init__()
        self.mlp = TinyMLP(d_model=d_model, d_ff=d_ff)

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        del kwargs
        return (hidden_states + self.mlp(hidden_states),)


class TinyCausalLM(nn.Module):
    def __init__(self, num_layers: int = 2, d_model: int = 4, d_ff: int = 8, vocab: int = 16):
        super().__init__()
        self.config = SimpleNamespace(model_type="tiny_causal", use_cache=True)
        self.embed_tokens = nn.Embedding(vocab, d_model)
        self.layers = nn.ModuleList(
            [TinyDecoderLayer(d_model=d_model, d_ff=d_ff) for _ in range(num_layers)]
        )
        self.lm_head = nn.Linear(d_model, vocab, bias=False)
        self._init_deterministic()

    def _init_deterministic(self) -> None:
        torch.manual_seed(0)
        for p in self.parameters():
            if p.ndim >= 2:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, input_ids, attention_mask=None, use_cache=False, **kwargs):
        del attention_mask, use_cache, kwargs
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)[0]
        logits = self.lm_head(hidden)
        return SimpleNamespace(logits=logits)

    def save_pretrained(self, output_dir, safe_serialization=True):
        del safe_serialization
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "tiny_model_marker.txt").write_text("saved", encoding="utf-8")
        torch.save({k: v.detach().cpu() for k, v in self.state_dict().items()}, out / "tiny.pt")


class TinyTokenizer:
    def __call__(self, text, add_special_tokens=False, return_tensors="pt", truncation=False):
        del add_special_tokens, truncation
        ids = [((ord(ch) % 15) + 1) for ch in text] or [1, 2]
        if return_tensors != "pt":
            raise ValueError("only pt supported")
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    def save_pretrained(self, output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "tiny_tokenizer_marker.txt").write_text("tok", encoding="utf-8")


def make_targets_from_tiny(model: TinyCausalLM, block_height: int = 2, block_width: int = 2):
    return collect_mlp_linears(model, block_height=block_height, block_width=block_width)


def make_block_masks_for_targets(
    targets: list[MLPLinearTarget],
    block_height: int,
    block_width: int,
    prune_first_block: bool = True,
) -> dict[str, torch.Tensor]:
    masks: dict[str, torch.Tensor] = {}
    for target in targets:
        d_out, d_in = target.module.weight.shape
        mask = torch.ones(
            d_out // block_height,
            d_in // block_width,
            dtype=torch.bool,
        )
        if prune_first_block:
            # Prefer pruning a right-half block for down to help diagnostics.
            if target.projection_type == "down_proj" and mask.shape[1] >= 2:
                mask[0, -1] = False
            else:
                mask[0, 0] = False
        masks[target.module_name] = mask
    return masks


def make_descending_permutation_payload(
    targets: list[MLPLinearTarget],
    intermediate_size: int,
) -> dict[str, dict[str, Any]]:
    from obs_compensation.permutation import group_mlp_projection_triplets

    triplets = group_mlp_projection_triplets(targets)
    payload: dict[str, dict[str, Any]] = {}
    # Reverse order: old last channel becomes new first (important) position.
    perm = torch.arange(intermediate_size - 1, -1, -1, dtype=torch.int64)
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(intermediate_size, dtype=torch.int64)
    score = torch.arange(intermediate_size, 0, -1, dtype=torch.float32)
    # After applying perm (new[k]=old[perm[k]]), ordered score must descend.
    # perm = [n-1,...,0], score_old = [n,...,1] => ordered = score[perm] = [1,...,n] ASCENDING - bad!
    # We need score such that score[perm[k]] descends. If perm reverses,
    # score should be ascending in old coords: score = [1,2,...,n], ordered = [n,...,1].
    score = torch.arange(1, intermediate_size + 1, dtype=torch.float32)
    for triplet in triplets:
        payload[str(triplet.layer_index)] = {
            "layer_index": triplet.layer_index,
            "gate_module_name": triplet.gate.module_name,
            "up_module_name": triplet.up.module_name,
            "down_module_name": triplet.down.module_name,
            "intermediate_size": intermediate_size,
            "combined_score": score.clone(),
            "permutation": perm.clone(),
            "inverse_permutation": inverse.clone(),
        }
    return payload
