"""Immutable contract for the native Qwen3-30B-A3B ModelOpt NVFP4 checkpoint.

The MoE path deliberately has its own contract instead of extending the legacy
Qwen3-8B/H16 assumptions.  Validation only reads JSON index metadata: it never
materializes checkpoint tensor payloads.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from Native_NVFP4_HiF4_Linear_Puncture.src.checkpoint import resolve_local_snapshot


QWEN3_30B_A3B_NVFP4 = "nvidia/Qwen3-30B-A3B-NVFP4"
NVFP4_SUFFIXES = ("weight", "weight_scale", "weight_scale_2", "input_scale")
ATTENTION_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
EXPERT_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class Qwen3MoeModelSpec:
    source_model: str
    architecture: str
    model_type: str
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_experts: int
    top_k: int
    moe_intermediate_size: int
    decoder_sparse_step: int
    mlp_only_layers: tuple[int, ...]
    norm_topk_prob: bool
    kv_cache_dtype: str = "bfloat16"
    native_has_online_rotation: bool = False

    @property
    def last_layer(self) -> int:
        return self.num_layers - 1

    @property
    def attention_linear_count(self) -> int:
        return self.num_layers * len(ATTENTION_PROJECTIONS)

    @property
    def expert_linear_count(self) -> int:
        return self.num_layers * self.num_experts * len(EXPERT_PROJECTIONS)

    @property
    def target_linear_count(self) -> int:
        return self.attention_linear_count + self.expert_linear_count

    @property
    def router_prefixes(self) -> tuple[str, ...]:
        return tuple(f"model.layers.{i}.mlp.gate" for i in range(self.num_layers))

    def target_prefixes(self) -> tuple[str, ...]:
        attention = tuple(
            f"model.layers.{layer}.self_attn.{proj}"
            for layer in range(self.num_layers)
            for proj in ATTENTION_PROJECTIONS
        )
        experts = tuple(
            f"model.layers.{layer}.mlp.experts.{expert}.{proj}"
            for layer in range(self.num_layers)
            for expert in range(self.num_experts)
            for proj in EXPERT_PROJECTIONS
        )
        return attention + experts

    @classmethod
    def from_snapshot(cls, snapshot: str | Path, *, source_model: str = QWEN3_30B_A3B_NVFP4) -> "Qwen3MoeModelSpec":
        root = Path(snapshot)
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        spec = cls(
            source_model=source_model,
            architecture=(config.get("architectures") or [""])[0],
            model_type=str(config.get("model_type", "")),
            hidden_size=int(config.get("hidden_size", -1)),
            num_layers=int(config.get("num_hidden_layers", -1)),
            num_attention_heads=int(config.get("num_attention_heads", -1)),
            num_key_value_heads=int(config.get("num_key_value_heads", -1)),
            head_dim=int(config.get("head_dim", -1)),
            num_experts=int(config.get("num_experts", -1)),
            top_k=int(config.get("num_experts_per_tok", -1)),
            moe_intermediate_size=int(config.get("moe_intermediate_size", -1)),
            decoder_sparse_step=int(config.get("decoder_sparse_step", -1)),
            mlp_only_layers=tuple(config.get("mlp_only_layers", ())),
            norm_topk_prob=bool(config.get("norm_topk_prob", False)),
        )
        spec.validate_snapshot(root)
        return spec

    @classmethod
    def from_model_path(cls, model_path: str) -> "Qwen3MoeModelSpec":
        path = Path(model_path)
        snapshot = path if path.is_dir() else resolve_local_snapshot(model_path)
        return cls.from_snapshot(snapshot, source_model=model_path)

    def validate_snapshot(self, snapshot: str | Path) -> None:
        root = Path(snapshot)
        quant = json.loads((root / "hf_quant_config.json").read_text(encoding="utf-8"))
        index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
        q = quant.get("quantization") or {}
        errors: list[str] = []
        expected = {
            "architecture": "Qwen3MoeForCausalLM",
            "model_type": "qwen3_moe",
            "hidden_size": 2048,
            "num_layers": 48,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "num_experts": 128,
            "top_k": 8,
            "moe_intermediate_size": 768,
            "decoder_sparse_step": 1,
        }
        for field, value in expected.items():
            if getattr(self, field) != value:
                errors.append(f"{field}={getattr(self, field)!r} != {value!r}")
        if self.mlp_only_layers:
            errors.append(f"mlp_only_layers={self.mlp_only_layers!r} must be empty")
        if not self.norm_topk_prob:
            errors.append("norm_topk_prob must be true")
        if q.get("quant_algo") != "NVFP4":
            errors.append(f"quant_algo={q.get('quant_algo')!r} != 'NVFP4'")
        if q.get("group_size") != 16:
            errors.append(f"group_size={q.get('group_size')!r} != 16")
        if q.get("kv_cache_quant_algo") != "FP8":
            errors.append("checkpoint kv_cache_quant_algo must be 'FP8'")
        excludes = set(q.get("exclude_modules") or ())
        expected_excludes = set(self.router_prefixes) | {"lm_head"}
        if excludes != expected_excludes:
            errors.append(
                f"exclude_modules mismatch missing={sorted(expected_excludes - excludes)} "
                f"extra={sorted(excludes - expected_excludes)}"
            )
        keys = set((index.get("weight_map") or {}).keys())
        missing = [
            f"{prefix}.{suffix}"
            for prefix in self.target_prefixes()
            for suffix in NVFP4_SUFFIXES
            if f"{prefix}.{suffix}" not in keys
        ]
        if missing:
            errors.append(f"missing NVFP4 tensors count={len(missing)} first={missing[:3]}")
        hadamard = [key for key in keys if key.endswith("forward_hadamard_matrix")]
        if hadamard:
            errors.append(f"native H16 tensors are forbidden, found {hadamard[:3]}")
        if errors:
            raise ValueError("Qwen3 MoE checkpoint contract failed: " + "; ".join(errors))


def load_qwen3_moe_model_spec(model_path: str = QWEN3_30B_A3B_NVFP4) -> Qwen3MoeModelSpec:
    return Qwen3MoeModelSpec.from_model_path(model_path)
