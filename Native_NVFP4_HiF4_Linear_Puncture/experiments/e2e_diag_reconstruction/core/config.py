"""E2E DIAG reconstruction: unique CLI defaults and legality checks."""

from __future__ import annotations

import argparse
import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_model_spec import (
    QWEN3_30B_A3B_NVFP4,
)

DEFAULT_MODEL_PATH = QWEN3_30B_A3B_NVFP4
# Legacy 8B code owns its own static model contract.  The new MoE path gets
# layer bounds and coverage from Qwen3MoeModelSpec at runtime.
DEFAULT_LAST_LAYER_INDEX = 47
# Transitional exports used only by the untouched legacy 8B implementation.
# They must never validate or size the Qwen3-MoE path.
NUM_DECODER_LAYERS = 36
LAST_LAYER_INDEX = NUM_DECODER_LAYERS - 1
TARGET_LINEARS_PER_LAYER = 7
TARGET_LINEAR_COUNT = NUM_DECODER_LAYERS * TARGET_LINEARS_PER_LAYER

DIAG_MODES = ("fusable", "online")
ROT_ORDERS = ("diag_then_rot", "rot_then_diag")
TRAIN_SCOPES = ("layer_joint", "linear_independent")
RECON_LOSSES = ("block_delta_nmse", "block_output_nmse", "mse")
SCHEDULERS = ("cosine", "constant")
CALIB_SOURCES = (
    "s1k_teacher_cot",
    "s1k_original",
    "s1k_question",
    "wikitext2",
    "c4",
)
TEACHER_TRACE_POLICIES = ("all", "regenerate_correct", "replace_question_correct")
FUSABLE_DIAG_COMPONENTS = ("all", "qkv", "vo", "gu", "ud", "attn", "mlp")
FUSABLE_COMPONENT_MAP = {
    "all": frozenset({"qkv", "vo", "gu", "ud"}),
    "qkv": frozenset({"qkv"}),
    "vo": frozenset({"vo"}),
    "gu": frozenset({"gu"}),
    "ud": frozenset({"ud"}),
    "attn": frozenset({"qkv", "vo"}),
    "mlp": frozenset({"gu", "ud"}),
}
CALIB_INPUT_MODES = ("progressive_student", "teacher")
LAYER_ROLLBACK_MODES = ("on", "off")
ROLLBACK_OVERRIDE_MODES = ("inherit", "on", "off")
ROUTER_ALIGN_TYPE = "kl"
ROUTER_ALIGN_TEMPERATURE = 1.0
WINDOW_CALIB_SOURCES = frozenset({"wikitext2", "c4"})
S1K_CALIB_SOURCES = frozenset({"s1k_teacher_cot", "s1k_original", "s1k_question"})


def parse_log2_clamp(text: str) -> tuple[float, float] | None:
    if text.strip().lower() == "none":
        return None
    parts = [x.strip() for x in text.split(",")]
    if len(parts) != 2:
        raise ValueError("--diag_log2_clamp must be 'none' or 'MIN,MAX'")
    lo, hi = map(float, parts)
    if not math.isfinite(lo) or not math.isfinite(hi) or not lo < hi:
        raise ValueError("--diag_log2_clamp requires finite MIN < MAX")
    return lo, hi


def format_log2_clamp(value: tuple[float, float] | None) -> str:
    if value is None:
        return "none"
    return f"{value[0]},{value[1]}"


def resolve_rollback_enabled(master: str, override: str) -> bool:
    """Resolve a per-cause rollback override while keeping legacy master semantics."""
    if override == "inherit":
        return master == "on"
    return override == "on"


@dataclass
class E2ETrainConfig:
    model_path: str
    output_dir: str
    diag_mode: str = "fusable"
    use_r64: bool = False
    rot_order: str = "diag_then_rot"
    diag_train_scope: str = "layer_joint"
    recon_loss: str = "block_delta_nmse"
    attn_aux_loss_weight: float = 0.0
    mlp_aux_loss_weight: float = 0.0
    diag_lr: float = 5e-3
    diag_epochs: int = 20
    diag_scheduler: str = "cosine"
    diag_log2_clamp: tuple[float, float] | None = (-4.0, 4.0)
    calib_source: str = "s1k_original"
    calib_nsamples: int = 128
    calib_val_nsamples: int = 32
    calib_seed: int = 42
    calib_seqlen: int = 1024
    diag_batch_size: int = 4
    teacher_trace_policy: str = "all"
    teacher_max_attempts: int = 4
    teacher_max_new_tokens: int = 32768
    start_layer: int = 0
    end_layer: int = DEFAULT_LAST_LAYER_INDEX
    calib_cache_dir: str = ""
    fusable_diag_components: str = "all"
    calib_input_mode: str = "progressive_student"
    layer_rollback: str = "on"
    loss_rollback: str = "inherit"
    router_rollback: str = "inherit"
    router_align_loss_weight: float = 0.0
    model_type: str = "qwen3_moe"
    num_layers: int = 48
    num_experts: int = 128
    top_k: int = 8
    kv_cache_dtype: str = "bfloat16"
    native_has_online_rotation: bool = False

    @classmethod
    def for_test(cls, **overrides: Any) -> E2ETrainConfig:
        kwargs: dict[str, Any] = {
            "model_path": DEFAULT_MODEL_PATH,
            "output_dir": "/tmp/e2e_diag_reconstruction_test",
        }
        kwargs.update(overrides)
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["diag_log2_clamp"] = format_log2_clamp(self.diag_log2_clamp)
        d["output_dir"] = str(self.output_dir)
        return d


def build_train_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Native NVFP4 → HiF4 layerwise DIAG reconstruction"
    )
    p.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--diag_mode", type=str, choices=DIAG_MODES, default="fusable")
    p.add_argument("--use_r64", action="store_true")
    p.add_argument(
        "--rot_order", type=str, choices=ROT_ORDERS, default="diag_then_rot"
    )
    p.add_argument(
        "--diag_train_scope",
        type=str,
        choices=TRAIN_SCOPES,
        default="layer_joint",
    )
    p.add_argument(
        "--recon_loss", type=str, choices=RECON_LOSSES, default="block_delta_nmse"
    )
    p.add_argument("--attn_aux_loss_weight", type=float, default=0.0)
    p.add_argument("--mlp_aux_loss_weight", type=float, default=0.0)
    p.add_argument("--diag_lr", type=float, default=5e-3)
    p.add_argument("--diag_epochs", type=int, default=20)
    p.add_argument(
        "--diag_scheduler", type=str, choices=SCHEDULERS, default="cosine"
    )
    p.add_argument("--diag_log2_clamp", type=str, default="-4,4")
    p.add_argument(
        "--calib_source", type=str, choices=CALIB_SOURCES, default="s1k_original"
    )
    p.add_argument("--calib_nsamples", type=int, default=128)
    p.add_argument("--calib_val_nsamples", type=int, default=32)
    p.add_argument("--calib_seed", type=int, default=42)
    p.add_argument("--calib_seqlen", type=int, default=1024)
    p.add_argument("--diag_batch_size", type=int, default=4)
    p.add_argument(
        "--teacher_trace_policy",
        type=str,
        choices=TEACHER_TRACE_POLICIES,
        default="all",
    )
    p.add_argument("--teacher_max_attempts", type=int, default=4)
    p.add_argument("--teacher_max_new_tokens", type=int, default=32768)
    p.add_argument("--start_layer", type=int, default=0)
    p.add_argument("--end_layer", type=int, default=DEFAULT_LAST_LAYER_INDEX)
    p.add_argument("--calib_cache_dir", type=str, default="")
    p.add_argument(
        "--fusable_diag_components",
        type=str,
        choices=FUSABLE_DIAG_COMPONENTS,
        default="all",
    )
    p.add_argument(
        "--calib_input_mode",
        type=str,
        choices=CALIB_INPUT_MODES,
        default="progressive_student",
    )
    p.add_argument(
        "--layer_rollback",
        type=str,
        choices=LAYER_ROLLBACK_MODES,
        default="on",
    )
    p.add_argument(
        "--loss_rollback",
        type=str,
        choices=ROLLBACK_OVERRIDE_MODES,
        default="inherit",
    )
    p.add_argument(
        "--router_rollback",
        type=str,
        choices=ROLLBACK_OVERRIDE_MODES,
        default="inherit",
    )
    p.add_argument("--router_align_loss_weight", type=float, default=0.0)
    return p


def config_from_namespace(args: argparse.Namespace) -> E2ETrainConfig:
    return E2ETrainConfig(
        model_path=str(args.model_path),
        output_dir=str(args.output_dir),
        diag_mode=str(args.diag_mode),
        use_r64=bool(args.use_r64),
        rot_order=str(args.rot_order),
        diag_train_scope=str(args.diag_train_scope),
        recon_loss=str(args.recon_loss),
        attn_aux_loss_weight=float(args.attn_aux_loss_weight),
        mlp_aux_loss_weight=float(args.mlp_aux_loss_weight),
        diag_lr=float(args.diag_lr),
        diag_epochs=int(args.diag_epochs),
        diag_scheduler=str(args.diag_scheduler),
        diag_log2_clamp=parse_log2_clamp(str(args.diag_log2_clamp)),
        calib_source=str(args.calib_source),
        calib_nsamples=int(args.calib_nsamples),
        calib_val_nsamples=int(args.calib_val_nsamples),
        calib_seed=int(args.calib_seed),
        calib_seqlen=int(args.calib_seqlen),
        diag_batch_size=int(args.diag_batch_size),
        teacher_trace_policy=str(args.teacher_trace_policy),
        teacher_max_attempts=int(args.teacher_max_attempts),
        teacher_max_new_tokens=int(args.teacher_max_new_tokens),
        start_layer=int(args.start_layer),
        end_layer=int(args.end_layer),
        calib_cache_dir=str(args.calib_cache_dir),
        fusable_diag_components=str(args.fusable_diag_components),
        calib_input_mode=str(args.calib_input_mode),
        layer_rollback=str(args.layer_rollback),
        loss_rollback=str(args.loss_rollback),
        router_rollback=str(args.router_rollback),
        router_align_loss_weight=float(args.router_align_loss_weight),
    )


def parse_train_args(argv: list[str] | None = None) -> E2ETrainConfig:
    args = build_train_parser().parse_args(argv)
    cfg = config_from_namespace(args)
    validate_train_config(cfg)
    return cfg


def validate_train_config(cfg: E2ETrainConfig) -> None:
    if not str(cfg.output_dir).strip():
        raise ValueError("--output_dir is required and must be non-empty")
    if cfg.diag_mode not in DIAG_MODES:
        raise ValueError(f"invalid diag_mode={cfg.diag_mode!r}")
    if cfg.rot_order not in ROT_ORDERS:
        raise ValueError(f"invalid rot_order={cfg.rot_order!r}")
    if cfg.diag_train_scope not in TRAIN_SCOPES:
        raise ValueError(f"invalid diag_train_scope={cfg.diag_train_scope!r}")
    if cfg.recon_loss not in RECON_LOSSES:
        raise ValueError(f"invalid recon_loss={cfg.recon_loss!r}")
    if cfg.diag_scheduler not in SCHEDULERS:
        raise ValueError(f"invalid diag_scheduler={cfg.diag_scheduler!r}")
    if cfg.calib_source not in CALIB_SOURCES:
        raise ValueError(f"invalid calib_source={cfg.calib_source!r}")
    if cfg.teacher_trace_policy not in TEACHER_TRACE_POLICIES:
        raise ValueError(f"invalid teacher_trace_policy={cfg.teacher_trace_policy!r}")
    if cfg.fusable_diag_components not in FUSABLE_DIAG_COMPONENTS:
        raise ValueError(
            f"invalid fusable_diag_components={cfg.fusable_diag_components!r}"
        )
    if cfg.calib_input_mode not in CALIB_INPUT_MODES:
        raise ValueError(f"invalid calib_input_mode={cfg.calib_input_mode!r}")
    if cfg.layer_rollback not in LAYER_ROLLBACK_MODES:
        raise ValueError(f"invalid layer_rollback={cfg.layer_rollback!r}")
    if cfg.loss_rollback not in ROLLBACK_OVERRIDE_MODES:
        raise ValueError(f"invalid loss_rollback={cfg.loss_rollback!r}")
    if cfg.router_rollback not in ROLLBACK_OVERRIDE_MODES:
        raise ValueError(f"invalid router_rollback={cfg.router_rollback!r}")
    if not math.isfinite(cfg.router_align_loss_weight) or cfg.router_align_loss_weight < 0:
        raise ValueError("router_align_loss_weight must be finite and >= 0")

    if cfg.diag_mode == "online" and cfg.fusable_diag_components != "all":
        raise ValueError("diag_mode=online only allows fusable_diag_components=all")
    if cfg.diag_mode != "fusable" and cfg.router_align_loss_weight > 0:
        raise ValueError("router_align_loss_weight > 0 is only valid for diag_mode=fusable")

    if cfg.diag_mode == "fusable" and cfg.rot_order == "rot_then_diag":
        raise ValueError("fusable does not allow rot_then_diag")
    if cfg.diag_mode == "fusable" and cfg.diag_train_scope == "linear_independent":
        raise ValueError("fusable does not allow linear_independent")
    if cfg.model_type == "qwen3_moe" and cfg.diag_train_scope != "layer_joint":
        raise ValueError("Qwen3 MoE only allows diag_train_scope=layer_joint")
    if cfg.model_type == "qwen3_moe":
        if (cfg.num_layers, cfg.num_experts, cfg.top_k) != (48, 128, 8):
            raise ValueError("Qwen3 MoE config contract must be 48 layers / 128 experts / top8")
        if cfg.kv_cache_dtype != "bfloat16":
            raise ValueError("Qwen3 MoE formal path requires kv_cache_dtype=bfloat16")
        if cfg.native_has_online_rotation:
            raise ValueError("Qwen3 MoE native path must not have online H16 rotation")

    last_layer = cfg.num_layers - 1
    if not (0 <= cfg.start_layer <= cfg.end_layer <= last_layer):
        raise ValueError(
            f"layer range must satisfy 0 <= start_layer <= end_layer <= {last_layer}, "
            f"got start_layer={cfg.start_layer}, end_layer={cfg.end_layer}"
        )
    if cfg.calib_nsamples <= 0:
        raise ValueError(f"--calib_nsamples must be > 0, got {cfg.calib_nsamples}")
    if cfg.calib_val_nsamples <= 0:
        raise ValueError(
            f"--calib_val_nsamples must be > 0, got {cfg.calib_val_nsamples}"
        )
    if cfg.diag_batch_size <= 0:
        raise ValueError(f"--diag_batch_size must be > 0, got {cfg.diag_batch_size}")
    if cfg.diag_epochs <= 0:
        raise ValueError(f"--diag_epochs must be > 0, got {cfg.diag_epochs}")
    if not math.isfinite(cfg.diag_lr) or cfg.diag_lr <= 0:
        raise ValueError(f"--diag_lr must be a finite value > 0, got {cfg.diag_lr}")
    if cfg.teacher_max_attempts <= 0:
        raise ValueError(
            f"--teacher_max_attempts must be > 0, got {cfg.teacher_max_attempts}"
        )
    if cfg.teacher_max_new_tokens <= 0:
        raise ValueError(
            f"--teacher_max_new_tokens must be > 0, got {cfg.teacher_max_new_tokens}"
        )
    if not math.isfinite(cfg.attn_aux_loss_weight) or cfg.attn_aux_loss_weight < 0:
        raise ValueError("--attn_aux_loss_weight must be finite and >= 0")
    if not math.isfinite(cfg.mlp_aux_loss_weight) or cfg.mlp_aux_loss_weight < 0:
        raise ValueError("--mlp_aux_loss_weight must be finite and >= 0")
    if cfg.diag_log2_clamp is not None:
        lo, hi = cfg.diag_log2_clamp
        if not math.isfinite(lo) or not math.isfinite(hi) or not lo < hi:
            raise ValueError("--diag_log2_clamp requires finite MIN < MAX")
    if cfg.calib_source in WINDOW_CALIB_SOURCES and cfg.calib_seqlen <= 0:
        raise ValueError(
            f"--calib_seqlen must be > 0 for window dataset {cfg.calib_source}, "
            f"got {cfg.calib_seqlen}"
        )


def resolved_config_lines(cfg: E2ETrainConfig, extra: dict[str, Any] | None = None) -> list[str]:
    rows = [
        f"model_path={cfg.model_path}",
        f"output_dir={cfg.output_dir}",
        f"diag_mode={cfg.diag_mode}",
        f"use_r64={cfg.use_r64}",
        f"rot_order={cfg.rot_order}",
        f"diag_train_scope={cfg.diag_train_scope}",
        f"recon_loss={cfg.recon_loss}",
        f"attn_aux_loss_weight={cfg.attn_aux_loss_weight}",
        f"mlp_aux_loss_weight={cfg.mlp_aux_loss_weight}",
        f"diag_lr={cfg.diag_lr}",
        f"diag_epochs={cfg.diag_epochs}",
        f"diag_scheduler={cfg.diag_scheduler}",
        f"diag_log2_clamp={format_log2_clamp(cfg.diag_log2_clamp)}",
        f"calib_source={cfg.calib_source}",
        f"calib_nsamples={cfg.calib_nsamples}",
        f"calib_val_nsamples={cfg.calib_val_nsamples}",
        f"calib_seed={cfg.calib_seed}",
        f"calib_seqlen={cfg.calib_seqlen}",
        f"diag_batch_size={cfg.diag_batch_size}",
        f"teacher_trace_policy={cfg.teacher_trace_policy}",
        f"teacher_max_attempts={cfg.teacher_max_attempts}",
        f"teacher_max_new_tokens={cfg.teacher_max_new_tokens}",
        f"start_layer={cfg.start_layer}",
        f"end_layer={cfg.end_layer}",
        f"calib_cache_dir={cfg.calib_cache_dir}",
        f"fusable_diag_components={cfg.fusable_diag_components}",
        f"calib_input_mode={cfg.calib_input_mode}",
        f"layer_rollback={cfg.layer_rollback}",
        f"loss_rollback={cfg.loss_rollback}",
        f"router_rollback={cfg.router_rollback}",
        f"router_align_type={ROUTER_ALIGN_TYPE}",
        f"router_align_temperature={ROUTER_ALIGN_TEMPERATURE}",
        f"router_align_loss_weight={cfg.router_align_loss_weight}",
        f"model_type={cfg.model_type}",
        f"num_layers={cfg.num_layers}",
        f"num_experts={cfg.num_experts}",
        f"top_k={cfg.top_k}",
        f"kv_cache_dtype={cfg.kv_cache_dtype}",
        f"native_has_online_rotation={cfg.native_has_online_rotation}",
    ]
    if extra:
        for k, v in extra.items():
            rows.append(f"{k}={v}")
    return rows


def output_path(cfg: E2ETrainConfig) -> Path:
    return Path(cfg.output_dir)
