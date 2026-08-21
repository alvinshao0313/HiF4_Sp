"""Unified AX1–AX4 shard pipeline for activation incremental experiments."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn

from Inference_Paradigm_Conversion.ipc_analysis.analysis.activation_grid_occupancy import (
    analyze_grid_occupancy_row,
    build_theoretical_grid_json,
)
from Inference_Paradigm_Conversion.ipc_analysis.analysis.activation_group_size_ablation import (
    run_dispersion_sweep,
    run_group_size_ablation,
)
from Inference_Paradigm_Conversion.ipc_analysis.analysis.activation_incremental_io import (
    assert_split_isolation,
    build_incremental_input,
)
from Inference_Paradigm_Conversion.ipc_analysis.analysis.activation_s0_divisor_search import (
    candidate_alphas,
    search_output_aware_group_alphas,
    search_s0_divisor_oracle,
)
from Inference_Paradigm_Conversion.ipc_analysis.analysis.activation_scale_payload_factorization import (
    run_cross_format_factorization,
)
from Inference_Paradigm_Conversion.ipc_analysis.capture.activation_capture import (
    capture_linear_inputs,
)
from Inference_Paradigm_Conversion.ipc_analysis.capture.model_adapter import (
    load_source_model_for_capture,
)
from Inference_Paradigm_Conversion.ipc_analysis.capture.prompts import (
    discovery_items,
    validation_items,
)
from Inference_Paradigm_Conversion.ipc_analysis.config import (
    LINEAR_PROJECTIONS,
    resolve_representative_layers,
)
from Inference_Paradigm_Conversion.ipc_analysis.formats.fingerprint import (
    load_nvfp4_qat_dequant_weight,
)
from Inference_Paradigm_Conversion.ipc_analysis.formats.nvfp4 import (
    load_nvfp4_activation_scales,
    resolve_activation_scale_path,
    resolve_nvfp4_scale_for_module,
)
from Inference_Paradigm_Conversion.ipc_analysis.io_utils import (
    GzipJsonlWriter,
    atomic_write_json,
    ensure_dir,
    write_csv,
)

_PROJ_RE = re.compile(
    r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
)


def _proj_of(name: str) -> str | None:
    m = _PROJ_RE.search(name)
    return m.group(1) if m else None


def _layer_idx(name: str) -> int | None:
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


@torch.no_grad()
def run_ax_shard(
    checkpoint: Path,
    out_dir: Path,
    *,
    device: str = "cuda:0",
    shard_id: int = 0,
    num_shards: int = 1,
    split: Literal["discovery", "validation"] = "discovery",
    samples_per_family: int = 8,
    max_seq_len: int = 128,
    decode_steps: int = 4,
    max_tokens_prefill: int = 32,
    phases: tuple[str, ...] = ("prefill",),
    run_ax1: bool = True,
    run_ax2: bool = True,
    run_ax3: bool = True,
    run_ax4: bool = True,
    dispersion_d: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0),
    a2_run_dir: Path | None = None,
) -> dict[str, Any]:
    out_dir = ensure_dir(out_dir)
    torch.set_num_threads(4)
    device_t = torch.device(device)
    if device_t.type == "cuda":
        torch.cuda.set_device(device_t)

    with (checkpoint / "config.json").open("r", encoding="utf-8") as f:
        num_layers = int(json.load(f)["num_hidden_layers"])
    rep_layers = set(resolve_representative_layers(num_layers))
    bank = discovery_items(samples_per_family) if split == "discovery" else validation_items(samples_per_family)
    assert_split_isolation(split, bank)
    prompts = [p for i, p in enumerate(bank) if i % num_shards == shard_id]

    model, tok = load_source_model_for_capture(checkpoint, device=device_t)
    scales = load_nvfp4_activation_scales(resolve_activation_scale_path(checkpoint))
    weight_cache: dict[str, torch.Tensor] = {}

    def module_filter(name: str, _mod: nn.Module) -> bool:
        li = _layer_idx(name)
        if li not in rep_layers:
            return False
        return _proj_of(name) in LINEAR_PROJECTIONS

    ax1_rows: list[dict[str, Any]] = []
    ax2_gs_rows: list[dict[str, Any]] = []
    ax2_disp_rows: list[dict[str, Any]] = []
    ax3_rows: list[dict[str, Any]] = []
    ax3_scale_rows: list[dict[str, Any]] = []
    ax4_rows: list[dict[str, Any]] = []
    oracle_jsonl = out_dir / f"ax1_s0_output_oracle_samples_shard{shard_id}.jsonl.gz"

    with GzipJsonlWriter(oracle_jsonl, flush_every=32) as oracle_writer:
        for item in prompts:
            enc = tok(item.text, return_tensors="pt", truncation=True, max_length=max_seq_len)
            batch = {k: v.to(device_t) for k, v in enc.items()}
            if "attention_mask" not in batch:
                batch["attention_mask"] = torch.ones_like(batch["input_ids"])

            caps: list = []
            if "prefill" in phases:
                caps.extend(
                    capture_linear_inputs(
                        model,
                        batch,
                        phase="prefill",
                        module_filter=module_filter,
                        sample_id=item.sample_id,
                        max_tokens_per_module=2048,
                        max_raw_tokens_per_module=max_tokens_prefill,
                    )
                )

            if "decode" in phases:
                warm = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=True,
                )
                past = warm.past_key_values
                next_id = warm.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                input_ids = torch.cat([batch["input_ids"], next_id], dim=-1)
                attn = torch.cat(
                    [
                        batch["attention_mask"],
                        torch.ones((1, 1), device=device_t, dtype=batch["attention_mask"].dtype),
                    ],
                    dim=-1,
                )
                for step in range(decode_steps):
                    step_batch = {
                        "input_ids": input_ids[:, -1:],
                        "attention_mask": attn,
                        "past_key_values": past,
                        "use_cache": True,
                    }
                    if step == decode_steps - 1:
                        step_caps, out = capture_linear_inputs(
                            model,
                            step_batch,
                            phase="decode",
                            module_filter=module_filter,
                            sample_id=f"{item.sample_id}_d{step}",
                            max_tokens_per_module=64,
                            max_raw_tokens_per_module=64,
                            return_outputs=True,
                        )
                        caps.extend(step_caps)
                    else:
                        out = model(**step_batch)
                    past = out.past_key_values
                    next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    input_ids = torch.cat([input_ids, next_id], dim=-1)
                    attn = torch.cat(
                        [attn, torch.ones((1, 1), device=device_t, dtype=attn.dtype)],
                        dim=-1,
                    )

            for cap in caps:
                if cap.phase not in phases:
                    continue
                proj = _proj_of(cap.module_name)
                li = cap.layer_idx if cap.layer_idx is not None else _layer_idx(cap.module_name)
                if proj is None or li is None:
                    continue
                x = cap.extras.get("stat_sample")
                if x is None or x.numel() == 0:
                    x = cap.tensor
                if x.numel() == 0:
                    continue
                x = x.to(device=device_t, dtype=torch.bfloat16)
                k = x.shape[-1]
                usable = k - (k % 64)
                if usable < 64:
                    continue
                if x.ndim == 2 and x.shape[0] > max_tokens_prefill:
                    x = x[:max_tokens_prefill]
                elif x.ndim == 3 and x.shape[1] > max_tokens_prefill:
                    x = x[:, :max_tokens_prefill]
                x = x[..., :usable]

                scale = resolve_nvfp4_scale_for_module(scales, cap.module_name).to(device_t)
                wname = cap.module_name + ".weight"
                if wname not in weight_cache:
                    w = load_nvfp4_qat_dequant_weight(checkpoint, wname, device=device_t).dequantized
                    if w.shape[1] != usable:
                        w = w[:, :usable]
                    weight_cache[wname] = w.detach()
                w_n = weight_cache[wname]

                inc = build_incremental_input(
                    run_id=out_dir.name,
                    sample_id=item.sample_id,
                    phase=cap.phase,
                    layer_idx=li,
                    module_name=cap.module_name,
                    projection=proj,
                    prompt_family=item.family,
                    split=split,
                    x_bf16=x,
                    input_global_scale=scale,
                    weight_fp32=w_n,
                )
                base = {
                    "sample_id": item.sample_id,
                    "prompt_family": item.family,
                    "split": split,
                    "layer_idx": li,
                    "module_name": cap.module_name,
                    "projection": proj,
                    "phase": cap.phase,
                }

                if run_ax1:
                    ax1 = search_s0_divisor_oracle(
                        inc.x_bf16, inc.a_nvfp4, inc.weight_fp32, alpha_chunk=8
                    )
                    ax1_row = {**base, **{k: v for k, v in ax1.items() if not isinstance(v, dict)}}
                    ax1_rows.append(ax1_row)
                    alphas = candidate_alphas()
                    samples = search_output_aware_group_alphas(
                        inc.x_bf16,
                        inc.a_nvfp4,
                        inc.weight_fp32,
                        alphas,
                        top_k=256,
                        random_k=256,
                        energy_k=256,
                    )
                    for s in samples:
                        oracle_writer.write({**base, **s})

                if run_ax2:
                    for r in run_group_size_ablation(inc.x_bf16, inc.a_nvfp4, inc.weight_fp32):
                        ax2_gs_rows.append({**base, **r})
                    for r in run_dispersion_sweep(
                        inc.x_bf16, inc.a_nvfp4, inc.weight_fp32, dispersion_d
                    ):
                        ax2_disp_rows.append({**base, **r})

                if run_ax3:
                    ax3 = analyze_grid_occupancy_row(
                        inc.x_bf16,
                        inc.a_nvfp4,
                        inc.a_hif4,
                        inc.weight_fp32,
                        inc.nvfp4_metadata,
                        inc.hif4_metadata,
                        alpha_oracle=float(ax1_rows[-1]["alpha_oracle_nvfp4"]) if ax1_rows else None,
                    )
                    flat_ax3 = {**base}
                    for k, v in ax3.items():
                        if isinstance(v, dict):
                            for sk, sv in v.items():
                                flat_ax3[f"{k}.{sk}"] = sv
                        else:
                            flat_ax3[k] = v
                    ax3_rows.append(flat_ax3)
                    from Inference_Paradigm_Conversion.ipc_analysis.analysis.activation_grid_occupancy import (
                        compare_local_scales,
                    )

                    ax3_scale_rows.append({**base, **compare_local_scales(inc.nvfp4_metadata, inc.hif4_metadata)})

                if run_ax4:
                    for r in run_cross_format_factorization(
                        inc.x_bf16, inc.a_nvfp4, inc.weight_fp32, scale
                    ):
                        ax4_rows.append({**base, **r})

            print(f"[AX] shard{shard_id} {item.sample_id}", flush=True)

    if shard_id == 0:
        atomic_write_json(
            out_dir / "ax3_theoretical_grid.json",
            build_theoretical_grid_json(out_dir),
        )

    write_csv(out_dir / f"ax1_s0_divisor_oracle_shard{shard_id}.csv", ax1_rows)
    write_csv(out_dir / f"ax2_group_size_ablation_shard{shard_id}.csv", ax2_gs_rows)
    write_csv(out_dir / f"ax2_sub16_dispersion_shard{shard_id}.csv", ax2_disp_rows)
    write_csv(out_dir / f"ax3_grid_occupancy_shard{shard_id}.csv", ax3_rows)
    write_csv(out_dir / f"ax3_local_scale_distribution_shard{shard_id}.csv", ax3_scale_rows)
    write_csv(out_dir / f"ax4_cross_format_factorization_shard{shard_id}.csv", ax4_rows)

    summary = {
        "shard_id": shard_id,
        "split": split,
        "phases": list(phases),
        "num_prompts": len(prompts),
        "ax1_rows": len(ax1_rows),
        "ax2_gs_rows": len(ax2_gs_rows),
        "ax2_disp_rows": len(ax2_disp_rows),
        "ax3_rows": len(ax3_rows),
        "ax4_rows": len(ax4_rows),
        "a2_run_dir": str(a2_run_dir) if a2_run_dir else None,
    }
    atomic_write_json(out_dir / f"ax_shard_summary_shard{shard_id}.json", summary)
    del model
    if device_t.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def merge_ax_shards(
    run_dir: Path,
    *,
    a2_run_dir: Path | None = None,
    run_ax5_rules: bool = True,
) -> dict[str, Any]:
    """Merge shard CSVs, run AX5 ranking + optional AX5-R."""
    import csv

    from Inference_Paradigm_Conversion.ipc_analysis.analysis.activation_rule_selection import (
        build_root_cause_ranking,
        run_rule_selection,
    )
    from Inference_Paradigm_Conversion.ipc_analysis.reporting.activation_incremental_report import (
        build_activation_incremental_report,
    )

    def _merge(pattern: str, out_name: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for p in sorted(run_dir.glob(pattern)):
            rows.extend(csv.DictReader(p.open(encoding="utf-8")))
        write_csv(run_dir / out_name, rows)
        return rows

    ax1 = _merge("ax1_s0_divisor_oracle_shard*.csv", "ax1_s0_divisor_oracle.csv")
    ax2 = _merge("ax2_group_size_ablation_shard*.csv", "ax2_group_size_ablation.csv")
    _merge("ax2_sub16_dispersion_shard*.csv", "ax2_sub16_dispersion.csv")
    ax3 = _merge("ax3_grid_occupancy_shard*.csv", "ax3_grid_occupancy.csv")
    _merge("ax3_local_scale_distribution_shard*.csv", "ax3_local_scale_distribution.csv")
    ax4 = _merge("ax4_cross_format_factorization_shard*.csv", "ax4_cross_format_factorization.csv")

    a2_rows: list[dict[str, Any]] = []
    a2_path = (a2_run_dir or run_dir) / "a2_variants.csv"
    if a2_path.is_file():
        a2_rows = list(csv.DictReader(a2_path.open(encoding="utf-8")))

    ranking = build_root_cause_ranking(
        {"ax1": ax1, "ax2": ax2, "ax3": ax3, "ax4": ax4},
        a2_csv_rows=a2_rows,
    )
    write_csv(run_dir / "ax5_root_cause_ranking.csv", ranking)

    rule_result: dict[str, Any] = {"status": "skipped"}
    if run_ax5_rules:
        disc = [r for r in ax1 if r.get("split") == "discovery"]
        rule_result = run_rule_selection(disc if disc else ax1)
        write_csv(run_dir / "ax5_rule_validation.csv", [rule_result])

    report = build_activation_incremental_report(run_dir)
    summary = {
        "run_id": run_dir.name,
        "ranking_count": len(ranking),
        "rule_selection": rule_result,
        "report": report,
    }
    atomic_write_json(run_dir / "activation_incremental_summary.json", summary)
    return summary
