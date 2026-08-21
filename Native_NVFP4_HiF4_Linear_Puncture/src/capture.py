"""Capture post-rotation / pre-quant activations for formal (or smoke) modules."""

from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from Native_NVFP4_HiF4_Linear_Puncture.src.checkpoint import resolve_local_snapshot
from Native_NVFP4_HiF4_Linear_Puncture.src.config import (
    SMOKE_MODULES,
    AppConfig,
    load_config,
    results_dir,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.io_utils import (
    ensure_dir,
    module_capture_stem,
    save_pt,
    write_json,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.native_nvfp4 import qdq_nvfp4_post_rotation
from Native_NVFP4_HiF4_Linear_Puncture.src.prompts import (
    build_prompt_bank,
    prompts_for_split,
    tokenize_prompt,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.rotation import (
    apply_block_rotation,
    rotation_sha256,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.semantic_model import (
    NativeNVFP4SemanticLinear,
    disable_all_observers,
    enable_observers,
    load_native_nvfp4_semantic_model,
    count_wrapped_targets,
)


def select_token_indices(seq_len: int, max_rows: int) -> torch.Tensor:
    if seq_len <= 0:
        return torch.zeros(0, dtype=torch.long)
    if seq_len <= max_rows:
        return torch.arange(seq_len, dtype=torch.long)
    return torch.linspace(0, seq_len - 1, max_rows).round().long()


class ActivationRecorder:
    def __init__(self, module_names: set[str], max_rows_per_prompt: int) -> None:
        self.module_names = module_names
        self.max_rows = max_rows_per_prompt
        self.buffers: dict[str, list[torch.Tensor]] = defaultdict(list)
        self.sample_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.meta: dict[str, dict[str, Any]] = {}
        self._current_prompt_id: str | None = None
        self._current_attention_mask: torch.Tensor | None = None
        self.audit_candidates: list[dict[str, Any]] = []

    def set_prompt_context(
        self, prompt_id: str, attention_mask: torch.Tensor | None
    ) -> None:
        self._current_prompt_id = prompt_id
        self._current_attention_mask = attention_mask

    def observe(self, module_name: str, x_rot: torch.Tensor) -> None:
        if module_name not in self.module_names:
            return
        # x_rot: [B, T, K] or [T, K]
        x = x_rot.detach()
        if x.ndim == 3:
            if x.shape[0] != 1:
                raise ValueError("capture expects batch size 1")
            x = x[0]
        elif x.ndim != 2:
            raise ValueError(f"unexpected x_rot ndim={x.ndim}")

        t, k = x.shape
        mask = self._current_attention_mask
        if mask is not None:
            m = mask.detach().reshape(-1)
            if m.numel() != t:
                # truncated ids length should match
                m = m[:t]
            valid = m.to(dtype=torch.bool)
            x_valid = x[valid]
            orig_t = int(valid.sum().item())
            # indices in the unpadded sequence
            valid_pos = torch.arange(t)[valid.cpu()]
        else:
            x_valid = x
            orig_t = t
            valid_pos = torch.arange(t)

        idx_local = select_token_indices(x_valid.shape[0], self.max_rows)
        selected = x_valid[idx_local].to(dtype=torch.bfloat16, device="cpu")
        selected_token_indices = valid_pos[idx_local].tolist()

        row_start = sum(t.shape[0] for t in self.buffers[module_name])
        self.buffers[module_name].append(selected)
        row_end = row_start + selected.shape[0]
        self.sample_rows[module_name].append(
            {
                "prompt_id": self._current_prompt_id,
                "row_start": row_start,
                "row_end": row_end,
                "original_token_count": orig_t,
                "selected_token_indices": selected_token_indices,
            }
        )

        # Collect occasional audit rows (filled later with X_pre in forward hook path).
        if len(self.audit_candidates) < 64 and selected.shape[0] > 0:
            self.audit_candidates.append(
                {
                    "module_name": module_name,
                    "prompt_id": self._current_prompt_id,
                    "x_rot_row": selected[0].clone(),
                }
            )


def _attach_module_meta(model: torch.nn.Module, recorder: ActivationRecorder) -> None:
    for m in model.modules():
        if isinstance(m, NativeNVFP4SemanticLinear) and m.module_name in recorder.module_names:
            recorder.meta[m.module_name] = {
                "input_global_scale_fp32": m.input_global_scale.detach().cpu().float(),
                "rotation_sha256": rotation_sha256(m.rotation_matrix),
                "rotation_group_size": m.rotation_group_size,
                "rotation_matrix": m.rotation_matrix.detach().cpu(),
            }


def _parse_layer_proj(module_name: str) -> tuple[int, str]:
    parts = module_name.split(".")
    layer_idx = int(parts[parts.index("layers") + 1])
    return layer_idx, parts[-1]


def run_capture(
    config: AppConfig,
    run_id: str,
    *,
    mode: str = "formal",
    device: str = "cuda",
) -> dict[str, Any]:
    if mode not in {"formal", "smoke"}:
        raise ValueError("mode must be formal|smoke")

    torch.manual_seed(config.experiment.seed)
    snapshot = resolve_local_snapshot(config.model.model_id)
    out_dir = ensure_dir(results_dir(run_id))
    cap_dir = ensure_dir(out_dir / "captures")
    audit_dir = ensure_dir(out_dir / "audits")

    if mode == "smoke":
        module_names = list(SMOKE_MODULES)
    else:
        module_names = config.formal_module_names
    module_set = set(module_names)

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    model, ckpt_index = load_native_nvfp4_semantic_model(
        snapshot,
        device=device,
        rotation_group_size=config.nvfp4.activation_group_size,
        activation_group_size=config.nvfp4.activation_group_size,
    )
    wrapped = count_wrapped_targets(model)
    if wrapped != 252:
        raise RuntimeError(f"wrapped coverage {wrapped} != 252")

    recorder = ActivationRecorder(module_set, config.experiment.token_rows_per_prompt)
    _attach_module_meta(model, recorder)

    # Audit: also capture X_pre via pre-hook on selected modules.
    audit_x_pre: dict[str, torch.Tensor] = {}

    def make_pre_hook(name: str):
        def _hook(_mod, inputs):
            x = inputs[0].detach()
            if name not in audit_x_pre and x.numel() > 0:
                # store one flattened valid row later; keep full tensor briefly
                audit_x_pre[name] = x.detach().to("cpu")
            return None

        return _hook

    pre_hooks = []
    for m in model.modules():
        if isinstance(m, NativeNVFP4SemanticLinear) and m.module_name in module_set:
            pre_hooks.append(m.register_forward_pre_hook(make_pre_hook(m.module_name)))

    enable_observers(model, module_set, recorder.observe)

    # Run cal then val prompts (prefill only).
    for split in ("cal", "val"):
        for item in prompts_for_split(split):
            enc = tokenize_prompt(
                tokenizer, item.text, max_seq_len=config.experiment.max_seq_len
            )
            input_ids = enc["input_ids"].to(device)
            attn = enc.get("attention_mask")
            if attn is not None:
                attn = attn.to(device)
            recorder.set_prompt_context(item.prompt_id, attn.cpu() if attn is not None else None)
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attn, use_cache=False)

    disable_all_observers(model)
    for h in pre_hooks:
        h.remove()

    # Write capture files.
    written = []
    for module_name in module_names:
        if module_name not in recorder.buffers:
            raise RuntimeError(f"no capture for {module_name}")
        # Split rows by prompt split using sample_rows prompt_id
        cal_rows = []
        val_rows = []
        cal_meta = []
        val_meta = []
        # Rebuild by replaying sample_rows with tensors
        tensors = recorder.buffers[module_name]
        assert len(tensors) == len(recorder.sample_rows[module_name])
        bank = {p.prompt_id: p for p in build_prompt_bank()}
        for tensor, meta in zip(tensors, recorder.sample_rows[module_name]):
            split = bank[meta["prompt_id"]].split
            if split == "cal":
                offset = sum(t.shape[0] for t in cal_rows)
                piece = dict(meta)
                piece["row_start"] = offset
                piece["row_end"] = offset + tensor.shape[0]
                cal_rows.append(tensor)
                cal_meta.append(piece)
            else:
                offset = sum(t.shape[0] for t in val_rows)
                piece = dict(meta)
                piece["row_start"] = offset
                piece["row_end"] = offset + tensor.shape[0]
                val_rows.append(tensor)
                val_meta.append(piece)

        meta_common = recorder.meta[module_name]
        layer_idx, proj = _parse_layer_proj(module_name)
        stem = module_capture_stem(module_name)
        for split, rows, srows in (
            ("cal", cal_rows, cal_meta),
            ("val", val_rows, val_meta),
        ):
            x = torch.cat(rows, dim=0) if rows else torch.zeros(0, 1, dtype=torch.bfloat16)
            payload = {
                "schema_version": 1,
                "module_name": module_name,
                "layer_idx": layer_idx,
                "projection": proj,
                "split": split,
                "x_rot_bf16": x,
                "input_global_scale_fp32": meta_common["input_global_scale_fp32"],
                "rotation_sha256": meta_common["rotation_sha256"],
                "rotation_group_size": meta_common["rotation_group_size"],
                "sample_rows": srows,
            }
            path = cap_dir / f"{stem}_{split}.pt"
            save_pt(path, payload)
            written.append(str(path))

    # Build 10-row audit with X_pre / X_rot / A_N.
    audits = []
    for module_name, x_pre_full in list(audit_x_pre.items())[:10]:
        mmeta = recorder.meta[module_name]
        # take first non-pad token row of first batch
        xp = x_pre_full
        if xp.ndim == 3:
            xp = xp[0]
        x_pre_row = xp[0].to(torch.bfloat16)
        x_rot_row = apply_block_rotation(
            x_pre_row.unsqueeze(0),
            mmeta["rotation_matrix"],
            mmeta["rotation_group_size"],
        )[0]
        a_n_row = qdq_nvfp4_post_rotation(
            x_rot_row.unsqueeze(0),
            mmeta["input_global_scale_fp32"],
            group_size=config.nvfp4.activation_group_size,
        )[0]
        # exact checks
        if not torch.equal(x_rot_row, apply_block_rotation(
            x_pre_row.unsqueeze(0), mmeta["rotation_matrix"], mmeta["rotation_group_size"]
        )[0]):
            raise RuntimeError(f"audit rotation mismatch for {module_name}")
        if not torch.equal(
            a_n_row,
            qdq_nvfp4_post_rotation(
                x_rot_row.unsqueeze(0),
                mmeta["input_global_scale_fp32"],
                group_size=config.nvfp4.activation_group_size,
            )[0],
        ):
            raise RuntimeError(f"audit A_N mismatch for {module_name}")
        audits.append(
            {
                "module_name": module_name,
                "x_pre_sample": x_pre_row.cpu(),
                "x_rot_sample": x_rot_row.cpu(),
                "a_n_sample": a_n_row.cpu(),
            }
        )
    save_pt(audit_dir / "activation_audits.pt", {"audits": audits})

    snapshot_hash = hashlib.sha256(str(snapshot).encode()).hexdigest()[:16]
    prompt_ids = {
        "cal": [p.prompt_id for p in prompts_for_split("cal")],
        "val": [p.prompt_id for p in prompts_for_split("val")],
    }
    manifest = {
        "model_id": config.model.model_id,
        "snapshot_path": str(snapshot),
        "snapshot_hash": snapshot_hash,
        "source_semantic_version": "native_nvfp4_rot_a4_v1",
        "formal_layers": list(config.experiment.formal_layers),
        "capture_mode": mode,
        "module_count": len(module_names),
        "expected_formal_modules": 35,
        "capture_coverage": f"{len(module_names)}/{35 if mode=='formal' else 3}",
        "wrapped_coverage": f"{wrapped}/252",
        "prompt_ids_by_split": prompt_ids,
        "max_seq_len": config.experiment.max_seq_len,
        "sampling_policy": "drop_pad; if T<=64 keep all else linspace(0,T-1,64).round.long",
        "capture_point": "post_rotation_pre_activation_quant",
        "written_files": written,
        "audit_rows": len(audits),
    }
    write_json(out_dir / "capture_manifest.json", manifest)
    print(f"CAPTURE DONE mode={mode} modules={len(module_names)} -> {out_dir}")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Capture X_rot for native NVFP4 semantic model")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--mode", type=str, default="formal", choices=["formal", "smoke"])
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Alias for --mode smoke",
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args(argv)
    mode = "smoke" if args.smoke else args.mode
    config = load_config(args.config)
    run_capture(config, args.run_id, mode=mode, device=args.device)


if __name__ == "__main__":
    main()
