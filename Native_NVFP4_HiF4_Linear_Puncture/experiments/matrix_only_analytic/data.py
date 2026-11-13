"""CPU layer-0 q_proj inputs from native embeddings and fixed native RMSNorm."""
from __future__ import annotations

import json

import torch
from safetensors import safe_open

from ..matrix_norm_efficiency.config import NATIVE, REPO
from ..matrix_norm_efficiency.prepare import samples_for
from ..e2e_diag_reconstruction.core.modelopt_moe_checkpoint import _decode_weight
from ..e2e_diag_reconstruction.core.moe_semantic_hif4 import _rms_norm, native_nvfp4_linear
from ..e2e_diag_reconstruction.core.moe_transforms import apply_r64_no_cross_head
from ..non_equivalent_reconstruction.artifact import sha256
from ..reasoning_trace_calibration.artifacts import tensor_digest
from ...src.formats import qdq_hif4_direct
from .solver import ProjectionData


SOURCE_RUN = REPO / "Native_NVFP4_HiF4_Linear_Puncture/results/matrix_norm_efficiency/run_20261029T023839Z"


@torch.no_grad()
def load_cpu_projection():
    """Sample token rows, not truncated forward contexts (valid only at layer 0).

    Before the first attention operation, embedding plus RMSNorm is token-local.
    This construction must not be reused for deeper/context-dependent inputs.
    """
    training, validation, calibration_hashes = samples_for(SOURCE_RUN, "full")
    config = json.loads((NATIVE / "config.json").read_text())
    weight_map = json.loads((NATIVE / "model.safetensors.index.json").read_text())["weight_map"]
    tensor_hashes = {}

    def read(key):
        with safe_open(NATIVE / weight_map[key], framework="pt", device="cpu") as handle:
            value = handle.get_tensor(key)
        tensor_hashes[key] = tensor_digest(value.reshape(-1))
        return value

    prefix = "model.layers.0.self_attn"
    weight_global = torch.stack([read(f"{prefix}.{p}.weight_scale_2").reshape(())
                                 for p in ("q_proj", "k_proj", "v_proj")]).max()
    input_global = torch.stack([read(f"{prefix}.{p}.input_scale").reshape(())
                                for p in ("q_proj", "k_proj", "v_proj")]).max()
    weight = _decode_weight(read(f"{prefix}.q_proj.weight"), read(f"{prefix}.q_proj.weight_scale"),
                            weight_global, device=torch.device("cpu"))
    norm = read("model.layers.0.input_layernorm.weight")
    if weight.shape != (4096, 2048) or norm.shape != (2048,):
        raise ValueError("unexpected native layer-0 q_proj dimensions")
    if not torch.isfinite(input_global) or input_global <= 0:
        raise ValueError("invalid native QKV input scale")
    input_global_inv = 1. / input_global

    inputs, selections = {}, {}
    embedding_key = "model.embed_tokens.weight"
    with safe_open(NATIVE / weight_map[embedding_key], framework="pt", device="cpu") as handle:
        embedding = handle.get_tensor(embedding_key)
        for split, samples in (("train", training), ("validation", validation)):
            chosen = sorted(samples, key=lambda s: s.input_ids.numel(), reverse=True)[:4]
            rows, selections[split] = [], []
            for sample in chosen:
                valid = sample.loss_mask.bool().nonzero().flatten()
                if valid.numel() < 64:
                    raise ValueError("CPU diagnostic requires 64 valid tokens per sequence")
                positions = valid[torch.linspace(0, valid.numel() - 1, 64).round().long()]
                ids = sample.input_ids[positions].long()
                rows.append(embedding.index_select(0, ids))
                selections[split].append({"sample_id": sample.sample_id, "domain": sample.meta["domain"],
                                          "sequence_tokens": sample.input_ids.numel(),
                                          "positions": positions.tolist(), "token_ids": ids.tolist()})
            embeddings = torch.cat(rows).to(torch.bfloat16)
            tensor_hashes[f"selected_embeddings.{split}"] = tensor_digest(embeddings)
            x = _rms_norm(embeddings, norm, float(config["rms_norm_eps"]))
            target = native_nvfp4_linear(x, weight, input_global_inv)
            activation = qdq_hif4_direct(apply_r64_no_cross_head(x.float()), output_dtype=torch.bfloat16)
            inputs[split] = ProjectionData(activation, target)

    frozen = {"weight": weight, "norm": norm, "input_global_scale_inv": input_global_inv.reshape(1),
              **{f"{split}.{key}": getattr(data, key) for split, data in inputs.items()
                 for key in ("activation", "target")}}
    manifest = {"native_snapshot": str(NATIVE), "native_config_sha256": sha256(NATIVE / "config.json"),
                "native_index_sha256": sha256(NATIVE / "model.safetensors.index.json"),
                "source_run": str(SOURCE_RUN), "calibration_sha256": calibration_hashes,
                "source_tensor_hashes": tensor_hashes, "selection": selections,
                "selection_rule": "longest 4 sequences per split; 64 evenly spaced valid token positions each",
                "input_semantics": "exact token-local native embedding + fixed native RMSNorm at layer 0",
                "teacher": "native NVFP4 activation QDQ and decoded QKV-max-scale q_proj; BF16 Linear output",
                "student": "fixed R64 activation + HiF4 G64 W4A4 QDQ; BF16 Linear output",
                "scope": "CPU local Linear diagnostic, not full-context decoder or downstream evaluation"}
    return weight, inputs["train"], inputs["validation"], frozen, manifest
