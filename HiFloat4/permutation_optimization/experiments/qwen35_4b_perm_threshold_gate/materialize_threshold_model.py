#!/usr/bin/env python3
"""Materialize a BF16 model variant from a permutation map.

Applies the map offline (gate/up row perm + down column perm), verifies map
completeness and per-layer FP32 MLP equivalence for every non-identity layer,
then saves the model + tokenizer. Refuses to save on any check failure.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
HIFLOAT4_ROOT = SCRIPT_DIR.parents[2]
if str(HIFLOAT4_ROOT) not in sys.path:
    sys.path.insert(0, str(HIFLOAT4_ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from permutation_optimization.model_permutation import (  # noqa: E402
    apply_permutations_from_dict,
    discover_swiglu_mlps,
    get_mlp_modules,
    validate_permutation,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--permutations", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--dtype", type=str, default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--expect-mlps", type=int, default=32)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--metadata", type=str, default="{}")
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    permutations = torch.load(args.permutations, map_location="cpu", weights_only=True)
    if not isinstance(permutations, dict) or not permutations:
        raise ValueError("permutations must be a non-empty dict")

    print(f"[materialize] loading {args.model} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        device_map=None,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=False, trust_remote_code=args.trust_remote_code
    )

    specs = discover_swiglu_mlps(model)
    if len(specs) != args.expect_mlps:
        raise RuntimeError(f"expected {args.expect_mlps} SwiGLU MLPs, found {len(specs)}")
    spec_names = {s.name for s in specs}
    map_names = set(permutations)
    if map_names != spec_names:
        raise RuntimeError(
            f"permutation map mismatch: missing={sorted(spec_names - map_names)} "
            f"extra={sorted(map_names - spec_names)}"
        )
    for spec in specs:
        validate_permutation(permutations[spec.name], spec.intermediate_size)

    non_identity = [
        name
        for name, perm in permutations.items()
        if not torch.equal(perm.to(torch.long), torch.arange(perm.numel()))
    ]
    print(f"[materialize] non-identity layers: {len(non_identity)}", flush=True)

    # Snapshot FP32 MLP references before permutation (per non-identity layer).
    refs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for spec in specs:
        if spec.name not in non_identity:
            continue
        gate, up, down = get_mlp_modules(model, spec)
        d_model = gate.weight.shape[1]
        g = torch.Generator().manual_seed(20260731 + spec.layer_index)
        x = torch.randn(4, d_model, generator=g, dtype=torch.float32)
        with torch.no_grad():
            y = (
                torch.nn.functional.silu(gate.weight.float() @ x.T).T
                * (up.weight.float() @ x.T).T
            ) @ down.weight.float().T
        refs[spec.name] = (x, y)

    apply_permutations_from_dict(model, permutations, specs)

    for spec in specs:
        if spec.name not in refs:
            continue
        gate, up, down = get_mlp_modules(model, spec)
        x, y_before = refs[spec.name]
        with torch.no_grad():
            y_after = (
                torch.nn.functional.silu(gate.weight.float() @ x.T).T
                * (up.weight.float() @ x.T).T
            ) @ down.weight.float().T
        if not torch.allclose(y_before, y_after, rtol=1e-5, atol=1e-5):
            raise RuntimeError(
                f"FP32 MLP equivalence failed for {spec.name}; refusing to save"
            )
    print("[materialize] FP32 equivalence checks passed", flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(out_dir)
    metadata = json.loads(args.metadata) if args.metadata else {}
    metadata.update(
        {
            "source_model": args.model,
            "permutations_file": str(args.permutations),
            "n_non_identity_layers": len(non_identity),
            "non_identity_layers": sorted(non_identity),
        }
    )
    (out_dir / "materialize_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"[materialize] saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
