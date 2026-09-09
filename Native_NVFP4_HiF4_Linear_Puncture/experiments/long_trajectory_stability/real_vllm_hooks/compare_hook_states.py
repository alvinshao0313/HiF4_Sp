#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--reference_root", required=True)
    p.add_argument("--variant_root", required=True)
    p.add_argument("--reference_variant", default="E0")
    p.add_argument("--variant", required=True)
    p.add_argument("--mode", default="forced_core")
    p.add_argument("--output", required=True)
    p.add_argument("--router_topk", type=int, default=8)
    return p.parse_args()


def load_manifest(root: Path, variant: str, mode: str) -> dict:
    path = root / f"{variant}_{mode}_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def tensor_metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    af = a.float().reshape(-1)
    bf = b.float().reshape(-1)
    if af.shape != bf.shape:
        raise RuntimeError(f"tensor shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    diff = bf - af
    anorm = torch.linalg.vector_norm(af)
    bnorm = torch.linalg.vector_norm(bf)
    dnorm = torch.linalg.vector_norm(diff)
    return {
        "rel_l2": float((dnorm / (anorm + 1e-12)).item()),
        "cosine": float(F.cosine_similarity(af, bf, dim=0, eps=1e-12).item()),
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "norm_ratio": float((bnorm / (anorm + 1e-12)).item()),
    }


def index_records(payload: dict) -> dict[tuple, dict]:
    out = {}
    for row in payload["records"]:
        key = (
            int(row["decode_index"]),
            row["boundary"],
            row["role"],
            row["layer"],
        )
        if key in out:
            raise RuntimeError(f"duplicate record {key}")
        out[key] = row
    return out


def router_metrics(a: torch.Tensor, b: torch.Tensor, topk: int) -> dict:
    af = a.float().reshape(-1)
    bf = b.float().reshape(-1)
    if af.shape != bf.shape:
        raise RuntimeError("router logits shape mismatch")
    pa = af.softmax(dim=-1)
    pb = bf.softmax(dim=-1)
    kl = torch.sum(pa * (torch.log(pa.clamp_min(1e-30)) - torch.log(pb.clamp_min(1e-30))))
    ia = torch.topk(af, k=topk + 1).indices
    ib = torch.topk(bf, k=topk + 1).indices
    set_a = {int(x) for x in ia[:topk].tolist()}
    set_b = {int(x) for x in ib[:topk].tolist()}
    margin_a = float((af[ia[topk - 1]] - af[ia[topk]]).item())
    margin_b = float((bf[ib[topk - 1]] - bf[ib[topk]]).item())
    return {
        "router_kl_e0_to_variant": float(kl.item()),
        "topk_exact": set_a == set_b,
        "topk_overlap": len(set_a & set_b) / topk,
        "topk_jaccard": len(set_a & set_b) / max(len(set_a | set_b), 1),
        "e0_boundary_margin": margin_a,
        "variant_boundary_margin": margin_b,
        "e0_topk": sorted(set_a),
        "variant_topk": sorted(set_b),
    }


def load_raw_logits(root: Path, variant: str, sample_key: str, decode_index: int) -> torch.Tensor:
    files = sorted((root / "raw_logits" / variant / sample_key).glob(f"rank*_decode{decode_index}.pt"))
    if not files:
        raise RuntimeError(f"raw logits missing: {variant}/{sample_key}/{decode_index}")
    captures = [torch.load(path, map_location="cpu", weights_only=False) for path in files]
    tensors = [row["logits"] for row in captures]
    for tensor in tensors[1:]:
        if not torch.equal(tensors[0], tensor):
            raise RuntimeError(f"raw logits differ across ranks for {variant}/{sample_key}/{decode_index}")
    return tensors[0].float()


def logits_metrics(a: torch.Tensor, b: torch.Tensor, target: int) -> dict:
    la = a.float().reshape(-1)
    lb = b.float().reshape(-1)
    logpa = F.log_softmax(la, dim=-1)
    logpb = F.log_softmax(lb, dim=-1)
    pa = logpa.exp()
    pb = logpb.exp()
    kl = torch.sum(pa * (logpa - logpb))
    m = 0.5 * (pa + pb)
    logm = torch.log(m.clamp_min(1e-30))
    js = 0.5 * torch.sum(pa * (logpa - logm)) + 0.5 * torch.sum(pb * (logpb - logm))
    top2a = torch.topk(la, 2)
    top2b = torch.topk(lb, 2)
    rank_b = int((lb > lb[target]).sum().item()) + 1
    return {
        "logit_kl_e0_to_variant": float(kl.item()),
        "logit_js": float(js.item()),
        "centered_cosine": float(
            F.cosine_similarity(la - la.mean(), lb - lb.mean(), dim=0, eps=1e-12).item()
        ),
        "e0_top1": int(top2a.indices[0].item()),
        "variant_top1": int(top2b.indices[0].item()),
        "top1_agree": int(top2a.indices[0]) == int(top2b.indices[0]),
        "e0_margin": float((top2a.values[0] - top2a.values[1]).item()),
        "variant_margin": float((top2b.values[0] - top2b.values[1]).item()),
        "target_token": int(target),
        "target_logit_delta": float((lb[target] - la[target]).item()),
        "target_rank_variant": rank_b,
        "target_nll_e0": float((-logpa[target]).item()),
        "target_nll_variant": float((-logpb[target]).item()),
    }


def main() -> None:
    args = parse_args()
    ref_root = Path(args.reference_root).resolve()
    var_root = Path(args.variant_root).resolve()
    ref_manifest = load_manifest(ref_root, args.reference_variant, args.mode)
    var_manifest = load_manifest(var_root, args.variant, args.mode)
    ref_samples = {row["sample_key"]: row for row in ref_manifest["samples"]}
    var_samples = {row["sample_key"]: row for row in var_manifest["samples"]}
    if set(ref_samples) != set(var_samples):
        raise RuntimeError("reference/variant sample sets differ")

    rows: list[dict] = []
    for sample_key in sorted(ref_samples):
        ref_sample = ref_samples[sample_key]
        var_sample = var_samples[sample_key]
        if ref_sample["probe_decode_indices"] != var_sample["probe_decode_indices"]:
            raise RuntimeError(f"probe list differs for {sample_key}")
        target_ids = ref_sample["expected_forced_ids"]
        if target_ids is None:
            raise RuntimeError("comparison requires forced-trajectory manifests")
        for rank in (0, 1):
            ref_payload = torch.load(
                ref_root / "hooks" / args.reference_variant / sample_key / f"rank{rank}.pt",
                map_location="cpu",
                weights_only=False,
            )
            var_payload = torch.load(
                var_root / "hooks" / args.variant / sample_key / f"rank{rank}.pt",
                map_location="cpu",
                weights_only=False,
            )
            iref = index_records(ref_payload)
            ivar = index_records(var_payload)
            if set(iref) != set(ivar):
                raise RuntimeError(f"hook record key mismatch for {sample_key}/rank{rank}")
            for key in sorted(iref, key=str):
                decode_index, boundary, role, layer = key
                metrics = tensor_metrics(iref[key]["tensor"], ivar[key]["tensor"])
                row = {
                    "sample_key": sample_key,
                    "decode_index": decode_index,
                    "tp_rank": rank,
                    "boundary": boundary,
                    "role": role,
                    "layer": layer,
                    **metrics,
                }
                if boundary == "router_logits" and role == "logits":
                    row.update(
                        router_metrics(
                            iref[key]["tensor"], ivar[key]["tensor"], args.router_topk
                        )
                    )
                rows.append(row)

        for decode_index in ref_sample["probe_decode_indices"]:
            e0_logits = load_raw_logits(ref_root, args.reference_variant, sample_key, decode_index)
            var_logits = load_raw_logits(var_root, args.variant, sample_key, decode_index)
            rows.append(
                {
                    "sample_key": sample_key,
                    "decode_index": int(decode_index),
                    "tp_rank": None,
                    "boundary": "raw_logits",
                    "role": "full_vocab",
                    "layer": None,
                    **logits_metrics(e0_logits, var_logits, int(target_ids[decode_index])),
                }
            )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(out)


if __name__ == "__main__":
    main()
