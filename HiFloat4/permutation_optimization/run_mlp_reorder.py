"""CLI: search HiF4-friendly MLP intermediate permutations and optionally save a reordered model.

Intended future hook in HiFloat4/main.py (NOT wired in this change):
  after AutoModelForCausalLM.from_pretrained(...), before hif4_rtn_quant / gptq_fwrd,
  call reorder_model_mlps(...) or apply_permutations_from_file(...).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_HIFLOAT4 = Path(__file__).resolve().parents[1]
if str(_HIFLOAT4) not in sys.path:
    sys.path.insert(0, str(_HIFLOAT4))
_HIF4GPTQ = _HIFLOAT4 / "hif4gptq"
if str(_HIF4GPTQ) not in sys.path:
    sys.path.insert(0, str(_HIF4GPTQ))

from permutation_optimization.config import SearchConfig
from permutation_optimization.model_permutation import (
    discover_swiglu_mlps,
    get_mlp_modules,
)
from permutation_optimization.pipeline import reorder_model_mlps

logger = logging.getLogger("run_mlp_reorder")


def _parse_layers(spec: str, n_layers: int) -> list[int] | None:
    """Return list of layer indices to keep, or None for all."""
    if spec.strip().lower() in {"all", "*"}:
        return None
    if ":" in spec:
        start_s, end_s = spec.split(":", 1)
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else n_layers
        return list(range(start, end))
    return [int(x) for x in spec.split(",") if x.strip()]


_S1K_HF_ID = "simplescaling/s1K-1.1_tokenized"


def _ensure_output_dir(out_dir: Path, overwrite: bool) -> None:
    """Create ``out_dir``; refuse to reuse a non-empty dir unless ``overwrite``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"output dir {out_dir} is not empty; pass --overwrite-output to reuse it"
        )


def _s1k_calib_batches(
    model_name: str,
    nsamples: int,
    seqlen: int,
    seed: int,
):
    """Load s1K-1.1_tokenized samples.

    ``seqlen=0``: keep full text (no truncate). ``seqlen>0``: refuse if any
    sample is longer (no silent truncation). Each batch carries its source
    dataset index under ``sample_index`` for traceability.
    """
    import random

    from datasets import load_dataset

    if seqlen < 0:
        raise ValueError(f"calibration-seqlen must be >= 0 (0=no truncate), got {seqlen}")
    if nsamples < 1:
        raise ValueError(f"calibration-nsamples must be >= 1, got {nsamples}")

    ds = load_dataset(_S1K_HF_ID, split="train")
    if nsamples > len(ds):
        raise ValueError(f"Requested {nsamples} s1k samples but dataset only has {len(ds)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    indices = random.Random(seed).sample(range(len(ds)), k=nsamples)
    for idx in indices:
        text = ds[idx]["text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"s1k sample {idx} has empty or non-string text")
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
            truncation=False,
        )
        inp = encoded["input_ids"]
        cur = int(inp.shape[1])
        if seqlen > 0 and cur > seqlen:
            raise ValueError(
                f"s1k sample {idx} length {cur} exceeds calibration-seqlen={seqlen}; "
                "use --calibration-seqlen 0 for no truncate"
            )
        if cur < 2:
            raise ValueError(f"s1k sample {idx} too short: length={cur}")
        yield {
            "input_ids": inp.contiguous(),
            "attention_mask": torch.ones_like(inp),
            "sample_index": idx,
        }


def _calib_batches_from_loaders(
    dataset: str,
    model_name: str,
    nsamples: int,
    seqlen: int,
    seed: int,
):
    if dataset == "s1k":
        yield from _s1k_calib_batches(model_name, nsamples, seqlen, seed)
        return

    if seqlen <= 0:
        raise ValueError(
            f"calibration-seqlen must be > 0 for {dataset}, got {seqlen} "
            "(use 0 only with s1k)"
        )
    from brq.calib import get_loaders

    trainloader = get_loaders(
        dataset, nsamples=nsamples, seed=seed, seqlen=seqlen, model=model_name
    )
    for inp, _tar in trainloader:
        yield {"input_ids": inp, "attention_mask": torch.ones_like(inp)}


def _load_bf16_probes(
    model_name: str,
    n_probes: int,
    seqlen: int,
    seed: int,
    exclude_indices: set[int],
) -> list[torch.Tensor]:
    """Fixed BF16-control probes from s1k, disjoint from search calibration rows.

    Uses real s1k text (not random vocab noise). Sampling seed is ``seed + 100``
    so probe texts differ from search calibration samples; any residual overlap
    with ``exclude_indices`` is filtered out explicitly.
    """
    import random

    from datasets import load_dataset

    ds = load_dataset(_S1K_HF_ID, split="train")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    order = random.Random(seed + 100).sample(range(len(ds)), k=len(ds))
    probes: list[torch.Tensor] = []
    used: list[int] = []
    for idx in order:
        if len(probes) >= n_probes:
            break
        if idx in exclude_indices:
            continue
        text = ds[idx]["text"]
        if not isinstance(text, str) or not text.strip():
            continue
        inp = tokenizer(
            text, add_special_tokens=False, return_tensors="pt", truncation=False
        )["input_ids"]
        if int(inp.shape[1]) < seqlen:
            continue
        probes.append(inp[:, :seqlen].contiguous())
        used.append(idx)
    if len(probes) < n_probes:
        raise RuntimeError(
            f"Could only build {len(probes)}/{n_probes} probes of length {seqlen}"
        )
    logger.info("BF16 control probes: n=%d seqlen=%d indices=%s", len(probes), seqlen, used)
    return probes


@torch.no_grad()
def _model_logits(model, probes: list[torch.Tensor], device: torch.device) -> list[torch.Tensor]:
    outs: list[torch.Tensor] = []
    for p in probes:
        logits = model(input_ids=p.to(device=device)).logits.float().cpu()
        outs.append(logits)
    return outs


def _probe_compare(ref: list[torch.Tensor], new: list[torch.Tensor]) -> dict[str, float]:
    """Logit drift metrics between two model states on the same probes."""
    nrmse: list[float] = []
    cos: list[float] = []
    flips: list[float] = []
    max_abs = 0.0
    for r, n in zip(ref, new):
        rf = r.reshape(-1, r.shape[-1])
        nf = n.reshape(-1, n.shape[-1])
        num = torch.linalg.norm(nf - rf)
        den = torch.linalg.norm(rf)
        nrmse.append(float((num / (den + 1e-8)).item()))
        max_abs = max(max_abs, float((nf - rf).abs().max().item()))
        cos.append(
            float(torch.nn.functional.cosine_similarity(rf, nf, dim=-1).mean().item())
        )
        flips.append(float((rf.argmax(dim=-1) != nf.argmax(dim=-1)).float().mean().item()))
    nrmse_t = torch.tensor(nrmse, dtype=torch.float64)
    return {
        "n_probes": len(nrmse),
        "mean_logit_nrmse": float(nrmse_t.mean().item()),
        "p95_logit_nrmse": float(torch.quantile(nrmse_t, 0.95).item()),
        "max_abs_logit_delta": max_abs,
        "mean_cosine_similarity": float(sum(cos) / len(cos)),
        "argmax_flip_rate": float(sum(flips) / len(flips)),
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=str, required=True)
    p.add_argument(
        "--calibration-dataset",
        type=str,
        default="wikitext2",
        choices=["wikitext2", "c4", "s1k"],
    )
    p.add_argument("--calibration-nsamples", type=int, default=128)
    p.add_argument(
        "--calibration-seqlen",
        type=int,
        default=2048,
        help="Fixed window for wikitext2/c4. For s1k: 0=no truncate; >0 errors if longer.",
    )
    p.add_argument("--activation-rows", type=int, default=512)
    p.add_argument("--weight-rows", type=int, default=512)
    p.add_argument("--candidate-window", type=int, default=128)
    p.add_argument("--neighbor-k", type=int, default=32)
    p.add_argument("--beam-width-g4", type=int, default=4)
    p.add_argument("--beam-width-g64", type=int, default=4)
    p.add_argument("--refine-passes", type=int, default=2)
    p.add_argument("--refine-bad-blocks", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=1, help="Parallel layer search workers (CPU)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--validation-seeds", type=str, default="42,43,44")
    p.add_argument("--min-relative-improvement", type=float, default=0.001)
    p.add_argument("--min-validation-wins", type=int, default=2)
    p.add_argument("--improvement-std-multiplier", type=float, default=2.0)
    p.add_argument("--max-bf16-reorder-drift", type=float, default=0.002)
    p.add_argument("--refine-enabled", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--proxy-audit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--refine-max-rounds", type=int, default=2)
    p.add_argument("--refine-candidates-per-round", type=int, default=64)
    p.add_argument("--overwrite-output", action="store_true")
    p.add_argument("--save-identity-copy", type=str, default="")
    p.add_argument("--bf16-control-probes", type=int, default=16)
    p.add_argument("--bf16-control-seqlen", type=int, default=128)
    p.add_argument("--bf16-control-seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--save-reordered-model", type=str, default="")
    p.add_argument("--layers", type=str, default="all")
    p.add_argument("--trust-remote-code", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    t_start = time.time()
    args = build_argparser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        args.dtype
    ]
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    config = SearchConfig(
        activation_rows=args.activation_rows,
        weight_rows=args.weight_rows,
        candidate_window=args.candidate_window,
        neighbor_k=args.neighbor_k,
        beam_width_g4=args.beam_width_g4,
        beam_width_g64=args.beam_width_g64,
        refine_passes=args.refine_passes,
        refine_bad_blocks=args.refine_bad_blocks,
        refine_max_swaps_per_stage=8,
        seed=args.seed,
        validation_seeds=tuple(int(s) for s in args.validation_seeds.split(",")),
        min_relative_improvement=args.min_relative_improvement,
        min_validation_wins=args.min_validation_wins,
        improvement_std_multiplier=args.improvement_std_multiplier,
        max_bf16_reorder_drift=args.max_bf16_reorder_drift,
        refine_enabled=args.refine_enabled,
        refine_max_rounds=args.refine_max_rounds,
        refine_candidates_per_round=args.refine_candidates_per_round,
        proxy_audit_enabled=args.proxy_audit,
    )
    out_dir = Path(args.output_dir)
    _ensure_output_dir(out_dir, args.overwrite_output)

    logger.info("Loading model %s ...", args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        device_map=None,
    )
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=False, trust_remote_code=args.trust_remote_code
    )

    all_specs = discover_swiglu_mlps(model)
    logger.info("Discovered %d SwiGLU MLPs", len(all_specs))
    layer_indices = _parse_layers(args.layers, max((s.layer_index for s in all_specs), default=0) + 1)

    calib = list(
        _calib_batches_from_loaders(
            args.calibration_dataset,
            args.model,
            args.calibration_nsamples,
            args.calibration_seqlen,
            args.seed,
        )
    )
    search_sample_indices = {
        int(b["sample_index"]) for b in calib if b.get("sample_index") is not None
    }
    probes: list[torch.Tensor] = []
    ref_logits: list[torch.Tensor] = []
    if args.bf16_control_probes > 0:
        probes = _load_bf16_probes(
            args.model,
            args.bf16_control_probes,
            args.bf16_control_seqlen,
            args.bf16_control_seed,
            search_sample_indices,
        )
        ref_logits = _model_logits(model, probes, device)

    if args.save_identity_copy:
        id_path = Path(args.save_identity_copy)
        id_path.mkdir(parents=True, exist_ok=True)
        logger.info("Saving identity copy to %s", id_path)
        model.save_pretrained(id_path, safe_serialization=True, max_shard_size="5GB")
        tokenizer.save_pretrained(id_path)
    # Spread activation rows across calibration samples (critical for long s1k).
    max_rows_per_batch = max(
        1, (args.activation_rows + len(calib) - 1) // len(calib)
    )
    if args.calibration_dataset == "s1k":
        lengths = [int(b["input_ids"].shape[1]) for b in calib]
        logger.info(
            "s1k calib: n=%d seqlen=min/median/max=%d/%d/%d max_rows_per_batch=%d",
            len(calib),
            min(lengths),
            sorted(lengths)[len(lengths) // 2],
            max(lengths),
            max_rows_per_batch,
        )

    import transformers as _tf

    cfg_dump = {
        "model": args.model,
        "model_revision": getattr(getattr(model, "config", None), "_commit_hash", None),
        "calibration_dataset": args.calibration_dataset,
        "calibration_nsamples": args.calibration_nsamples,
        "calibration_seqlen": args.calibration_seqlen,
        "calibration_sample_indices": [
            b.get("sample_index") for b in calib if b.get("sample_index") is not None
        ],
        "calibration_text_lengths": [
            int(b["input_ids"].shape[1]) for b in calib
        ],
        "max_rows_per_batch": max_rows_per_batch,
        "layers": args.layers,
        "search_config": {k: getattr(config, k) for k in config.__dataclass_fields__},
        "quant_qtype": "hifx4",
        "torch_version": torch.__version__,
        "transformers_version": _tf.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "dtype": args.dtype,
        "device": str(device),
        "n_mlp": len(all_specs),
        "command": " ".join(sys.argv),
        "start_time_unix": t_start,
    }
    (out_dir / "config.json").write_text(json.dumps(cfg_dump, indent=2))
    logger.info("Config:\n%s", json.dumps(cfg_dump, indent=2))

    packed = reorder_model_mlps(
        model,
        calib,
        config,
        device=device,
        layer_indices=layer_indices,
        metrics_path=out_dir / "layer_metrics.jsonl",
        num_workers=args.num_workers,
        max_rows_per_batch=max_rows_per_batch,
    )

    # Strict FP32 check on each non-identity MLP, then soft full-model BF16 check.
    # Near-zero logits make relative error meaningless for the full model.
    for spec in packed["specs"]:
        perm = packed["permutations"][spec.name]
        if torch.equal(perm, torch.arange(perm.numel())):
            continue
        gate, up, down = get_mlp_modules(model, spec)
        d_model = gate.weight.shape[1]
        x = torch.randn(4, d_model, dtype=torch.float32)
        g_w = gate.weight.detach().float().cpu()
        u_w = up.weight.detach().float().cpu()
        d_w = down.weight.detach().float().cpu()
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.numel())
        g_old = g_w[inv]
        u_old = u_w[inv]
        d_old = d_w[:, inv]
        y_perm = (torch.nn.functional.silu(x @ g_w.T) * (x @ u_w.T)) @ d_w.T
        y_old = (torch.nn.functional.silu(x @ g_old.T) * (x @ u_old.T)) @ d_old.T
        if not torch.allclose(y_perm, y_old, rtol=1e-5, atol=1e-5):
            raise RuntimeError(
                f"FP32 MLP equivalence failed for {spec.name}. Refusing to save."
            )

    if probes:
        new_logits = _model_logits(model, probes, device)
        probe_metrics = _probe_compare(ref_logits, new_logits)
        (out_dir / "bf16_probes.json").write_text(json.dumps(probe_metrics, indent=2))
        logger.info("BF16 reorder probe metrics:\n%s", json.dumps(probe_metrics, indent=2))
    logger.info("FP equivalence check passed (FP32 MLP)")

    torch.save(packed["permutations"], out_dir / "permutations.pt")
    torch.save(packed["permutations"], out_dir / "selected_permutations.pt")
    torch.save(
        packed["best_candidate_permutations"], out_dir / "candidate_permutations.pt"
    )
    torch.save(
        packed["candidate_permutations"], out_dir / "all_candidate_permutations.pt"
    )
    summary = {
        "n_layers": len(packed["results"]),
        "n_accepted": sum(1 for r in packed["results"] if r["accepted"]),
        "results": packed["results"],
        "end_time_unix": time.time(),
        "elapsed_sec": time.time() - t_start,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if args.save_reordered_model:
        save_path = Path(args.save_reordered_model)
        save_path.mkdir(parents=True, exist_ok=True)
        logger.info("Saving reordered model to %s", save_path)
        model.save_pretrained(save_path, safe_serialization=True, max_shard_size="5GB")
        tokenizer.save_pretrained(save_path)

    logger.info("Done. Artifacts in %s", out_dir)


if __name__ == "__main__":
    main()
