"""End-to-end scheme evaluation: PPL + ARC/MMLU (lm_eval) + MMLU-Pro (lighteval)."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

_ROOT = Path(__file__).resolve().parents[1]
_HIFLOAT4 = _ROOT.parent
_REPO = _HIFLOAT4.parent
for p in (_ROOT, _HIFLOAT4):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.evaluate import (  # noqa: E402
    apply_activation_params,
    apply_weight_quantization,
    evaluate_lm_eval_tasks,
    evaluate_ppl_wikitext2,
    load_model_tokenizer,
    save_json,
)
from src.fixed_thresholds import get_baseline_config  # noqa: E402
from src.quantizer import HiF4QuantConfig  # noqa: E402


def _env_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
    return info


def load_param_map(path: str | None) -> dict[str, HiF4QuantConfig] | None:
    if not path:
        return None
    blob = torch.load(path, map_location="cpu", weights_only=False)
    out: dict[str, HiF4QuantConfig] = {}
    for k, v in blob.items():
        if isinstance(v, HiF4QuantConfig):
            out[k] = v
        elif isinstance(v, dict):
            out[k] = HiF4QuantConfig(
                s0_divisor=float(v["s0_divisor"]),
                e8_threshold=float(v["e8_threshold"]),
                e4_threshold=float(v["e4_threshold"]),
            )
        else:
            raise TypeError(f"bad param_map entry type {type(v)} for {k}")
    return out


def extract_task_metrics(lm_results: dict[str, Any]) -> dict[str, Any]:
    results = lm_results.get("results", lm_results)
    slim = {}
    for task, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        slim[task] = {
            k: v
            for k, v in metrics.items()
            if isinstance(v, (int, float)) and "stderr" not in k
        }
    return slim


def _pick_metric(task_metrics: dict) -> tuple[str | None, float | None]:
    for key in (
        "extractive_match",
        "exact_match",
        "loglikelihood_acc",
        "acc",
        "acc_norm",
        "quasi_exact_match",
    ):
        if key in task_metrics and isinstance(task_metrics[key], (int, float)):
            return key, float(task_metrics[key])
    for key, val in task_metrics.items():
        if key.endswith("_stderr") or key == "alias":
            continue
        if isinstance(val, (int, float)):
            return key, float(val)
    return None, None


def parse_lighteval_mmlu_pro(output_dir: Path) -> dict[str, Any]:
    files = sorted(output_dir.rglob("results_*.json"))
    if not files:
        return {"status": "missing"}
    path = files[-1]
    obj = json.loads(path.read_text(encoding="utf-8"))
    results = obj.get("results", obj)
    for task_key, metrics in results.items():
        if task_key == "all" or not isinstance(metrics, dict):
            continue
        base = task_key.split("|", 1)[0]
        if base == "mmlu_pro" or base.startswith("mmlu_pro:"):
            mk, val = _pick_metric(metrics)
            if val is not None:
                return {
                    "status": "ok",
                    "file": str(path),
                    "score": val,
                    "metric_key": f"{task_key}:{mk}",
                    "raw_task": metrics,
                }
    return {"status": "no_score", "file": str(path)}


def save_weight_ckpt(model, tokenizer, ckpt_dir: Path) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)


def run_lighteval_mmlu_pro(
    *,
    ckpt_dir: Path,
    output_dir: Path,
    max_samples: int,
    gpu: str | None,
) -> dict[str, Any]:
    """Call repo-root main.py: vLLM + lighteval, max_samples=300."""
    output_dir.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    cmd = [
        py,
        str(_REPO / "main.py"),
        "--model_path",
        str(ckpt_dir.resolve()),
        "--datasets",
        "mmlu_pro|0",
        "--max_samples",
        str(max_samples),
        "--tensor_parallel_size",
        "1",
        "--max_model_length",
        "32768",
        "--max_new_tokens",
        "32768",
        "--temperature",
        "0.7",
        "--top_p",
        "0.8",
        "--top_k",
        "20",
        "--gpu_memory_utilization",
        "0.9",
        "--fake_act_quant",
        "hif4",
        "--fake_act_quant_exclude",
        "lm_head",
        "--disable_thinking",
        "--output_dir",
        str(output_dir.resolve()),
    ]
    env = dict(**{k: v for k, v in __import__("os").environ.items()})
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    print("  lighteval cmd:", " ".join(cmd), flush=True)
    stdout_path = output_dir / "lighteval_stdout.log"
    stderr_path = output_dir / "lighteval_stderr.log"
    with stdout_path.open("w", encoding="utf-8") as out_f, stderr_path.open(
        "w", encoding="utf-8"
    ) as err_f:
        proc = subprocess.run(
            cmd,
            cwd=str(_REPO),
            env=env,
            check=False,
            stdout=out_f,
            stderr=err_f,
            text=True,
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"lighteval failed rc={proc.returncode}; see {stderr_path}"
        )
    parsed = parse_lighteval_mmlu_pro(output_dir)
    parsed["max_samples"] = max_samples
    parsed["backend"] = "lighteval_vllm"
    parsed["fake_act_quant"] = "hif4"
    return parsed


def run_scheme(
    *,
    model_name: str,
    scheme: str,
    device: str,
    act_param_map: dict[str, HiF4QuantConfig] | None,
    fixed_best: HiF4QuantConfig | None,
    weight_updates: dict[str, torch.Tensor] | None,
    ppl_max_length: int,
    lm_batch_size: int,
    mmlu_pro_limit: int,
    skip_downstream: bool,
    out_dir: Path,
    keep_ckpt: bool,
    cuda_visible: str | None,
) -> dict[str, Any]:
    print(f"=== scheme={scheme} loading model ===")
    model, tok = load_model_tokenizer(model_name, device=device)

    weight_mode = "standard"
    act_default = get_baseline_config("standard")
    act_map = None

    if scheme == "baseline_standard":
        weight_mode = "standard"
        act_default = get_baseline_config("standard")
    elif scheme == "weight_fixed_best":
        if fixed_best is None:
            raise ValueError("weight_fixed_best requires --fixed-best-*")
        weight_mode = "fixed"
        act_default = get_baseline_config("standard")
    elif scheme == "weight_search_fast":
        weight_mode = "search_fast"
        act_default = get_baseline_config("standard")
    elif scheme == "weight_search_full":
        weight_mode = "search_full"
        act_default = get_baseline_config("standard")
    elif scheme == "act_calib_only":
        weight_mode = "standard"
        act_map = act_param_map
        act_default = None if act_map else get_baseline_config("standard")
    elif scheme == "act_fixed_best":
        if fixed_best is None:
            raise ValueError("act_fixed_best requires --fixed-best-*")
        weight_mode = "standard"
        act_default = fixed_best
    elif scheme == "joint":
        weight_mode = "search_fast"
        act_map = act_param_map
        act_default = None if act_map else get_baseline_config("standard")
    else:
        raise ValueError(f"unknown scheme {scheme}")

    print(f"  applying weight quantization mode={weight_mode}")
    pre = weight_updates if weight_mode in ("search_fast", "search_full") else None
    wstats = apply_weight_quantization(
        model,
        mode=weight_mode,
        device=device,
        fixed_config=fixed_best,
        precomputed_updates=pre,
    )

    # Save weight-only ckpt BEFORE act wrappers (for lighteval / vLLM).
    ckpt_dir = out_dir / "tmp_ckpt" / scheme
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    print(f"  saving weight ckpt -> {ckpt_dir}")
    save_weight_ckpt(model, tok, ckpt_dir)

    print(f"  applying activation params (layers={0 if act_map is None else len(act_map)})")
    replaced = apply_activation_params(model, act_map, default=act_default)
    print(f"  act-quantized linears: {len(replaced)}")

    out: dict[str, Any] = {
        "scheme": scheme,
        "weight_nmse_by_layer": wstats,
        "mean_weight_nmse": (sum(wstats.values()) / len(wstats) if wstats else None),
        "num_act_quant_layers": len(replaced),
        "mmlu_pro_note": (
            "MMLU-Pro via lighteval+vLLM fake_act_quant=hif4 (standard 7/4/2). "
            "Custom offline act thresholds apply to PPL/ARC/MMLU only."
            if act_map is not None or act_default != get_baseline_config("standard")
            else "MMLU-Pro via lighteval+vLLM fake_act_quant=hif4."
        ),
    }

    print("  evaluating PPL...")
    ppl = evaluate_ppl_wikitext2(
        model, tok, max_length=ppl_max_length, stride=ppl_max_length, device=device
    )
    out["ppl_wikitext2"] = ppl
    print(f"  PPL={ppl:.4f}")

    if not skip_downstream:
        print("  evaluating ARC/MMLU (lm_eval)...")
        lm = evaluate_lm_eval_tasks(
            model,
            tok,
            tasks=["arc_easy", "arc_challenge", "mmlu"],
            num_fewshot=0,
            batch_size=lm_batch_size,
        )
        out["lm_eval"] = extract_task_metrics(lm)
        save_json(out_dir / f"{scheme}_mid_arc_mmlu.json", out)

        # Free HF model before vLLM.
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"  evaluating MMLU-Pro via lighteval max_samples={mmlu_pro_limit}...")
        mpro_dir = out_dir / "mmlu_pro" / scheme
        out["mmlu_pro"] = run_lighteval_mmlu_pro(
            ckpt_dir=ckpt_dir,
            output_dir=mpro_dir,
            max_samples=mmlu_pro_limit,
            gpu=cuda_visible,
        )
        print(f"  mmlu_pro={out['mmlu_pro']}")

    if not keep_ckpt and ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--schemes",
        type=str,
        default="baseline_standard,weight_fixed_best,weight_search_fast,act_calib_only,joint",
    )
    parser.add_argument("--act-param-map", type=str, default=None)
    parser.add_argument("--weight-updates", type=str, default=None)
    parser.add_argument("--fixed-best-d", type=float, default=7.0)
    parser.add_argument("--fixed-best-t8", type=float, default=3.75)
    parser.add_argument("--fixed-best-t4", type=float, default=1.875)
    parser.add_argument("--ppl-max-length", type=int, default=2048)
    parser.add_argument("--lm-batch-size", type=int, default=4)
    parser.add_argument("--mmlu-pro-limit", type=int, default=300)
    parser.add_argument("--skip-downstream", action="store_true")
    parser.add_argument("--keep-ckpt", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else _ROOT / "results" / f"{stamp}_phase6_e2e"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(vars(args)), encoding="utf-8")
    (out_dir / "environment.json").write_text(json.dumps(_env_info(), indent=2), encoding="utf-8")

    act_map = load_param_map(args.act_param_map)
    weight_updates = None
    if args.weight_updates:
        weight_updates = torch.load(args.weight_updates, map_location="cpu", weights_only=False)
        print(f"Loaded {len(weight_updates)} precomputed weight updates")
    fixed_best = HiF4QuantConfig(
        s0_divisor=args.fixed_best_d,
        e8_threshold=args.fixed_best_t8,
        e4_threshold=args.fixed_best_t4,
    )
    cuda_visible = __import__("os").environ.get("CUDA_VISIBLE_DEVICES")

    all_results = {}
    for scheme in [s.strip() for s in args.schemes.split(",") if s.strip()]:
        try:
            all_results[scheme] = run_scheme(
                model_name=args.model,
                scheme=scheme,
                device=args.device,
                act_param_map=act_map,
                fixed_best=fixed_best,
                weight_updates=weight_updates,
                ppl_max_length=args.ppl_max_length,
                lm_batch_size=args.lm_batch_size,
                mmlu_pro_limit=args.mmlu_pro_limit,
                skip_downstream=args.skip_downstream,
                out_dir=out_dir,
                keep_ckpt=args.keep_ckpt,
                cuda_visible=cuda_visible,
            )
            save_json(out_dir / f"{scheme}.json", all_results[scheme])
        except Exception as e:
            all_results[scheme] = {"error": repr(e)}
            save_json(out_dir / f"{scheme}.json", all_results[scheme])
            print(f"ERROR in {scheme}: {e}")
            import traceback

            traceback.print_exc()

    save_json(out_dir / "raw_metrics.json", all_results)
    lines = [
        "# Phase 6 End-to-End Summary",
        "",
        f"Model: `{args.model}`",
        f"Device: `{args.device}`",
        "MMLU-Pro: lighteval + vLLM (`max_samples=300`, `fake_act_quant=hif4`)",
        "",
        "| scheme | PPL | mean_weight_nmse | mmlu_pro |",
        "| --- | ---: | ---: | ---: |",
    ]
    for scheme, res in all_results.items():
        if "error" in res:
            lines.append(f"| {scheme} | ERROR | | |")
            continue
        ppl = res.get("ppl_wikitext2")
        wn = res.get("mean_weight_nmse")
        mp = (res.get("mmlu_pro") or {}).get("score")
        lines.append(
            f"| {scheme} | {ppl:.4f} | {wn if wn is None else f'{wn:.6e}'} | "
            f"{'-' if mp is None else f'{mp:.4f}'} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
