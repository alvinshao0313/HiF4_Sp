"""Compare offline weight-quantization wall time: analytic standard vs S0+e8/e4 search.

Streams the 128 target Linear weights from the HF snapshot via safetensors,
times per-layer standard quantization and search (fast/full) on GPU, and
writes an aggregate table.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.quantizer import HiF4QuantConfig, quantize_hif4  # noqa: E402
from src.weight_search import search_weight_groups  # noqa: E402


def iter_target_weight_names(model: str) -> list[str]:
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(model, allow_patterns=["*.json"]))
    index = json.loads((path / "model.safetensors.index.json").read_text())
    suffixes = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    names = [
        k[: -len(".weight")]
        for k in index["weight_map"]
        if k.endswith(".weight") and k[: -len(".weight")].endswith(suffixes)
        and ".layers." in k
    ]
    return sorted(names)


def load_weights(model: str, names: list[str]) -> dict[str, torch.Tensor]:
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    path = Path(snapshot_download(model))
    index = json.loads((path / "model.safetensors.index.json").read_text())
    wm = index["weight_map"]
    out: dict[str, torch.Tensor] = {}
    by_file: dict[str, list[str]] = {}
    for n in names:
        by_file.setdefault(wm[f"{n}.weight"], []).append(n)
    for fname, keys in sorted(by_file.items()):
        with safe_open(path / fname, framework="pt") as f:
            for k in keys:
                out[k] = f.get_tensor(f"{k}.weight")
    return out


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--reps-standard", type=int, default=5)
    parser.add_argument("--reps-fast", type=int, default=3)
    parser.add_argument("--reps-full", type=int, default=1)
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = iter_target_weight_names(args.model)
    print(f"{len(names)} target layers")
    weights = load_weights(args.model, names)

    rows = []
    for name in names:
        W = weights[name]
        # warmup standard
        quantize_hif4(W.to(device), config=HiF4QuantConfig())
        sync(device)

        t0 = time.perf_counter()
        for _ in range(args.reps_standard):
            quantize_hif4(W.to(device), config=HiF4QuantConfig())
            sync(device)
        t_std = (time.perf_counter() - t0) / args.reps_standard

        search_weight_groups(W, budget="fast", device=device)  # warmup
        t0 = time.perf_counter()
        for _ in range(args.reps_fast):
            search_weight_groups(W, budget="fast", device=device)
        t_fast = (time.perf_counter() - t0) / args.reps_fast

        t0 = time.perf_counter()
        for _ in range(args.reps_full):
            search_weight_groups(W, budget="full", device=device)
        t_full = (time.perf_counter() - t0) / args.reps_full

        rows.append({
            "layer": name,
            "shape": list(W.shape),
            "t_standard_s": t_std,
            "t_search_fast_s": t_fast,
            "t_search_full_s": t_full,
        })
        print(f"  {name}: std={t_std*1e3:.1f}ms fast={t_fast:.3f}s full={t_full:.3f}s", flush=True)
        del weights[name]
        if device.type == "cuda":
            torch.cuda.empty_cache()

    tot_std = sum(r["t_standard_s"] for r in rows)
    tot_fast = sum(r["t_search_fast_s"] for r in rows)
    tot_full = sum(r["t_search_full_s"] for r in rows)
    summary = {
        "layers": len(rows),
        "total_standard_s": tot_std,
        "total_search_fast_s": tot_fast,
        "total_search_full_s": tot_full,
        "fast_overhead_x": tot_fast / tot_std if tot_std > 0 else None,
        "full_overhead_x": tot_full / tot_std if tot_std > 0 else None,
        "fast_minus_standard_s": tot_fast - tot_std,
    }
    (out_dir / "raw_metrics.json").write_text(
        json.dumps({"summary": summary, "layers": rows}, indent=2), encoding="utf-8"
    )
    lines = [
        "# 权重量化耗时对比（离线，一次性成本）",
        "",
        f"- 模型：`{args.model}`；设备：{device}",
        f"- 层数：{len(rows)}；standard reps={args.reps_standard}，fast reps={args.reps_fast}，full reps={args.reps_full}",
        "",
        "| 方案 | 总耗时 | 相对 standard |",
        "| --- | ---: | ---: |",
        f"| standard 解析阈值 | {tot_std:.2f} s | 1.0x |",
        f"| 搜索 fast（S0±1 × e8/e4 8 组合） | {tot_fast:.2f} s | {summary['fast_overhead_x']:.1f}x |",
        f"| 搜索 full（S0±2 × e8/e4 8 组合） | {tot_full:.2f} s | {summary['full_overhead_x']:.1f}x |",
        "",
        "逐层数据见 raw_metrics.json。",
    ]
    (out_dir / "bench.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
