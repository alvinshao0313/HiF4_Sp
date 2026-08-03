"""Diagnose why per-layer activation calibration gives no downstream gain.

Four analyses on the existing activation store + calibration results:

A. cal->val generalization gap (from summary_per_layer.json, CPU only)
B. objective validity: diagonal-approx output MSE vs real ||(X-Xq)W||^2
C. per-block-index calibrated (d,t8,t4): storable offline, no online search
D. per-row-block oracle (search_weight_groups): online-search upper bound

Ladder reported per layer: standard -> per-layer -> per-block-index -> oracle.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.activation_calibration import _grid, _split_rows  # noqa: E402
from src.formats import quantize_s1p2_magnitude, round_bfloat16, round_e6m2  # noqa: E402
from src.metrics import nmse  # noqa: E402
from src.quantizer import HiF4QuantConfig, quantize_hif4  # noqa: E402
from src.weight_search import search_weight_groups  # noqa: E402


# ---------------------------------------------------------------------------
# vectorized (d,t8,t4)-parametrized quantization of block tensor [R, 64]
# ---------------------------------------------------------------------------

def block_err_for_configs(
    x: torch.Tensor,  # [R, 64] float32 on device
    energy: torch.Tensor,  # [64] float32 on device
    ds: torch.Tensor,  # [C]
    t8s: torch.Tensor,
    t4s: torch.Tensor,
) -> torch.Tensor:
    """Return weighted squared error per config: [C]."""
    device = x.device
    C = ds.numel()
    R = x.shape[0]
    abs_x = x.abs()  # [R,64]
    amax64 = abs_x.amax(dim=-1)  # [R]
    amax8 = abs_x.reshape(R, 8, 8).amax(dim=-1)  # [R,8]
    amax4 = abs_x.reshape(R, 16, 4).amax(dim=-1)  # [R,16]
    sign = x.sign()

    # S0 per config: [R, C]
    recip_d = round_bfloat16(1.0 / ds)
    ratio = round_bfloat16(amax64.unsqueeze(1) * recip_d.unsqueeze(0))
    s0 = round_e6m2(ratio)
    s0 = torch.where(s0 > 0, s0, torch.ones_like(s0))
    rec = round_bfloat16(1.0 / s0)  # [R, C]

    # e8: [R, C, 8]
    e8 = (amax8.unsqueeze(1) * rec.unsqueeze(-1) >= t8s.view(1, C, 1)).float()
    e8_per4 = e8.repeat_interleave(2, dim=-1)  # [R, C, 16]
    # e4: [R, C, 16]
    e4 = (
        amax4.unsqueeze(1) * rec.unsqueeze(-1) / (2.0 ** e8_per4)
        >= t4s.view(1, C, 1)
    ).float()
    e8_elem = e8.repeat_interleave(8, dim=-1)  # [R, C, 64]
    e4_elem = e4.repeat_interleave(4, dim=-1)  # [R, C, 64]
    scale = s0.unsqueeze(-1) * (2.0 ** (e8_elem + e4_elem))
    normalized = abs_x.unsqueeze(1) * (rec.unsqueeze(-1) / (2.0 ** (e8_elem + e4_elem)))
    payload = quantize_s1p2_magnitude(normalized)
    recon = sign.unsqueeze(1) * scale * payload
    w = energy.view(1, 1, 64) if energy.dim() == 1 else energy.unsqueeze(1)
    err2 = (x.unsqueeze(1) - recon).pow(2) * w
    return err2.sum(dim=(0, 2))  # [C]


def quantize_block_with_params(
    x: torch.Tensor, d: float, t8: float, t4: float
) -> torch.Tensor:
    """Single-config quantization of [R, 64]; returns reconstruction."""
    cfg = HiF4QuantConfig(s0_divisor=d, e8_threshold=t8, e4_threshold=t4)
    return quantize_hif4(x, config=cfg).reconstruction


def config_grid(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[float, float, float]]]:
    ds, t8s, t4s = _grid()
    combos = list(product(ds, t8s, t4s))
    d_t = torch.tensor([c[0] for c in combos], device=device, dtype=torch.float32)
    t8_t = torch.tensor([c[1] for c in combos], device=device, dtype=torch.float32)
    t4_t = torch.tensor([c[2] for c in combos], device=device, dtype=torch.float32)
    return d_t, t8_t, t4_t, combos


# ---------------------------------------------------------------------------
# analyses
# ---------------------------------------------------------------------------

def analysis_generalization(calib_dir: Path) -> dict:
    blob = json.loads((calib_dir / "summary_per_layer.json").read_text())
    layers = blob["layers"]
    cal_imp, val_imp = [], []
    flip = []
    for name, v in layers.items():
        cal_i = v["standard_val_output_mse"] - v["cal_output_mse"]  # proxy not exact
        val_i = v["val_improvement"]
        cal_imp.append(cal_i)
        val_imp.append(val_i)
        if cal_i > 0 and val_i <= 0:
            flip.append(name)
    n = len(layers)
    return {
        "layers": n,
        "val_improved": sum(1 for x in val_imp if x > 0),
        "val_worse_or_equal": sum(1 for x in val_imp if x <= 0),
        "mean_val_improvement": sum(val_imp) / n,
        "cal_positive_val_nonpositive": flip,
    }


def load_weights_for_layers(model: str, names: list[str]) -> dict[str, torch.Tensor]:
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    path = Path(snapshot_download(model))
    index = json.loads((path / "model.safetensors.index.json").read_text())
    wm = index["weight_map"]
    out: dict[str, torch.Tensor] = {}
    by_file: dict[str, list[str]] = {}
    for n in names:
        key = f"{n}.weight"
        if key not in wm and key.startswith("model.layers."):
            key = "model.language_model." + key[len("model."):]
        if key not in wm:
            raise KeyError(f"weight key not found for layer {n}")
        by_file.setdefault(wm[key], []).append(key)
    for fname, keys in by_file.items():
        with safe_open(path / fname, framework="pt") as f:
            for k in keys:
                store_name = k[: -len(".weight")]
                if store_name.startswith("model.language_model."):
                    store_name = "model." + store_name[len("model.language_model."):]
                out[store_name] = f.get_tensor(k)
    return out


def analysis_objective(
    inputs: dict[str, torch.Tensor],
    energy: dict[str, torch.Tensor],
    model: str,
    device: torch.device,
    sample_names: list[str],
) -> dict:
    weights = load_weights_for_layers(model, sample_names)
    d_t, t8_t, t4_t, combos = config_grid(device)
    rows = []
    for name in sample_names:
        if name not in inputs or name not in weights:
            continue
        x = inputs[name].float()
        x_cal, x_val = _split_rows(x)
        W = weights[name].float().to(device)  # [out, in]
        e_cal = energy[name].float().to(device)
        xc = x_cal.to(device)
        xv = x_val.to(device)

        # diag-approx search (same as calibration)
        errs_diag = block_err_for_configs_full(xc, e_cal, d_t, t8_t, t4_t)
        best_diag = int(errs_diag.argmin().item())

        # real output MSE search
        best_real, best_real_err = -1, float("inf")
        real_errs = []
        for ci, (d, t8, t4) in enumerate(combos):
            xq = quantize_block_with_params(xc, d, t8, t4)
            out_err = ((xc - xq) @ W.T).pow(2).sum().item()
            real_errs.append(out_err)
            if out_err < best_real_err:
                best_real_err, best_real = out_err, ci

        # val real output MSE for both choices + standard
        def val_real(ci):
            d, t8, t4 = combos[ci]
            xq = quantize_block_with_params(xv, d, t8, t4)
            return float(((xv - xq) @ W.T).pow(2).sum().item())

        std_cfg = HiF4QuantConfig()
        xq_std = quantize_hif4(xv, config=std_cfg).reconstruction
        val_std = float(((xv - xq_std) @ W.T).pow(2).sum().item())
        rows.append({
            "layer": name,
            "diag_pick": combos[best_diag],
            "real_pick": combos[best_real],
            "same_pick": best_diag == best_real,
            "val_real_diag_pick": val_real(best_diag),
            "val_real_real_pick": val_real(best_real),
            "val_real_standard": val_std,
        })
        del W
        torch.cuda.empty_cache()
    return {"layers": rows}


def block_err_for_configs_full(x, energy, d_t, t8_t, t4_t):
    """Grid error over the whole [R, C] matrix (all blocks share config)."""
    R, Cdim = x.shape
    B = Cdim // 64
    xb = x.reshape(R * B, 64)
    # xb row (r*B + b) needs energy of block b
    eb = energy.reshape(1, Cdim).expand(R, Cdim).reshape(R * B, 64)
    return block_err_for_configs(xb, eb, d_t, t8_t, t4_t)


def analysis_ladder(
    inputs: dict[str, torch.Tensor],
    energy: dict[str, torch.Tensor],
    param_map: dict[str, HiF4QuantConfig] | None,
    device: torch.device,
) -> dict:
    d_t, t8_t, t4_t, combos = config_grid(device)
    layer_rows = []
    for name in sorted(inputs):
        x = inputs[name].float()
        e = energy[name].float()
        R, Cdim = x.shape
        B = Cdim // 64
        x_cal, x_val = _split_rows(x)
        xc, xv = x_cal.to(device), x_val.to(device)
        ec, ev = e.to(device), e.to(device)

        # --- standard on val
        xq_std = quantize_hif4(xv, config=HiF4QuantConfig()).reconstruction
        nmse_std = nmse(xv, xq_std)
        wstd = float(((xv - xq_std).pow(2) * ev.view(1, -1)).sum().item())

        # --- per-layer grid pick on cal -> val
        errs = block_err_for_configs_full(xc, ec, d_t, t8_t, t4_t)
        bi = int(errs.argmin().item())
        d, t8, t4 = combos[bi]
        xq_pl = quantize_hif4(
            xv, config=HiF4QuantConfig(s0_divisor=d, e8_threshold=t8, e4_threshold=t4)
        ).reconstruction
        nmse_pl = nmse(xv, xq_pl)
        wpl = float(((xv - xq_pl).pow(2) * ev.view(1, -1)).sum().item())

        # --- per-block-index pick on cal -> val
        xcb = xc.reshape(xc.shape[0], B, 64)
        xvb = xv.reshape(xv.shape[0], B, 64)
        ecb = ec.reshape(B, 64)
        block_params: list[tuple[float, float, float]] = []
        for b in range(B):
            errs_b = block_err_for_configs(
                xcb[:, b, :].contiguous(), ecb[b], d_t, t8_t, t4_t
            )
            block_params.append(combos[int(errs_b.argmin().item())])
        xq_pb = torch.empty_like(xvb)
        for b in range(B):
            d_b, t8_b, t4_b = block_params[b]
            xq_pb[:, b, :] = quantize_block_with_params(xvb[:, b, :].contiguous(), d_b, t8_b, t4_b)
        xq_pb = xq_pb.reshape(xv.shape)
        nmse_pb = nmse(xv, xq_pb)
        wpb = float(((xv - xq_pb).pow(2) * ev.view(1, -1)).sum().item())

        # --- oracle: per-row-block search (upper bound, needs online search)
        oracle = search_weight_groups(xv, budget="full", device=device)
        xq_or = oracle.reconstruction.float()
        nmse_or = nmse(xv, xq_or)
        wor = float(((xv - xq_or).pow(2) * ev.view(1, -1)).sum().item())

        # distinct params used across blocks
        n_distinct = len(set(block_params))
        layer_rows.append({
            "layer": name,
            "blocks": B,
            "per_layer_pick": [d, t8, t4],
            "block_distinct_params": n_distinct,
            "nmse_standard": nmse_std,
            "nmse_per_layer": nmse_pl,
            "nmse_per_block": nmse_pb,
            "nmse_oracle": nmse_or,
            "wmse_standard": wstd,
            "wmse_per_layer": wpl,
            "wmse_per_block": wpb,
            "wmse_oracle": wor,
        })
        print(f"  [ladder] {name}: std={nmse_std:.4e} layer={nmse_pl:.4e} "
              f"block={nmse_pb:.4e} oracle={nmse_or:.4e} distinct={n_distinct}/{B}", flush=True)

    def mean(key):
        return sum(r[key] for r in layer_rows) / max(len(layer_rows), 1)

    return {
        "layers": layer_rows,
        "mean": {
            "nmse_standard": mean("nmse_standard"),
            "nmse_per_layer": mean("nmse_per_layer"),
            "nmse_per_block": mean("nmse_per_block"),
            "nmse_oracle": mean("nmse_oracle"),
            "wmse_standard": mean("wmse_standard"),
            "wmse_per_layer": mean("wmse_per_layer"),
            "wmse_per_block": mean("wmse_per_block"),
            "wmse_oracle": mean("wmse_oracle"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=str, required=True)
    parser.add_argument("--calib-dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--obj-check-layers", type=int, default=8)
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    blob = torch.load(args.store, map_location="cpu", weights_only=False)
    inputs = blob["inputs"]
    energy = blob["weight_col_energy"]

    print("[A] cal->val generalization", flush=True)
    gen = analysis_generalization(Path(args.calib_dir))

    print("[B] objective validity (diag-approx vs real output MSE)", flush=True)
    names = sorted(inputs)
    picks = [names[0], names[len(names) // 2], names[-1]]
    for suffix in ("q_proj", "gate_proj", "up_proj", "down_proj", "o_proj"):
        cand = [n for n in names if n.endswith(suffix)]
        if cand:
            picks.append(cand[len(cand) // 2])
    picks = list(dict.fromkeys(picks))[: args.obj_check_layers]
    obj = analysis_objective(inputs, energy, args.model, device, picks)

    print("[C/D] NMSE ladder: standard -> per-layer -> per-block -> oracle", flush=True)
    ladder = analysis_ladder(inputs, energy, None, device)

    raw = {"generalization": gen, "objective": obj, "ladder": ladder}
    (out_dir / "raw_metrics.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")

    m = ladder["mean"]
    lines = [
        "# 激活标定诊断",
        "",
        "## A. 校准→验证泛化",
        "",
        f"- 层数：{gen['layers']}；验证集改善层数：{gen['val_improved']}；"
        f"验证集未改善：{gen['val_worse_or_equal']}",
        f"- 平均验证集输出 MSE 改善：`{gen['mean_val_improvement']:.6e}`",
        f"- 校准改善但验证未改善的层数：{len(gen['cal_positive_val_nonpositive'])}",
        "",
        "## B. 标定目标函数检验（对角近似 vs 真实输出 MSE）",
        "",
        "| layer | diag pick | real pick | 一致 | val real MSE (diag pick) | (real pick) | (standard) |",
        "| --- | --- | --- | :-: | ---: | ---: | ---: |",
    ]
    for r in obj["layers"]:
        lines.append(
            f"| `{r['layer']}` | {tuple(round(v, 3) for v in r['diag_pick'])} | "
            f"{tuple(round(v, 3) for v in r['real_pick'])} | "
            f"{'Y' if r['same_pick'] else 'N'} | {r['val_real_diag_pick']:.4e} | "
            f"{r['val_real_real_pick']:.4e} | {r['val_real_standard']:.4e} |"
        )
    lines += [
        "",
        "## C/D. NMSE 阶梯（验证集，128 层平均）",
        "",
        "| 方案 | 说明 | mean NMSE | mean 加权输出 MSE |",
        "| --- | --- | ---: | ---: |",
        f"| standard (7,4,2) | 解析阈值 | {m['nmse_standard']:.6e} | {m['wmse_standard']:.6e} |",
        f"| per-layer 标定 | 每层一组 (d,t8,t4) | {m['nmse_per_layer']:.6e} | {m['wmse_per_layer']:.6e} |",
        f"| per-block 标定 | 每层每 block 一组 (d,t8,t4)，可离线存储 | {m['nmse_per_block']:.6e} | {m['wmse_per_block']:.6e} |",
        f"| oracle | 每行每 block 搜索（在线搜索上限） | {m['nmse_oracle']:.6e} | {m['wmse_oracle']:.6e} |",
        "",
        "逐层明细见 raw_metrics.json。",
    ]
    (out_dir / "diagnosis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
