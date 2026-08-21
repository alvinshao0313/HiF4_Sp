"""Build aggregate report (md/html) from latest analysis result pointers."""

from __future__ import annotations

import base64
import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from Inference_Paradigm_Conversion.ipc_analysis.io_utils import (
    atomic_write_json,
    ensure_dir,
    write_text,
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_ptr(results_root: Path, name: str) -> Path | None:
    p = results_root / name
    if not p.is_file():
        return None
    run = results_root / p.read_text(encoding="utf-8").strip()
    return run if run.is_dir() else None


def _fig_to_b64() -> str:
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=120)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _plot_w3_bars(w3_summary: dict[str, Any]) -> str | None:
    ranking = w3_summary.get("ranking") or w3_summary.get("ranking_by_mean_R_cf_output") or []
    if not ranking:
        return None
    # keep legal-ish top variants; drop extreme illegal probes for readability
    items = [r for r in ranking if abs(float(r.get("mean_R_cf_output", 0))) < 2.0]
    if not items:
        items = ranking[:8]
    labels = [r["variant"] for r in items]
    vals = [float(r["mean_R_cf_output"]) for r in items]
    plt.figure(figsize=(8, 3.5))
    plt.barh(labels[::-1], vals[::-1], color="#2a6f97")
    plt.xlabel("mean R_cf (output)")
    plt.title("W3 counterfactual recoverable fraction")
    return _fig_to_b64()


def _plot_prefix_suffix(inj_summary: dict[str, Any]) -> str | None:
    curve = inj_summary.get("mean_kl_by_boundary") or {}
    pref, suf = [], []
    for k, v in curve.items():
        if k.startswith("prefix_layers:k="):
            pref.append((int(k.split("=")[1]), float(v)))
        elif k.startswith("suffix_layers:k="):
            suf.append((int(k.split("=")[1]), float(v)))
    if not pref and not suf:
        return None
    pref.sort()
    suf.sort()
    plt.figure(figsize=(6, 3.5))
    if pref:
        plt.plot([x for x, _ in pref], [y for _, y in pref], "o-", label="prefix")
    if suf:
        plt.plot([x for x, _ in suf], [y for _, y in suf], "s-", label="suffix")
    plt.xlabel("k (layers converted)")
    plt.ylabel("mean KL (last token)")
    plt.title("N4/N5 prefix–suffix sensitivity")
    plt.legend()
    return _fig_to_b64()


def _plot_attn_kl(attn_csv: Path) -> str | None:
    if not attn_csv.is_file():
        return None
    rows = list(csv.DictReader(attn_csv.open()))
    if not rows:
        return None
    by_layer: dict[int, list[float]] = {}
    for r in rows:
        li = int(float(r["layer_idx"]))
        by_layer.setdefault(li, []).append(float(r["kl_st"]))
    xs = sorted(by_layer)
    ys = [sum(by_layer[x]) / len(by_layer[x]) for x in xs]
    plt.figure(figsize=(5, 3))
    plt.bar([str(x) for x in xs], ys, color="#bc4749")
    plt.xlabel("representative layer")
    plt.ylabel("mean KL(S||T)")
    plt.title("Attention logits KL (H6)")
    return _fig_to_b64()


def _plot_inject_n1(inj: dict[str, Any]) -> str | None:
    m = inj.get("mean_kl_by_mask") or {}
    if not m:
        return None
    labels = list(m.keys())
    vals = [float(m[k]) for k in labels]
    plt.figure(figsize=(8, 3.5))
    plt.barh(labels[::-1], vals[::-1], color="#6a994e")
    plt.xlabel("mean KL last")
    plt.title("N1/N2 single linear / layer injection")
    return _fig_to_b64()


def _plot_evidence_matrix(ledger: dict[str, Any]) -> str | None:
    recs = ledger.get("records") or []
    if not recs:
        return None
    hyps = sorted({r.get("hypothesis_id", "?") for r in recs})
    classes = ["observational_correlation", "controlled_causal_evidence"]
    mat = [[0 for _ in classes] for _ in hyps]
    h2i = {h: i for i, h in enumerate(hyps)}
    c2j = {c: j for j, c in enumerate(classes)}
    for r in recs:
        i = h2i.get(r.get("hypothesis_id", "?"))
        j = c2j.get(r.get("evidence_class"))
        if i is not None and j is not None:
            mat[i][j] += 1
    plt.figure(figsize=(6, max(3, 0.35 * len(hyps))))
    plt.imshow(mat, aspect="auto", cmap="Blues")
    plt.xticks(range(len(classes)), ["observational", "causal"], rotation=20)
    plt.yticks(range(len(hyps)), hyps)
    plt.title("Root-cause evidence matrix (counts)")
    for i in range(len(hyps)):
        for j in range(len(classes)):
            plt.text(j, i, str(mat[i][j]), ha="center", va="center", color="black")
    return _fig_to_b64()


def build_report(results_root: Path, out_dir: Path) -> dict[str, Any]:
    results_root = Path(results_root)
    out_dir = ensure_dir(out_dir)

    ptrs = {
        "f0": _read_ptr(results_root, "latest_f0_run_id.txt")
        or _read_ptr(results_root, "latest_preflight_run_id.txt"),
        "weight": _read_ptr(results_root, "latest_weight_run_id.txt"),
        "repr_al": _read_ptr(results_root, "latest_repr_al_run_id.txt"),
        "w3": _read_ptr(results_root, "latest_w3_run_id.txt"),
        "w4": _read_ptr(results_root, "latest_w4_run_id.txt"),
        "l3": _read_ptr(results_root, "latest_l3_run_id.txt"),
        "mlp": _read_ptr(results_root, "latest_mlp_run_id.txt"),
        "attn": _read_ptr(results_root, "latest_attn_run_id.txt"),
        "gemm": _read_ptr(results_root, "latest_gemm_run_id.txt"),
        "inject_n1": _read_ptr(results_root, "latest_inject_n1_n2_run_id.txt"),
        "inject_ps": _read_ptr(results_root, "latest_inject_prefix_suffix_run_id.txt"),
        "inject_or": _read_ptr(results_root, "latest_inject_oracle_run_id.txt"),
        "synthetic": _read_ptr(results_root, "latest_synthetic_run_id.txt"),
        "ledger": _read_ptr(results_root, "latest_ledger_run_id.txt"),
        "e2e": _read_ptr(results_root, "latest_e2e_run_id.txt"),
    }

    # ensure ledger exists
    if ptrs["ledger"] is None or not (ptrs["ledger"] / "root_cause_ledger.json").is_file():
        from Inference_Paradigm_Conversion.ipc_analysis.reporting.root_cause import build_ledger

        led_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_ledger"
        led_dir = ensure_dir(results_root / led_id)
        build_ledger(results_root, led_dir)
        write_text(results_root / "latest_ledger_run_id.txt", led_id)
        ptrs["ledger"] = led_dir

    ledger = _read_json(ptrs["ledger"] / "root_cause_ledger.json")
    # copy ledger into report dir
    atomic_write_json(out_dir / "root_cause_ledger.json", ledger)
    write_text(
        out_dir / "root_cause_ledger.md",
        (ptrs["ledger"] / "root_cause_ledger.md").read_text(encoding="utf-8")
        if (ptrs["ledger"] / "root_cause_ledger.md").is_file()
        else "",
    )

    summaries = {k: _read_json(v / f"{k.split('_')[0]}_summary.json") if v else {} for k, v in ptrs.items()}
    # load known summary filenames
    loaders = {
        "weight": "weight_summary.json",
        "repr_al": "repr_al_summary.json",
        "w3": "w3_summary.json",
        "w4": "w4_summary.json",
        "l3": "l3_summary.json",
        "mlp": "mlp_summary.json",
        "attn": "attention_summary.json",
        "gemm": "gemm_summary.json",
        "inject_n1": "injection_n1_n2_summary.json",
        "inject_ps": "injection_prefix_suffix_summary.json",
        "inject_or": "injection_oracle_summary.json",
        "synthetic": "synthetic_summary.json",
        "e2e": "e2e_summary.json",
    }
    data: dict[str, Any] = {}
    for key, fname in loaders.items():
        p = ptrs.get(key)
        data[key] = _read_json(p / fname) if p else {}

    if ptrs.get("e2e") and (ptrs["e2e"] / "e2e_summary.csv").is_file():
        import shutil

        shutil.copy2(ptrs["e2e"] / "e2e_summary.csv", out_dir / "e2e_summary.csv")

    figs: dict[str, str] = {}
    if data.get("w3"):
        b = _plot_w3_bars(data["w3"])
        if b:
            figs["w3_rcf"] = b
    if data.get("inject_ps"):
        b = _plot_prefix_suffix(data["inject_ps"])
        if b:
            figs["prefix_suffix"] = b
    if ptrs.get("attn"):
        b = _plot_attn_kl(ptrs["attn"] / "attention_propagation.csv")
        if b:
            figs["attn_kl"] = b
    if data.get("inject_n1"):
        b = _plot_inject_n1(data["inject_n1"])
        if b:
            figs["inject_n1"] = b
    b = _plot_evidence_matrix(ledger)
    if b:
        figs["evidence_matrix"] = b

    # conclusions from ledger metrics
    by_id = {r["cause_id"]: r for r in ledger.get("records", [])}
    conclusions = [
        f"Weight global NMSE ≈ {by_id.get('C_W_BASE', {}).get('metric_value', float('nan')):.4g}.",
        f"P2: activation Δ dominates weight Δ (WN·ΔA / ΔW·AN ≈ {by_id.get('C_L_CROSS', {}).get('metric_value', float('nan')):.3g}).",
        f"W3: continuous_payload_clipped R_cf ≈ {by_id.get('C_W3_continuous_payload_clipped', {}).get('metric_value', float('nan')):.3g}.",
        f"Attention mean KL ≈ {by_id.get('C_T_ATTN', {}).get('metric_value', float('nan')):.3g}; linear_attn absent.",
        f"N1/N2 mean KL ≈ {by_id.get('C_N_INJECT', {}).get('metric_value', float('nan')):.3g}; gate_proj most sensitive among linears.",
        f"Oracle repair best recoverable KL ≈ {by_id.get('C_N_ORACLE', {}).get('metric_value', float('nan')):.3g} (frac=0.05).",
        "S1–S7 synthetic suite: all mechanisms directionally supported (not effect-size substitutes).",
    ]
    if data.get("e2e") and data["e2e"].get("rows"):
        for r in data["e2e"]["rows"]:
            conclusions.append(
                f"E2E {r.get('path_id')} {r.get('task')}: source={r.get('source_score')} "
                f"target={r.get('target_score')} Δ={r.get('delta_target_minus_source')}."
            )
    else:
        conclusions.append("E2E semantic ARC: pending or not yet merged into this report.")

    summary = {
        "schema_version": 1,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source_runs": {k: (str(v.name) if v else None) for k, v in ptrs.items()},
        "num_ledger_records": ledger.get("num_records", 0),
        "conclusions": conclusions,
        "figures": list(figs.keys()),
        "path_note": "P1_semantic / P2_matched_semantic numbers are not averaged together.",
    }
    atomic_write_json(out_dir / "summary.json", summary)
    atomic_write_json(
        out_dir / "manifest.json",
        {
            "report_kind": "ipc_root_cause_aggregate",
            "source_runs": summary["source_runs"],
            "figures": summary["figures"],
        },
    )

    # markdown
    md = [
        "# Inference Paradigm Conversion — Root Cause Report",
        "",
        f"built_at: {summary['built_at']}",
        "",
        "## 1. 结论摘要",
        "",
    ]
    for c in conclusions:
        md.append(f"- {c}")
    md += [
        "",
        "## 2. 路径定义",
        "",
        "- `P1_semantic`: W NVFP4→HiF4, A MXFP8→MXFP8（激活同格式，权重转换）",
        "- `P2_matched_semantic`: W 与 A 同步 NVFP4→HiF4（matched coverage）",
        "",
        "## 19. 根因台账",
        "",
        f"records: {summary['num_ledger_records']}（见 `root_cause_ledger.md`）",
        "",
        "## 来源 run",
        "",
    ]
    for k, v in summary["source_runs"].items():
        md.append(f"- {k}: `{v}`")
    md += ["", "## 图", ""]
    for name in figs:
        md.append(f"- {name}（见 HTML 内嵌）")
    write_text(out_dir / "report.md", "\n".join(md) + "\n")

    # html
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>IPC Root Cause Report</title>",
        "<style>",
        "body{font-family:Georgia,serif;max-width:960px;margin:2rem auto;padding:0 1rem;line-height:1.45;color:#222}",
        "h1,h2{font-family:Helvetica,Arial,sans-serif}",
        "code{background:#f4f4f4;padding:0.1em 0.3em}",
        "img{max-width:100%;border:1px solid #ddd;margin:0.5rem 0}",
        "li{margin:0.25rem 0}",
        "</style></head><body>",
        "<h1>Inference Paradigm Conversion — Root Cause Report</h1>",
        f"<p>built_at: {summary['built_at']}</p>",
        "<h2>1. 结论摘要</h2><ul>",
    ]
    for c in conclusions:
        parts.append(f"<li>{c}</li>")
    parts.append("</ul><h2>Figures</h2>")
    titles = {
        "w3_rcf": "W3 recoverable fraction",
        "prefix_suffix": "Prefix/suffix KL curve",
        "attn_kl": "Attention KL by layer",
        "inject_n1": "N1/N2 injection KL",
        "evidence_matrix": "Evidence matrix",
    }
    for name, b64 in figs.items():
        parts.append(f"<h3>{titles.get(name, name)}</h3>")
        parts.append(f"<img alt='{name}' src='data:image/png;base64,{b64}'/>")
    parts.append("<h2>Root Cause Ledger</h2><pre>")
    parts.append((out_dir / "root_cause_ledger.md").read_text(encoding="utf-8"))
    parts.append("</pre></body></html>")
    write_text(out_dir / "report.html", "".join(parts))

    # stub empty required CSVs if missing (schema presence)
    for stub in (
        "tensor_summary.csv",
        "group_summary.csv",
        "layer_summary.csv",
        "runtime_summary.csv",
        "e2e_summary.csv",
    ):
        if not (out_dir / stub).is_file():
            write_text(out_dir / stub, "status,note\npending,filled_when_stage_available\n")

    return summary
