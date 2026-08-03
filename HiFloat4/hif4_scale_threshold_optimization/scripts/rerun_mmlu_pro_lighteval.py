"""Rerun only lighteval MMLU-Pro for schemes that already have mid ARC/MMLU + weight ckpt."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_MOD_PATH = _ROOT / "scripts" / "evaluate_model.py"


def _load_eval_mod():
    spec = importlib.util.spec_from_file_location("sto_evaluate_model", _MOD_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2e-root", type=str, required=True)
    parser.add_argument("--schemes", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=300)
    parser.add_argument("--gpu", type=str, default=None)
    args = parser.parse_args()

    em = _load_eval_mod()
    root = Path(args.e2e_root)
    for scheme in [s.strip() for s in args.schemes.split(",") if s.strip()]:
        scheme_dir = root / scheme
        ckpt = scheme_dir / "tmp_ckpt" / scheme
        if not (ckpt / "config.json").is_file():
            raise FileNotFoundError(f"missing ckpt: {ckpt}")
        mid = scheme_dir / f"{scheme}_mid_arc_mmlu.json"
        out = json.loads(mid.read_text(encoding="utf-8")) if mid.is_file() else {"scheme": scheme}
        mpro_dir = scheme_dir / "mmlu_pro" / scheme
        print(f"=== {scheme} lighteval on {ckpt.resolve()} ===", flush=True)
        out["mmlu_pro"] = em.run_lighteval_mmlu_pro(
            ckpt_dir=ckpt,
            output_dir=mpro_dir,
            max_samples=args.max_samples,
            gpu=args.gpu,
        )
        em.save_json(scheme_dir / f"{scheme}.json", out)
        em.save_json(scheme_dir / "raw_metrics.json", {scheme: out})
        print(f"  done: {out['mmlu_pro']}", flush=True)


if __name__ == "__main__":
    main()
