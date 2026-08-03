"""Run reasoning benchmarks (AIME25 avg@5 + LiveCodeBench) for e2e scheme ckpts.

Reuses the weight-quantized HF checkpoints under <e2e-root>/<scheme>/tmp_ckpt/<scheme>.
vLLM + lighteval via repo-root main.py, thinking enabled, fake_act_quant=hif4.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT.parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2e-root", type=str, required=True)
    parser.add_argument("--schemes", type=str, required=True)
    parser.add_argument(
        "--datasets", type=str, default="aime25_avg5|0,lcb:codegeneration_v6|0"
    )
    parser.add_argument("--gpu", type=str, required=True)
    args = parser.parse_args()

    root = Path(args.e2e_root)
    for scheme in [s.strip() for s in args.schemes.split(",") if s.strip()]:
        ckpt = (root / scheme / "tmp_ckpt" / scheme).resolve()
        if not (ckpt / "config.json").is_file():
            raise FileNotFoundError(f"missing ckpt: {ckpt}")
        out_dir = (root / scheme / "reasoning" / scheme).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(_REPO / "main.py"),
            "--model_path", str(ckpt),
            "--datasets", args.datasets,
            "--tensor_parallel_size", "1",
            "--max_model_length", "32768",
            "--max_new_tokens", "32768",
            "--temperature", "0.7",
            "--top_p", "0.8",
            "--top_k", "20",
            "--gpu_memory_utilization", "0.9",
            "--fake_act_quant", "hif4",
            "--fake_act_quant_exclude", "lm_head",
            "--output_dir", str(out_dir),
        ]
        print(f"=== {scheme} reasoning eval on {ckpt} (gpu {args.gpu}) ===", flush=True)
        print("  cmd:", " ".join(cmd), flush=True)
        stdout_path = out_dir / "reasoning_stdout.log"
        stderr_path = out_dir / "reasoning_stderr.log"
        with stdout_path.open("w", encoding="utf-8") as out_f, stderr_path.open(
            "w", encoding="utf-8"
        ) as err_f:
            proc = subprocess.run(
                cmd,
                cwd=str(_REPO),
                stdout=out_f,
                stderr=err_f,
                text=True,
                check=False,
            )
        if proc.returncode != 0:
            raise RuntimeError(f"reasoning eval failed rc={proc.returncode}; see {stderr_path}")
        files = sorted(out_dir.rglob("results_*.json"))
        if not files:
            raise RuntimeError(f"no results json under {out_dir}")
        results = json.loads(files[-1].read_text(encoding="utf-8"))
        print(f"  results: {json.dumps(results.get('results', results), indent=2)}", flush=True)


if __name__ == "__main__":
    main()
