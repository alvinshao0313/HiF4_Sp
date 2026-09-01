"""Guard E2/E3-E7 evaluation from obsolete HiF4 runtime artifacts/results."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.evaluation.moe_materialize import (
    HIF4_RUNTIME_ABI_VERSION,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
SHARED_VLLM_ROOT = (
    REPO_ROOT
    / "Native_NVFP4_HiF4_Linear_Puncture"
    / "results"
    / "e2e_diag_reconstruction"
    / "shared_vllm_qwen3_30b"
)


def _guarded_variant(variant: str) -> bool:
    return variant in {"r64_only", "artifact"}


def _materialized_variant(variant: str, artifact_diag_variant: str) -> str:
    if variant == "r64_only":
        return "r64_only"
    if variant == "artifact":
        return (
            "artifact"
            if artifact_diag_variant == "adopted"
            else f"artifact_{artifact_diag_variant}"
        )
    raise ValueError(f"runtime ABI guard does not apply to variant={variant!r}")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _is_current_abi(payload: dict[str, Any] | None) -> bool:
    if payload is None:
        return False
    try:
        return int(payload.get("runtime_abi_version", -1)) == HIF4_RUNTIME_ABI_VERSION
    except (TypeError, ValueError):
        return False


def _quarantine(path: Path, label: str) -> Path:
    if not path.exists():
        return path
    suffix = f".pre_runtime_abi_v{HIF4_RUNTIME_ABI_VERSION}.{os.getpid()}"
    target = path.with_name(path.name + suffix)
    counter = 0
    while target.exists():
        counter += 1
        target = path.with_name(path.name + suffix + f".{counter}")
    path.rename(target)
    print(f"runtime ABI guard: quarantined {label}: {path} -> {target}", flush=True)
    return target


def _metric_paths(run_dir: Path) -> tuple[Path, ...]:
    return (
        run_dir / "eval" / "arc" / "metrics.json",
        run_dir / "eval" / "mmlu" / "metrics.json",
        run_dir / "eval" / "mmlu_pro" / "metrics.json",
        run_dir / "eval" / "aime25" / "metrics.json",
        run_dir / "eval" / "livecodebench" / "metrics.json",
        run_dir / "eval" / "eval_summary.json",
    )


def prepare(
    run_dir: Path,
    *,
    variant: str,
    artifact_diag_variant: str = "adopted",
) -> None:
    if not _guarded_variant(variant):
        return
    run_dir = run_dir.resolve()
    materialized_variant = _materialized_variant(variant, artifact_diag_variant)
    cache_dir = SHARED_VLLM_ROOT / run_dir.name / materialized_variant
    cache_marker = _read_json(cache_dir / "hif4_runtime_abi.json")
    if cache_dir.exists() and not _is_current_abi(cache_marker):
        _quarantine(cache_dir, "obsolete materialized checkpoint")

    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    guard = {
        "runtime_abi_required": True,
        "runtime_abi_version": HIF4_RUNTIME_ABI_VERSION,
        "variant": variant,
        "artifact_diag_variant": artifact_diag_variant,
        "policy": (
            "E2/E3-E7 results without the current optimized HiF4 runtime ABI "
            "must not be reused."
        ),
    }
    (eval_dir / "runtime_abi_guard.json").write_text(
        json.dumps(guard, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    for path in _metric_paths(run_dir):
        if path.is_file() and not _is_current_abi(_read_json(path)):
            _quarantine(path, "obsolete eval result")


def stamp(run_dir: Path) -> None:
    run_dir = run_dir.resolve()
    for path in _metric_paths(run_dir):
        payload = _read_json(path)
        if payload is None:
            continue
        payload["runtime_abi_version"] = HIF4_RUNTIME_ABI_VERSION
        payload["runtime_kernel_family"] = "optimized_hif4_triton_e2_e7"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    prepare_p = sub.add_parser("prepare")
    prepare_p.add_argument("--run_dir", required=True)
    prepare_p.add_argument("--variant", required=True, choices=("r64_only", "artifact"))
    prepare_p.add_argument(
        "--artifact_diag_variant", choices=("adopted", "candidate"), default="adopted"
    )
    stamp_p = sub.add_parser("stamp")
    stamp_p.add_argument("--run_dir", required=True)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        prepare(
            Path(args.run_dir),
            variant=args.variant,
            artifact_diag_variant=args.artifact_diag_variant,
        )
    else:
        stamp(Path(args.run_dir))


if __name__ == "__main__":
    main()
