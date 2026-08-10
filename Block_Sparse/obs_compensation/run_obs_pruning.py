#!/usr/bin/env python3
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

BLOCK_SPARSE_ROOT = Path(__file__).resolve().parents[1]
if str(BLOCK_SPARSE_ROOT) not in sys.path:
    sys.path.insert(0, str(BLOCK_SPARSE_ROOT))

from obs_compensation.artifacts import load_source_artifacts  # noqa: E402
from obs_compensation.config import build_config, parse_args  # noqa: E402
from obs_compensation.pipeline import run_obs_compensation  # noqa: E402
from obs_compensation.solver import resolve_obs_order_policy  # noqa: E402


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(argv: list[str] | None = None) -> int:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args(argv)
    artifacts = load_source_artifacts(args.source_artifacts_dir)
    config = build_config(args, artifacts.metadata.model_path)
    print(f"[obs] requested obs_order_policy={config.obs_order_policy}", flush=True)
    order_policy = resolve_obs_order_policy(
        requested_policy=config.obs_order_policy,
        mlp_permutation=artifacts.metadata.mlp_permutation,
    )
    print(
        f"[obs] requested_policy={order_policy.requested_policy} "
        f"resolved_policy={order_policy.resolved_policy} "
        f"gate_up_direction={order_policy.gate_up_direction} "
        f"down_direction={order_policy.down_direction}",
        flush=True,
    )
    _set_seed(config.seed)
    output = run_obs_compensation(config)
    print(f"[obs] done output_dir={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
