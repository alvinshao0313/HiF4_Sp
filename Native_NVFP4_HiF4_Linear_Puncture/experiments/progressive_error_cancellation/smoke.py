"""Small mathematical and optional real-path smoke checks."""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch

from .config import Config
from .directions import build_direction
from .jvp import toy_jvp
from .losses import absolute_mse, batch_cosine, decompose_errors


def mathematical_smoke():
    x = torch.tensor([[[1., 2.], [3., 4.]]])
    target = torch.zeros_like(x)
    valid = torch.ones(1, 2, dtype=torch.bool)
    assert absolute_mse(x, target, valid).item() == 7.5
    loss, active, _, _ = batch_cosine(x, -x, valid)
    assert active and abs(float(loss)) < 1e-6
    p = torch.ones_like(x)
    q = -0.25 * p
    result = decompose_errors(p + q, p, torch.zeros_like(p), valid)
    assert torch.allclose(result["cumulative"], result["propagated"] + result["local"])
    W = torch.tensor([[2., 1.], [-1., 3.]])
    tangent = torch.tensor([[4., -2.]])
    assert torch.allclose(toy_jvp(W, x[:, :1], tangent.unsqueeze(0)), tangent.unsqueeze(0) @ W.T)
    return {"status": "PASS", "checks": ["mse", "cosine", "decomposition", "toy_jvp"]}


def real_smoke(output_dir, *, layers="0:2", train_samples=4, val_samples=2,
               holdout_samples=2, epochs=2, method="direct", lambda_direction=.1):
    """Run the real training entrypoint with a reduced protocol.

    This intentionally remains an explicit command; it never starts a formal
    48-layer run implicitly.
    """
    from .train import train
    root = Path(output_dir).resolve()
    cases = (("direct", .1), ("jvp", .1), ("shuffled", .3))
    completed = []
    for case_method, case_lambda in cases:
        cfg = Config(output_dir=str(root / case_method), method=case_method,
                     lambda_direction=case_lambda, layers=layers, epochs=epochs,
                     calib_nsamples=train_samples, calib_val_nsamples=val_samples,
                     calib_holdout_nsamples=holdout_samples)
        cfg.validate()
        train(cfg)
        completed.append(case_method)
    return {"status": "PASS", "output_dir": str(root), "methods": completed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir")
    parser.add_argument("--real", action="store_true")
    args = parser.parse_args()
    if args.real:
        if not args.output_dir:
            raise SystemExit("--output_dir is required with --real")
        print(real_smoke(args.output_dir))
    else:
        print(mathematical_smoke())
