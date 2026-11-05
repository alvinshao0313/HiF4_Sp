"""Run simultaneous real TP2 short checks on the two additional GPU pairs."""
import argparse
import os
from pathlib import Path

from ..reasoning_trace_calibration.qualify import inference
from ..reasoning_trace_full_kl_ce.artifacts import read_json, sha256, verify, write_json
from ..reasoning_trace_full_kl_ce.config import CANDIDATES


def qualify(root, output):
    root, _ = verify(root)
    pair = os.environ["CUDA_VISIBLE_DEVICES"]
    if pair not in ("0,1", "4,5"):
        raise ValueError("qualification is restricted to the two additional GPU pairs")
    candidate = CANDIDATES["full_kl"]
    selected = read_json(root / "development/full_kl/selection.json")
    model = root / "models" / candidate
    if not selected["complete"] or selected["export_sha256"] != sha256(model / "export.json"):
        raise ValueError("KL checkpoint must already be locked")
    models = output / "models"
    models.mkdir(parents=True, exist_ok=False)
    (models / candidate).symlink_to(model, target_is_directory=True)
    inference(output, candidate, qualification=False)
    result = read_json(output / "inference" / candidate / "metrics.json")
    if not result["passed"] or result["tensor_parallel_size"] != 2:
        raise ValueError("real TP2 prefill/decode check failed")
    write_json({"passed": True, "gpus": pair, "export_sha256": selected["export_sha256"],
                "scope": "two concurrent real TP2 engines; unchanged context, allocation and sampling except 64 decode steps"},
               output / "complete.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    qualify(args.run, args.output)
