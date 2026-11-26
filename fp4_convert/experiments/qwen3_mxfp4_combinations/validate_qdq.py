"""TP2 zero/midpoint/basis checks for all four fused quantizers."""
import argparse
from pathlib import Path
import torch
import torch.distributed as dist

from .artifacts import code_hashes, save
from .format import VARIANTS
from .validate_gpu import check_qdq, setup


def run(output):
    rank = setup()
    results = []
    for variant in VARIANTS:
        for k, sharded in ((2048, False), (384, True)):
            torch.manual_seed(1234 + rank if sharded else 1234)
            x = torch.randn(64, k, device="cuda", dtype=torch.bfloat16)
            x[0].zero_()
            x[1] = -0.
            x[2, :16] = torch.tensor([7.25, .25, .75, 1.25, 1.75, 2.5, 3.5, 5.,
                                     -7.25, -.25, -.75, -1.25, -1.75, -2.5, -3.5, -5.], device="cuda")
            x[3:35].zero_()
            x[3:35, :32] = torch.eye(32, device="cuda", dtype=torch.bfloat16)
            results.append({"variant": variant, "k": k, **check_qdq(x, variant, rank, sharded)})
        if rank == 0:
            print(f"{variant}: TP2 QDQ boundaries passed", flush=True)
    all_results = [None, None] if rank == 0 else None
    dist.gather_object(results, all_results, dst=0)
    if rank == 0:
        save(Path(output), {"status": "passed", "ranks": all_results, "source_sha256": code_hashes()})
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    run(parser.parse_args().output)
