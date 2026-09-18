"""Keep explicitly assigned GPUs reserved with a tiny CUDA allocation."""
import argparse
import json
import os
from pathlib import Path
import signal
import threading

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", type=int, default=None,
                        help="number of visible devices; physical mapping comes from CUDA_VISIBLE_DEVICES")
    parser.add_argument("--memory-mib", type=int, default=256)
    parser.add_argument('--ready-file', type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    count = torch.cuda.device_count()
    if args.devices is not None and count != args.devices:
        raise RuntimeError(f"expected {args.devices} visible CUDA devices, got {count}")
    if args.memory_mib <= 0:
        raise ValueError("memory-mib must be positive")
    allocations = []
    for index in range(count):
        with torch.cuda.device(index):
            value = torch.empty(args.memory_mib * 1024 * 1024 // 2,
                                dtype=torch.float16, device="cuda")
            # Touch the pages so nvidia-smi and other schedulers see a real
            # allocation rather than only a lazy virtual reservation.
            value.fill_(0)
            allocations.append(value)
    print(f"reserved {count} visible GPU(s), {args.memory_mib} MiB each", flush=True)
    stop = threading.Event()

    def terminate(signum, frame):
        stop.set()

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    for index in range(count):
        torch.cuda.synchronize(index)
    if args.ready_file:
        tmp = args.ready_file.with_suffix('.partial')
        tmp.write_text(json.dumps(dict(pid=os.getpid(), devices=count, memory_mib=args.memory_mib)))
        tmp.replace(args.ready_file)
    stop.wait()
    del allocations
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
