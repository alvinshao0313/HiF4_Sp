from dataclasses import asdict, dataclass
import csv
import io
import os
from pathlib import Path
import subprocess


@dataclass(frozen=True)
class Protocol:
    model: str = "nvidia/Qwen3-30B-A3B-NVFP4"
    layers: tuple[int, ...] = (8, 24, 40)
    train_samples: int = 32
    val_samples: int = 16
    test_samples: int = 32
    seed: int = 42
    batch_size: int = 4
    lr: float = 1e-4
    budget_seconds: float = 7200.
    checkpoint_seconds: float = 1800.
    router_k: int = 8
    router_temperature: float = 1.
    router_weight: float = 1.
    refresh_epochs: tuple[int, ...] = (1, 2, 4)
    token_chunk: int = 256
    capture_chunk: int = 256
    prefill_chunk: int = 2048

    def to_dict(self):
        return asdict(self)


PROTOCOL = Protocol()


def recipes():
    return [dict(layer=l, objective=o, reuse_epochs=k,
                 name=f"L{l:02d}_{o}" + (f"_k{k}" if o == "cached_kl" else ""))
            for l in PROTOCOL.layers
            for o, k in [("mse", 0), ("cached_kl", 1), ("cached_kl", 2),
                         ("cached_kl", 4), ("direct_kl", 0)]]


def require_gpus(count: int):
    """Never choose a GPU or inherit an unspecified device allocation."""
    value = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    ids = [x.strip() for x in value.split(",") if x.strip()]
    if len(ids) != count or len(set(ids)) != count or any(x == "-1" for x in ids):
        raise RuntimeError(f"explicit CUDA_VISIBLE_DEVICES with {count} distinct GPU(s) required")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != count:
        raise RuntimeError("visible CUDA device count differs from the requested allocation")
    return ids


def require_environment():
    import sys
    if Path(sys.prefix).name != "hif4":
        raise RuntimeError("run this experiment in the hif4 conda environment")


def idle_devices(gpu_ids):
    """Check the requested physical GPUs immediately before a stage starts."""
    selected = gpu_ids.split(",")
    if not selected or len(set(selected)) != len(selected) or not all(s.isdigit() for s in selected):
        raise ValueError("provide distinct physical GPU indices")
    def query(fields, scope):
        output = subprocess.check_output(["nvidia-smi", f"--query-{scope}={fields}", "--format=csv,noheader,nounits"], text=True)
        return [[v.strip() for v in row] for row in csv.reader(io.StringIO(output))]
    inventory = query("index,uuid,name,memory.total", "gpu")
    devices = {r[0]: dict(index=r[0], uuid=r[1], name=r[2], memory_mib=int(r[3])) for r in inventory}
    if not set(selected) <= set(devices):
        raise RuntimeError("requested GPUs do not exist")
    selected_uuids = {devices[i]["uuid"] for i in selected}
    conflicts = [r for r in query("gpu_uuid,pid,process_name", "compute-apps") if r and r[0] in selected_uuids]
    if conflicts:
        raise RuntimeError(f"requested GPUs have active processes: {conflicts}")
    return [devices[i] for i in selected]
