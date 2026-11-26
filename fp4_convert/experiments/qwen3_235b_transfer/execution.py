"""Store backward tensors on two peer GPUs without changing training math."""
from __future__ import annotations

import os

import torch


COMPUTE_DEVICE = torch.device("cuda:0")
STORAGE_DEVICES = (torch.device("cuda:1"), torch.device("cuda:2"))
DEVICES = (COMPUTE_DEVICE, *STORAGE_DEVICES)


def _copy_with_strides(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    copied = tensor.detach().to(
        device=device, non_blocking=True, memory_format=torch.preserve_format,
    )
    if copied.stride() == tensor.stride():
        return copied
    shape = list(tensor.shape)
    for index, (length, stride) in enumerate(zip(tensor.shape, tensor.stride())):
        if length > 1 and stride == 0:
            shape[index] = 1
    # A broadcast view stores one representative along each zero-stride axis.
    source = copied
    for index, length in enumerate(shape):
        if source.shape[index] != length:
            source = source.narrow(index, 0, length)
    restored = torch.empty_strided(
        shape, tensor.stride(), device=device, dtype=tensor.dtype,
    )
    restored.copy_(source, non_blocking=True)
    return restored.as_strided(tensor.shape, tensor.stride())


class SavedTensorsOnGpus(torch.autograd.graph.saved_tensors_hooks):
    """Compute on cuda:0, storing saved CUDA tensors round-robin on cuda:1/2."""

    def __init__(self) -> None:
        if not torch.cuda.is_available() or torch.cuda.device_count() < 3:
            raise RuntimeError("235B training requires three visible CUDA devices")
        for storage in STORAGE_DEVICES:
            if not (torch.cuda.can_device_access_peer(0, storage.index)
                    and torch.cuda.can_device_access_peer(storage.index, 0)):
                raise RuntimeError(f"CUDA peer access is required between cuda:0 and {storage}")
        self.packed_count = 0
        self.packed_bytes = [0, 0]
        self.restored_count = 0
        super().__init__(self._pack, self._unpack)

    def _pack(self, tensor: torch.Tensor) -> tuple[torch.device, torch.Tensor]:
        if tensor.device.type != "cuda":
            return tensor.device, tensor.detach()
        if tensor.device != COMPUTE_DEVICE:
            raise ValueError(f"unexpected saved tensor origin {tensor.device}")
        slot = self.packed_count % len(STORAGE_DEVICES)
        packed = _copy_with_strides(tensor, STORAGE_DEVICES[slot])
        if packed.dtype != tensor.dtype or packed.stride() != tensor.stride():
            raise ValueError("GPU saved tensor dtype or stride changed")
        self.packed_count += 1
        self.packed_bytes[slot] += tensor.numel() * tensor.element_size()
        return tensor.device, packed

    def _unpack(self, packed: tuple[torch.device, torch.Tensor]) -> torch.Tensor:
        original_device, tensor = packed
        if original_device.type != "cuda":
            return tensor
        restored = _copy_with_strides(tensor, original_device)
        if restored.dtype != tensor.dtype or restored.stride() != tensor.stride():
            raise ValueError("unpacked tensor dtype or stride changed")
        self.restored_count += 1
        return restored

    def metadata(self) -> dict:
        return {
            "saved_tensors_device": "cuda",
            "compute_device": str(COMPUTE_DEVICE),
            "storage_devices": [str(device) for device in STORAGE_DEVICES],
            "storage_policy": "round_robin",
            "preserve_dtype": True,
            "preserve_strides": True,
            "non_blocking": True,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "packed_count": self.packed_count,
            "packed_bytes": list(self.packed_bytes),
            "restored_count": self.restored_count,
        }


def reset_memory_peaks() -> None:
    for device in DEVICES:
        torch.cuda.reset_peak_memory_stats(device)


def memory_peaks() -> dict[str, dict[str, int]]:
    return {
        str(device): {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
        for device in DEVICES
    }
