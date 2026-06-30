"""VBench 运行时兼容补丁。

这些补丁尽量保持温和：只有在必要时才修改运行时行为，用于解决不同 torch / torchvision
版本下的 VBench 兼容问题。
"""

from __future__ import annotations


def patch_torch_load_weights_only() -> None:
    """兼容 PyTorch 新版本 torch.load 默认 weights_only 的变化。"""
    try:
        import torch
    except Exception:
        return

    if getattr(torch.load, "_hif4_vbench_patched", False):
        return

    old_load = torch.load

    def patched_load(*args, **kwargs):  # type: ignore[no-untyped-def]
        # 某些 VBench 外部模型 checkpoint 需要 weights_only=False。
        kwargs.setdefault("weights_only", False)
        return old_load(*args, **kwargs)

    patched_load._hif4_vbench_patched = True  # type: ignore[attr-defined]
    torch.load = patched_load  # type: ignore[assignment]


def apply_all_patches() -> None:
    patch_torch_load_weights_only()
