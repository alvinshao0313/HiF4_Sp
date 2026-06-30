"""通用工具函数。代码注释使用中文，便于直接提交到中文 README 对应的工具包。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在，并返回 Path。"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def list_mp4(path: str | Path) -> list[Path]:
    p = Path(path)
    if not p.exists():
        return []
    return sorted(p.glob("*.mp4"))


def norm_video_base(name: str | Path) -> str:
    """将 'xxx-0.mp4'/'xxx-4.mp4' 归一成 'xxx'，用于 repeat 映射。"""
    stem = Path(name).stem
    return re.sub(r"-[0-9]+$", "", stem)


def build_video_index(dirs: Iterable[str | Path]) -> dict[str, Path]:
    """按归一化 base 建立 mp4 索引。

    仅保留给旧诊断脚本使用。VBench-I2V 正式输入构建必须使用
    :func:`build_video_name_index`，避免把单个 ``base-0.mp4`` 误映射到
    同一 prompt 的 5 个 repeat。
    """
    idx: dict[str, Path] = {}
    for d in dirs:
        d = Path(d)
        if not d.exists():
            continue
        for p in sorted(d.glob("*.mp4")):
            b = norm_video_base(p.name)
            if b in idx:
                raise RuntimeError(f"重复视频 base={b!r}: {idx[b]} 与 {p}")
            idx[b] = p
    return idx


def build_video_name_index(dirs: Iterable[str | Path]) -> dict[str, Path]:
    """按完整文件名建立 mp4 索引；重复文件名会报错。

    VBench-I2V 官方采样协议要求每个 prompt 有 ``-0`` 到 ``-4`` 五个
    独立采样结果。因此输入构建时必须 exact filename match，不能按
    prompt base 复用同一个视频。
    """
    idx: dict[str, Path] = {}
    for d in dirs:
        d = Path(d)
        if not d.exists():
            continue
        for p in sorted(d.glob("*.mp4")):
            key = p.name
            if key in idx:
                raise RuntimeError(f"重复视频文件名={key!r}: {idx[key]} 与 {p}")
            idx[key] = p
    return idx


def copy_file(src: Path, dst: Path, mode: str = "physical") -> None:
    """复制文件。默认 physical copy；可选 hardlink/reflink/symlink。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "physical":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "symlink":
        os.symlink(src, dst)
    elif mode == "reflink":
        # Linux 上优先使用 CoW reflink；文件系统不支持时由 cp --reflink=auto 退化为普通复制。
        try:
            subprocess.run(["cp", "--reflink=auto", "--preserve=all", str(src), str(dst)], check=True)
        except Exception:
            shutil.copy2(src, dst)
    else:
        raise ValueError(f"未知 copy_mode: {mode}")


def copy_tree_physical(src: Path, dst: Path) -> None:
    """物理复制目录，解析 symlink 内容，避免 VBench scratch 中 symlink 失效。"""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=False)


def count_symlinks(path: str | Path) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    return sum(1 for x in p.rglob("*") if x.is_symlink())


def first_number(x: Any) -> float | None:
    """从 VBench 结果 JSON 中尽量提取第一个数值。"""
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, list):
        for v in x:
            n = first_number(v)
            if n is not None:
                return n
    if isinstance(x, dict):
        for k in ["score", "mean", "value", "avg", "average"]:
            if k in x:
                n = first_number(x[k])
                if n is not None:
                    return n
        for v in x.values():
            n = first_number(v)
            if n is not None:
                return n
    return None


def extract_score_from_file(path: str | Path, dim: str | None = None) -> float:
    data = read_json(path)
    if dim and isinstance(data, dict) and dim in data:
        n = first_number(data[dim])
        if n is not None:
            return n
    if isinstance(data, dict) and len(data) == 1:
        n = first_number(next(iter(data.values())))
        if n is not None:
            return n
    n = first_number(data)
    if n is None:
        raise ValueError(f"无法从 {path} 提取数值分数")
    return n


def parse_dims(value: str | list[str] | None, default: list[str]) -> list[str]:
    if value is None or value == "all":
        return list(default)
    if isinstance(value, list):
        return value
    return [x for x in value.replace(",", " ").split() if x]


def repeat_groups(path: str | Path) -> dict[str, list[int]]:
    """返回 {base: [repeat_id, ...]}，用于检查 exact repeat 是否完整。"""
    groups: dict[str, list[int]] = {}
    for p in list_mp4(path):
        m = re.match(r"^(.*)-([0-9]+)\.mp4$", p.name)
        if not m:
            groups.setdefault(p.stem, []).append(-1)
            continue
        base, idx = m.group(1), int(m.group(2))
        groups.setdefault(base, []).append(idx)
    return {k: sorted(v) for k, v in groups.items()}


def validate_repeat_layout(path: str | Path, expected_repeats: int = 5) -> list[str]:
    """检查每个 prompt base 是否拥有 0..expected_repeats-1 的 repeat 文件。"""
    errors: list[str] = []
    target = list(range(expected_repeats))
    for base, ids in repeat_groups(path).items():
        if ids != target:
            errors.append(f"base={base!r} repeats={ids} expected={target}")
    return errors


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """计算文件 SHA256，用于发现把同一个视频复制成多个 repeat 的情况。"""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def identical_repeat_groups(path: str | Path, expected_repeats: int = 5) -> list[str]:
    """返回疑似重复复制的 prompt base。

    如果某个 base 拥有完整 ``0..expected_repeats-1`` repeat，但所有文件
    SHA256 完全相同，通常意味着用户把一个生成结果复制成了多个 repeat。
    这违反 VBench-I2V 的官方采样语义。
    """
    p = Path(path)
    problems: list[str] = []
    for base, ids in repeat_groups(p).items():
        if ids != list(range(expected_repeats)):
            continue
        files = [p / f"{base}-{i}.mp4" for i in range(expected_repeats)]
        hashes = [sha256_file(x) for x in files]
        if len(set(hashes)) == 1:
            problems.append(f"base={base!r} all_{expected_repeats}_repeat_files_have_same_sha256={hashes[0]}")
    return problems
