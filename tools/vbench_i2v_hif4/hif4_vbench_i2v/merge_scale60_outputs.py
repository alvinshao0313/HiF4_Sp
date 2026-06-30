"""合并多个维度 shard 生成目录，得到一个 seed 的 scale60 输出目录。"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .utils import read_json, write_json


def main() -> None:
    ap = argparse.ArgumentParser(description="合并 HiF4 scale60 shard 输出")
    ap.add_argument("--shard", action="append", required=True, help="格式：dim=/path/to/generated")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--full-info", action="append", default=[], help="可选：每个 shard 对应的 full_info json")
    ap.add_argument("--merged-full-info", default=None)
    ap.add_argument("--copy-mode", choices=["physical", "hardlink"], default="physical")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    seen = set()
    total = 0
    for item in args.shard:
        if "=" not in item:
            raise SystemExit(f"--shard 需要 dim=path 格式: {item}")
        dim, path = item.split("=", 1)
        src_dir = Path(path)
        if not src_dir.is_dir():
            raise SystemExit(f"missing shard dir: {src_dir}")
        files = sorted(src_dir.glob("*.mp4"))
        print(f"SHARD {dim}: {len(files)} mp4")
        for f in files:
            if f.name in seen:
                raise SystemExit(f"重复文件名: {f.name}")
            seen.add(f.name)
            dst = out / f.name
            if args.copy_mode == "hardlink":
                if dst.exists():
                    dst.unlink()
                dst.hardlink_to(f)
            else:
                shutil.copy2(f, dst)
            total += 1

    print(f"MERGED_MP4_COUNT={total}")

    if args.full_info and args.merged_full_info:
        merged = []
        for p in args.full_info:
            data = read_json(p)
            merged.extend(list(data.values()) if isinstance(data, dict) else list(data))
        write_json(args.merged_full_info, merged)
        print(f"MERGED_FULL_INFO={args.merged_full_info} len={len(merged)}")


if __name__ == "__main__":
    main()
