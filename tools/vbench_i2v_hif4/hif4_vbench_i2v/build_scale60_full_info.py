"""从 VBench full_info 中抽取 scale60 子集。"""

from __future__ import annotations

import argparse
from pathlib import Path

from .utils import read_json, write_json


def get_dims(item) -> list[str]:  # type: ignore[no-untyped-def]
    for k in ["dimension", "dimensions", "dim"]:
        v = item.get(k) if isinstance(item, dict) else None
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            return [str(x) for x in v]
    return []


def main() -> None:
    ap = argparse.ArgumentParser(description="构建每个维度的 scaleN full_info")
    ap.add_argument("--full-info-json", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dims", default="i2v_subject i2v_background camera_motion")
    ap.add_argument("--start-offset", type=int, default=10)
    ap.add_argument("--n-per-dim", type=int, default=20)
    args = ap.parse_args()

    data = read_json(args.full_info_json)
    items = list(data.values()) if isinstance(data, dict) else list(data)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_picked = []
    for dim in args.dims.replace(",", " ").split():
        rows = [it for it in items if dim in get_dims(it)]
        picked = rows[args.start_offset: args.start_offset + args.n_per_dim]
        if len(picked) != args.n_per_dim:
            raise SystemExit(f"维度 {dim} 条目不足：got={len(picked)} expected={args.n_per_dim}")
        p = out / f"{dim}_start{args.start_offset}_n{args.n_per_dim}_full_info.json"
        write_json(p, picked)
        print(f"WROTE {p} len={len(picked)}")
        all_picked.extend(picked)

    merged = out / f"scale{len(all_picked)}_start{args.start_offset}_n{args.n_per_dim}_full_info.json"
    write_json(merged, all_picked)
    print(f"WROTE {merged} len={len(all_picked)}")


if __name__ == "__main__":
    main()
