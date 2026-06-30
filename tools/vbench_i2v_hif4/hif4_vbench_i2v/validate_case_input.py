"""验证 VBench-I2V case input 是否满足基本结构和官方 repeat 语义。"""

from __future__ import annotations

import argparse
from pathlib import Path

from .constants import CAM_GROUP, SB_GROUP
from .utils import count_symlinks, identical_repeat_groups, list_mp4, validate_repeat_layout


def main() -> None:
    ap = argparse.ArgumentParser(description="验证 VBench-I2V case input")
    ap.add_argument("--case-input", required=True)
    ap.add_argument("--expected-sb", type=int, default=200)
    ap.add_argument("--expected-camera", type=int, default=100)
    ap.add_argument("--mode", choices=["quant", "bf16", "both"], default="quant")
    ap.add_argument("--forbid-symlink", action="store_true")
    ap.add_argument("--expected-repeats", type=int, default=5, help="每个 prompt base 期望的 exact repeat 数；设 0 可跳过 repeat 布局检查")
    ap.add_argument(
        "--allow-identical-repeat-files",
        action="store_true",
        help="允许同一 prompt 的 5 个 repeat 文件 SHA256 完全相同。默认不允许，因为这通常意味着把单个视频复制成了 5 份。",
    )
    args = ap.parse_args()

    root = Path(args.case_input)
    if not root.is_dir():
        raise SystemExit(f"missing case_input: {root}")

    dirs = []
    if args.mode in {"quant", "both"}:
        dirs.extend([
            (root / SB_GROUP / "videos_quant_sb", args.expected_sb),
            (root / CAM_GROUP / "videos_quant_camera", args.expected_camera),
        ])
    if args.mode in {"bf16", "both"}:
        dirs.extend([
            (root / SB_GROUP / "videos_bf16_sb", args.expected_sb),
            (root / CAM_GROUP / "videos_bf16_camera", args.expected_camera),
        ])

    ok = True
    for d, expected in dirs:
        c = len(list_mp4(d))
        print(f"{d}: mp4_count={c} expected={expected}")
        if c != expected:
            ok = False
        if args.expected_repeats:
            repeat_errors = validate_repeat_layout(d, args.expected_repeats)
            if repeat_errors:
                ok = False
                print(f"{d}: repeat_layout_errors={len(repeat_errors)}")
                for item in repeat_errors[:20]:
                    print(f"  REPEAT_ERROR {item}")
            if not args.allow_identical_repeat_files:
                identical = identical_repeat_groups(d, args.expected_repeats)
                if identical:
                    ok = False
                    print(f"{d}: identical_repeat_file_errors={len(identical)}")
                    for item in identical[:20]:
                        print(f"  IDENTICAL_REPEAT_ERROR {item}")
                    print("  提示：请确认生成阶段为每个 prompt 真实采样 5 次，而不是复制同一个 mp4。")

    links = count_symlinks(root)
    print(f"symlink_count={links}")
    if args.forbid_symlink and links:
        ok = False

    if ok:
        print("VALIDATE_CASE_INPUT_OK")
    else:
        raise SystemExit("VALIDATE_CASE_INPUT_FAILED")


if __name__ == "__main__":
    main()
