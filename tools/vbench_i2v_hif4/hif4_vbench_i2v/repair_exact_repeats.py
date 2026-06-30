"""重建 VBench-I2V exact-repeat 输入目录。

本脚本只从已经真实生成的 exact repeat 源视频中复制同名文件；它不会把
``base-0.mp4`` 扩展复制成 ``base-1.mp4`` ... ``base-4.mp4``。如果缺少某个
repeat，说明生成阶段不完整，应该回到生成脚本按不同 seed/index 补生成。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .build_eval_inputs import replace_from_template
from .constants import CAM_GROUP, SB_GROUP
from .utils import build_video_name_index, count_symlinks


def main() -> None:
    ap = argparse.ArgumentParser(description="按 strict exact-repeat 规则重建 HiF4 case input")
    ap.add_argument("--template-case", required=True)
    ap.add_argument("--case-input", required=True)
    ap.add_argument("--generated-dir", default=None)
    ap.add_argument("--subject-dir", default=None)
    ap.add_argument("--background-dir", default=None)
    ap.add_argument("--camera-dir", default=None)
    ap.add_argument("--copy-mode", choices=["physical", "hardlink", "symlink", "reflink"], default="physical")
    ap.add_argument("--forbid-symlink", action="store_true", default=True, help="默认禁止输出中残留 symlink")
    ap.add_argument("--allow-symlink", action="store_false", dest="forbid_symlink", help="允许 symlink，仅在确认 VBench scratch/Slurm 节点可访问同一路径时使用")
    args = ap.parse_args()

    template = Path(args.template_case)
    case_input = Path(args.case_input)
    if not template.is_dir():
        raise SystemExit(f"missing template: {template}")
    if not case_input.is_dir():
        raise SystemExit(f"missing case_input: {case_input}")

    sb_dirs = []
    cam_dirs = []
    if args.generated_dir:
        sb_dirs.append(Path(args.generated_dir))
        cam_dirs.append(Path(args.generated_dir))
    if args.subject_dir:
        sb_dirs.append(Path(args.subject_dir))
    if args.background_dir:
        sb_dirs.append(Path(args.background_dir))
    if args.camera_dir:
        cam_dirs.append(Path(args.camera_dir))
    if not sb_dirs or not cam_dirs:
        raise SystemExit("需要 --generated-dir 或 subject/background/camera 专用目录")

    sb_idx = build_video_name_index(sb_dirs)
    cam_idx = build_video_name_index(cam_dirs)

    sb_count = replace_from_template(
        template / SB_GROUP / "videos_quant_sb",
        case_input / SB_GROUP / "videos_quant_sb",
        sb_idx,
        args.copy_mode,
    )
    cam_count = replace_from_template(
        template / CAM_GROUP / "videos_quant_camera",
        case_input / CAM_GROUP / "videos_quant_camera",
        cam_idx,
        args.copy_mode,
    )

    links = count_symlinks(case_input)
    print(f"videos_quant_sb={sb_count}")
    print(f"videos_quant_camera={cam_count}")
    print("repeat_policy=exact_filename_only")
    print(f"symlink_count={links}")
    if args.forbid_symlink and links:
        raise SystemExit("检测到 symlink；默认禁止 symlink")
    print("REPAIR_EXACT_REPEATS_OK")


if __name__ == "__main__":
    main()
