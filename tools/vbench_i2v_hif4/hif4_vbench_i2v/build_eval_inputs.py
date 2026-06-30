"""把 HiF4/Wan 生成视频整理为 VBench-I2V case input。

重要约束：本脚本只整理/复制已经生成好的视频，不生成视频，也不把单个
``base-0.mp4`` 复制成 ``base-1.mp4`` ... ``base-4.mp4``。

VBench-I2V 官方采样协议要求每个 image-prompt pair 采样 5 个视频，文件名为
``$prompt-0.mp4`` 到 ``$prompt-4.mp4``。因此本脚本默认使用 exact filename
匹配：模板目录中需要的每个 mp4 文件，都必须能在 generated/subject/background/
camera 源目录中找到同名文件。缺失任意 repeat 都会失败。
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .constants import CAM_GROUP, SB_GROUP
from .utils import build_video_name_index, copy_file, copy_tree_physical, count_symlinks, list_mp4


def replace_from_template(template_dir: Path, out_dir: Path, video_index: dict[str, Path], copy_mode: str) -> int:
    """按模板文件名 exact match 复制生成结果。

    旧版实现按 prompt base 匹配，会把一个 ``base-0.mp4`` 复制成 5 个 repeat。
    这会违反 VBench-I2V 的官方采样语义。新版只按完整文件名匹配：
    ``base-3.mp4`` 模板只能由源目录里的 ``base-3.mp4`` 填充。
    """
    files = list_mp4(template_dir)
    if not files:
        raise RuntimeError(f"模板目录没有 mp4: {template_dir}")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    missing = []
    copied = 0
    for tf in files:
        src = video_index.get(tf.name)
        if src is None:
            missing.append(tf.name)
            continue
        copy_file(src, out_dir / tf.name, copy_mode)
        copied += 1
    if missing:
        for name in missing[:50]:
            print(f"MISSING_EXACT_SOURCE template_name={name}")
        raise RuntimeError(
            f"缺少 {len(missing)} 个模板文件对应的 exact repeat 源视频；"
            "请先在生成阶段真实生成 prompt-0..prompt-4，而不是复制单个视频。"
        )
    return copied


def main() -> None:
    ap = argparse.ArgumentParser(description="构建 HiF4 的 VBench-I2V case input（严格 exact-repeat 模式）")
    ap.add_argument("--template-case", required=True)
    ap.add_argument("--out-case", required=True)
    ap.add_argument("--generated-dir", default=None, help="包含已真实生成 exact repeats 的 mp4 目录；若提供 subject/background/camera 专用目录，可不填")
    ap.add_argument("--subject-dir", default=None)
    ap.add_argument("--background-dir", default=None)
    ap.add_argument("--camera-dir", default=None)
    ap.add_argument("--copy-mode", choices=["physical", "hardlink", "symlink", "reflink"], default="physical")
    ap.add_argument("--forbid-symlink", action="store_true", default=True, help="默认禁止输出中残留 symlink")
    ap.add_argument("--allow-symlink", action="store_false", dest="forbid_symlink", help="允许 symlink，仅在确认 VBench scratch/Slurm 节点可访问同一路径时使用")
    args = ap.parse_args()

    template = Path(args.template_case)
    out_case = Path(args.out_case)
    if not template.is_dir():
        raise SystemExit(f"missing template case: {template}")

    print(f"COPY_TEMPLATE {template} -> {out_case}")
    copy_tree_physical(template, out_case)

    src_dirs_sb = []
    src_dirs_cam = []
    if args.generated_dir:
        src_dirs_sb.append(Path(args.generated_dir))
        src_dirs_cam.append(Path(args.generated_dir))
    if args.subject_dir:
        src_dirs_sb.append(Path(args.subject_dir))
    if args.background_dir:
        src_dirs_sb.append(Path(args.background_dir))
    if args.camera_dir:
        src_dirs_cam.append(Path(args.camera_dir))

    if not src_dirs_sb or not src_dirs_cam:
        raise SystemExit("需要 --generated-dir 或 subject/background/camera 专用目录")

    sb_idx = build_video_name_index(src_dirs_sb)
    cam_idx = build_video_name_index(src_dirs_cam)

    sb_template = template / SB_GROUP / "videos_quant_sb"
    cam_template = template / CAM_GROUP / "videos_quant_camera"
    sb_out = out_case / SB_GROUP / "videos_quant_sb"
    cam_out = out_case / CAM_GROUP / "videos_quant_camera"

    sb_count = replace_from_template(sb_template, sb_out, sb_idx, args.copy_mode)
    cam_count = replace_from_template(cam_template, cam_out, cam_idx, args.copy_mode)

    links = count_symlinks(out_case)
    print(f"videos_quant_sb={sb_count}")
    print(f"videos_quant_camera={cam_count}")
    print("repeat_policy=exact_filename_only")
    print(f"symlink_count={links}")
    if args.forbid_symlink and links:
        raise SystemExit("检测到 symlink；默认禁止 symlink")
    print("BUILD_EVAL_INPUTS_OK")


if __name__ == "__main__":
    main()
