"""官方 VBench-I2V 10 维评测 runner。

该模块故意不绑定 Slurm。Slurm 只应作为外层 wrapper 调用本模块。
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

from .compat import apply_all_patches
from .constants import CAM_GROUP, DIMS_10, I2V_NATIVE_DIMS, MODE_TO_CAM_DIR, MODE_TO_SB_DIR, SB_GROUP
from .utils import ensure_dir, parse_dims


def find_full_info(group_dir: Path, dim: str, allow_fallback: bool = False) -> Path:
    """寻找对应维度的 full_info json。

    规则尽量显式：
    1. 优先使用文件名包含当前维度的 full_info；
    2. 若目录下只有一个 full_info，可认为它是该 group 的通用 full_info；
    3. 多个候选且无精确匹配时默认报错，避免误用其他维度的 full_info。
    """
    exact = sorted(group_dir.glob(f"*{dim}*full_info*.json"))
    if exact:
        return exact[0]
    generic = sorted(group_dir.glob("*full_info*.json"))
    if len(generic) == 1:
        return generic[0]
    if allow_fallback and generic:
        print(f"[WARN] fallback full_info for dim={dim}: {generic[0]}")
        return generic[0]
    if not generic:
        generic = sorted(group_dir.glob("*.json"))
        if len(generic) == 1 and allow_fallback:
            print(f"[WARN] fallback json for dim={dim}: {generic[0]}")
            return generic[0]
    detail = ", ".join(str(x.name) for x in generic[:20])
    raise FileNotFoundError(f"找不到唯一 full_info json: group={group_dir}, dim={dim}, candidates=[{detail}]")


def select_group(case_input: Path, mode: str, dim: str, allow_full_info_fallback: bool = False) -> tuple[Path, Path]:
    """返回 videos_path 和 full_info_json。"""
    if dim == "camera_motion":
        group = case_input / CAM_GROUP
        videos = group / MODE_TO_CAM_DIR[mode]
    else:
        group = case_input / SB_GROUP
        videos = group / MODE_TO_SB_DIR[mode]
    if not videos.is_dir():
        raise FileNotFoundError(f"missing videos dir: {videos}")
    return videos, find_full_info(group, dim, allow_fallback=allow_full_info_fallback)


def call_vbench(device: str, full_info_json: Path, output_dir: Path, videos_path: Path, image_folder: Path, dim: str, name: str, resolution: str, use_compat_patches: bool = True) -> None:
    if use_compat_patches:
        print("[INFO] applying VBench compatibility patches")
        apply_all_patches()
    from vbench2_beta_i2v import VBenchI2V  # 延迟 import，方便 preflight 与 smoke test 分离。

    output_dir.mkdir(parents=True, exist_ok=True)

    # 不同 VBenchI2V 版本的 __init__ 可能略有差异；这里使用最常见形式。
    bench = VBenchI2V(device, str(full_info_json.parent), str(output_dir))

    kwargs = {
        "videos_path": str(videos_path),
        "name": name,
        "dimension_list": [dim],
        "custom_image_folder": str(image_folder),
        "resolution": resolution,
    }

    try:
        bench.evaluate(**kwargs)
    except TypeError as e:
        # 只在确认为 resolution 参数不兼容时重试；其他 TypeError 可能来自 VBench 内部，必须暴露。
        msg = str(e)
        if "resolution" not in msg and "unexpected keyword" not in msg:
            raise
        print(f"[WARN] VBench evaluate() 不接受 resolution 参数，移除后重试: {e!r}")
        kwargs.pop("resolution", None)
        bench.evaluate(**kwargs)


def main() -> None:
    ap = argparse.ArgumentParser(description="运行 VBench-I2V 官方 10 维评测")
    ap.add_argument("--case-input", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--image-folder", required=True)
    ap.add_argument("--mode", choices=["bf16", "quant"], default="quant")
    ap.add_argument("--dims", default="all")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resolution", default="832*480")
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--allow-full-info-fallback", action="store_true", help="多个 full_info 候选无法精确匹配时允许退化使用第一个候选；默认关闭以避免评测错配")
    ap.add_argument("--no-compat-patches", action="store_true", help="不应用 torch.load 等 VBench 兼容补丁")
    args = ap.parse_args()

    case_input = Path(args.case_input)
    out_root = Path(args.output_dir)
    image_folder = Path(args.image_folder)
    dims = parse_dims(args.dims, DIMS_10)

    if not case_input.is_dir():
        raise SystemExit(f"missing case_input: {case_input}")
    if not image_folder.is_dir():
        raise SystemExit(f"missing image_folder: {image_folder}")

    errors = []
    for dim in dims:
        dim_out = out_root / dim
        name = f"{case_input.name}_{args.mode}_{dim}"
        if args.skip_existing and list(dim_out.glob("*_eval_results.json")):
            print(f"[SKIP] {dim} existing result")
            continue
        try:
            videos_path, full_info_json = select_group(case_input, args.mode, dim, args.allow_full_info_fallback)
            print(f"=== EVAL dim={dim} ===")
            print(f"videos_path={videos_path}")
            print(f"full_info_json={full_info_json}")
            print(f"output_dir={dim_out}")
            call_vbench(args.device, full_info_json, dim_out, videos_path, image_folder, dim, name, args.resolution, use_compat_patches=not args.no_compat_patches)
        except Exception as e:
            print(f"[EVAL_FAILED] dim={dim} error={e!r}")
            errors.append((dim, repr(e)))
            if not args.continue_on_error:
                raise

    if errors:
        print("ERROR_COUNT=", len(errors))
        for dim, err in errors:
            print(f"ERROR dim={dim} {err}")
        raise SystemExit(1)
    print("RUN_EVAL_10DIM_OK")


if __name__ == "__main__":
    main()
