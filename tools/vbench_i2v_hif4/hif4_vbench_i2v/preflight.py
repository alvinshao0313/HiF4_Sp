"""VBench-I2V 环境预检器。

它的职责是在真正评测前发现问题，而不是让 VBench 跑到一半才失败。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
from pathlib import Path

from .constants import DIMS_10
from .utils import read_json

MODULES = [
    "torch", "torchvision", "vbench", "vbench2_beta_i2v", "dreamsim",
    "timm", "open_clip", "decord", "cv2", "PIL", "pkg_resources", "packaging",
]


def _extract_image_refs(full_info) -> list[str]:  # type: ignore[no-untyped-def]
    refs: list[str] = []
    if isinstance(full_info, dict):
        items = list(full_info.values())
    else:
        items = list(full_info)
    keys = ["image", "image_path", "image_name", "img", "input_image", "path"]
    for it in items:
        if not isinstance(it, dict):
            continue
        for k in keys:
            v = it.get(k)
            if isinstance(v, str) and any(v.lower().endswith(e) for e in [".jpg", ".jpeg", ".png", ".webp"]):
                refs.append(v)
                break
    return refs


def main() -> None:
    ap = argparse.ArgumentParser(description="VBench-I2V 兼容性预检")
    ap.add_argument("--vbench-root", default=None)
    ap.add_argument("--full-info-json", default=None)
    ap.add_argument("--image-folder", default=None)
    ap.add_argument("--model-root", default=None)
    ap.add_argument("--scratch-dir", default=".scratch_vbench_i2v")
    ap.add_argument("--skip-import", action="store_true", help="本地 smoke test 可跳过真实 VBench import")
    ap.add_argument("--json-report", default=None)
    args = ap.parse_args()

    report = {"ok": True, "modules": {}, "paths": {}, "dims": DIMS_10, "errors": [], "warnings": []}

    if not args.skip_import:
        for m in MODULES:
            spec = importlib.util.find_spec(m)
            report["modules"][m] = {"ok": spec is not None, "origin": getattr(spec, "origin", None) if spec else None}
            if spec is None:
                report["ok"] = False
                report["errors"].append(f"MISSING_MODULE: {m}")

        try:
            from vbench2_beta_i2v import VBenchI2V  # noqa: F401
            report["modules"]["VBenchI2V"] = {"ok": True}
        except Exception as e:
            report["ok"] = False
            report["errors"].append(f"IMPORT_VBenchI2V_FAILED: {e!r}")

    for name, value in [
        ("vbench_root", args.vbench_root),
        ("full_info_json", args.full_info_json),
        ("image_folder", args.image_folder),
        ("model_root", args.model_root),
    ]:
        if value is None:
            report["paths"][name] = {"ok": False, "path": None, "note": "not_provided"}
            continue
        p = Path(value)
        ok = p.exists()
        report["paths"][name] = {"ok": ok, "path": str(p)}
        if not ok and name in {"full_info_json", "image_folder"}:
            report["ok"] = False
            report["errors"].append(f"MISSING_PATH: {name}={p}")

    # scratch 可写性检查。
    scratch = Path(args.scratch_dir)
    try:
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=scratch, delete=True) as f:
            f.write(b"ok")
        report["paths"]["scratch_dir"] = {"ok": True, "path": str(scratch)}
    except Exception as e:
        report["ok"] = False
        report["errors"].append(f"SCRATCH_NOT_WRITABLE: {scratch}: {e!r}")

    # 图片引用检查。
    if args.full_info_json and args.image_folder and Path(args.full_info_json).exists() and Path(args.image_folder).exists():
        info = read_json(args.full_info_json)
        refs = _extract_image_refs(info)
        missing = []
        folder = Path(args.image_folder)
        for r in refs:
            rp = Path(r)
            if rp.is_absolute():
                ok = rp.exists()
            else:
                ok = (folder / r).exists() or (folder / rp.name).exists()
            if not ok:
                missing.append(r)
        report["image_refs"] = {"checked": len(refs), "missing": len(missing), "missing_examples": missing[:20]}
        if missing:
            report["ok"] = False
            report["errors"].append(f"MISSING_IMAGE_REFS: {len(missing)}")

    if args.model_root:
        mr = Path(args.model_root)
        cotracker = mr / "checkpoints" / "cotracker2.pth"
        cotracker_repo = mr / "facebookresearch_co-tracker_main"
        report["models"] = {
            "cotracker_ckpt": {"ok": cotracker.exists(), "path": str(cotracker)},
            "cotracker_repo": {"ok": cotracker_repo.exists(), "path": str(cotracker_repo)},
        }
        for k, v in report["models"].items():
            if not v["ok"]:
                report["warnings"].append(f"MISSING_MODEL_WARNING: {k}={v['path']}")

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.json_report:
        Path(args.json_report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_report).write_text(text, encoding="utf-8")

    if report["ok"]:
        print("VBENCH_PREFLIGHT_OK")
    else:
        print("VBENCH_PREFLIGHT_FAILED")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
