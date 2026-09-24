#!/usr/bin/env python3
"""Index, search, and check HiF4_Sp documentation without loading ML packages."""

import argparse
import os
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "docs/INDEX.md"
VENDOR = {"3rdparty", "NVFP4/llm-compressor"}
SKIP_DIRS = {".git", ".cache", "__pycache__", ".pytest_cache", "node_modules", ".venv", "venv"}
EXTENSIONS = {".md", ".mdc", ".rst", ".txt", ".adoc", ".org"}
NON_DOCS = {"requirements.txt", "merges.txt", "vocab.txt"}
GROUPS = {
    "guides": "指南与协作规则",
    "lessons": "可复用经验",
    "records": "实验与维护记录",
    "history": "方案与历史",
    "artifacts": "原位运行证据与生成报告",
    "internal": "工具交接记录",
    "navigation": "导航与旧路径入口",
}
SCOPES = {
    "default": {"guides", "lessons"},
    "guides": {"guides"},
    "lessons": {"lessons"},
    "records": {"records", "artifacts"},
    "history": {"history", "internal"},
    "all": set(GROUPS),
}


def documents():
    for directory, dirs, names in os.walk(ROOT, followlinks=False):
        base = Path(directory)
        dirs[:] = sorted(
            name for name in dirs
            if name not in SKIP_DIRS
            and not (base / name).is_symlink()
            and (base / name).relative_to(ROOT).as_posix() not in VENDOR
        )
        for name in sorted(names):
            path = base / name
            if path == INDEX or name in NON_DOCS:
                continue
            if path.suffix.lower() in EXTENSIONS:
                yield path


def category(path, body):
    rel = path.relative_to(ROOT)
    parts = rel.parts
    if body.startswith("<!-- doc-redirect -->"):
        return "navigation"
    if parts[0] == ".ai-bridge":
        return "internal"
    if parts[0] == "docs" and path.name == "README.md":
        return "navigation"
    if parts[:3] == ("docs", "experience", "lessons"):
        return "lessons"
    if parts[:3] == ("docs", "experience", "records"):
        return "records"
    if any(part in {"results", ".result", "reports"} for part in parts):
        return "artifacts"
    if ("history" in parts or "plans" in parts or parts[0] == "archive"
            or path.stem.startswith("PLAN") or "LEGACY" in path.stem):
        return "history"
    if path.name == "EXPERIMENT_REPORT.md":
        return "records"
    return "guides"


def title(path, body):
    for line in body.splitlines():
        if line.startswith("# "):
            return line[2:].strip().replace("|", "\\|").replace("[", "\\[").replace("]", "\\]")
    return path.name


def catalog():
    return [(path, body, category(path, body)) for path in sorted(documents())
            for body in [path.read_text(encoding="utf-8")]]


def render_index(entries):
    lines = [
        "# HiF4_Sp 自有文档索引", "",
        "由 `python tools/docs.py index` 自动生成，请勿手工维护。返回 [文档入口](README.md)。", "",
        "按用途列出当前工作区的自有文档（含忽略的结果文件）；不搬动实验目录，不包含第三方文档、缓存和模型数据。",
        "方案、历史状态及运行证据的出现不代表当前已完成或获准执行。", "",
        f"共 {len(entries)} 份正文或导航文件；本索引不计入自身。", "",
    ]
    for group, label in GROUPS.items():
        selected = [entry for entry in entries if entry[2] == group]
        lines += [f"## {label}（{len(selected)}）", "", "| 文档 | 路径 |", "| --- | --- |"]
        for path, body, _ in selected:
            link = os.path.relpath(path, INDEX.parent).replace(os.sep, "/")
            rel = path.relative_to(ROOT).as_posix()
            lines.append(f"| [{title(path, body)}]({link}) | `{rel}` |")
        lines.append("")
    lines += [
        "## 第三方入口", "",
        "- [vLLM](../3rdparty/vllm/README.md)",
        "- [lighteval](../3rdparty/lighteval/README.md)",
        "- [llm-compressor 目录](../NVFP4/llm-compressor/)", "",
    ]
    return "\n".join(lines)


def markdown_targets(body):
    # Code samples contain historical/example paths, not navigation links.
    fence = None
    for number, line in enumerate(body.splitlines(), 1):
        match = re.match(r"^\s*(`{3,}|~{3,})", line)
        if match:
            marker = match.group(1)
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence):
                fence = None
            continue
        if fence is not None:
            continue
        for match in re.finditer(r"\[[^\]\n]*\]\(\s*(<[^>]+>|[^\s)]+)(?:\s+[^)]*)?\)", line):
            yield number, match.group(1).strip("<>")
        match = re.match(r"^\s*\[[^\]]+\]:\s*(<[^>]+>|\S+)", line)
        if match:
            yield number, match.group(1).strip("<>")


def check(entries):
    errors = []
    expected = render_index(entries)
    if not INDEX.exists() or INDEX.read_text(encoding="utf-8") != expected:
        errors.append("docs/INDEX.md 未同步，请运行 python tools/docs.py index")
    checked = 0
    pages = [(path, body) for path, body, _ in entries if path.suffix in {".md", ".mdc"}]
    if INDEX.exists():
        pages.append((INDEX, INDEX.read_text(encoding="utf-8")))
    for path, body in pages:
        for line, target in markdown_targets(body):
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            checked += 1
            destination = path.parent / unquote(parsed.path)
            if not destination.exists():
                errors.append(f"{path.relative_to(ROOT)}:{line}: 本地链接不存在: {target}")
    for error in errors:
        print(error)
    if errors:
        return 1
    print(f"通过：{len(entries)} 份文档，{checked} 个本地链接；索引已同步。")
    print("范围：本地目标存在性；不检查标题锚点、外网或实验正确性。")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("index", help="刷新 docs/INDEX.md")
    commands.add_parser("check", help="检查索引与 Markdown 本地链接")
    search = commands.add_parser("search", help="按主题搜索正文（不搜索自动索引）")
    search.add_argument("query", help="不区分大小写的文字关键词")
    search.add_argument("--scope", choices=SCOPES, default="default")
    args = parser.parse_args()
    entries = catalog()
    if args.command == "index":
        INDEX.write_text(render_index(entries), encoding="utf-8")
        print(f"已更新 docs/INDEX.md：{len(entries)} 份文档。")
        return 0
    if args.command == "check":
        return check(entries)
    query = args.query.casefold()
    if not query.strip():
        parser.error("搜索关键词不能为空")
    matches = 0
    for path, body, group in entries:
        if group not in SCOPES[args.scope]:
            continue
        for number, line in enumerate(body.splitlines(), 1):
            if query in line.casefold():
                print(f"{path.relative_to(ROOT)}:{number}: {line.strip()}")
                matches += 1
    if not matches:
        print("无匹配文档。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
