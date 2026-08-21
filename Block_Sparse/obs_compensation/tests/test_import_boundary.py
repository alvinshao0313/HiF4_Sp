import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_BLOCK_PRUNING = {
    "block_pruning.config",
    "block_pruning.model_loader",
    "block_pruning.mlp_registry",
    "block_pruning.block_utils",
}
FORBIDDEN_TOP_LEVEL = {
    "HiFloat4",
    "QAD",
    "ChuanCi",
    "NVFP4",
    "MXFP4",
    "ScaleTuning",
    "tasks",
}
ALLOWED_PATH_BOOTSTRAPS = {
    "tests/conftest.py": (
        "BLOCK_SPARSE_ROOT = Path(__file__).resolve().parents[2]",
        "sys.path.insert(0, str(BLOCK_SPARSE_ROOT))",
    ),
    "run_obs_pruning.py": (
        "BLOCK_SPARSE_ROOT = Path(__file__).resolve().parents[1]",
        "sys.path.insert(0, str(BLOCK_SPARSE_ROOT))",
    ),
}


def _import_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.ImportFrom):
        return [] if node.module is None else [node.module]
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    return []


def _is_sys_path_mutation(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return False
    owner = node.func.value
    return (
        isinstance(owner, ast.Attribute)
        and isinstance(owner.value, ast.Name)
        and owner.value.id == "sys"
        and owner.attr == "path"
        and node.func.attr in {"append", "insert", "extend"}
    )


def test_obs_package_has_no_outside_project_imports():
    violations: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        relative = path.relative_to(PACKAGE_ROOT).as_posix()
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        path_mutations = 0
        for node in ast.walk(tree):
            for name in _import_names(node):
                top = name.split(".", 1)[0]
                if top in FORBIDDEN_TOP_LEVEL:
                    violations.append(f"{relative}: forbidden import {name}")
                if name.startswith("block_pruning.") and name not in ALLOWED_BLOCK_PRUNING:
                    violations.append(
                        f"{relative}: non-whitelisted Block_Sparse import {name}"
                    )
            if _is_sys_path_mutation(node):
                path_mutations += 1
        if relative in ALLOWED_PATH_BOOTSTRAPS:
            required_lines = ALLOWED_PATH_BOOTSTRAPS[relative]
            if path_mutations != 1:
                violations.append(
                    f"{relative}: expected exactly one sys.path mutation, got {path_mutations}"
                )
            for required in required_lines:
                if required not in source:
                    violations.append(f"{relative}: missing exact bootstrap line {required}")
        elif path_mutations:
            violations.append(f"{relative}: forbidden sys.path mutation")
    assert not violations, "\n".join(violations)
