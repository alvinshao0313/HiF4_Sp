"""Check the inactive H32 extension against the original direct/R64 kernel."""
import argparse
import ast
import importlib.util
from pathlib import Path

import torch

from .common import BASE, HERE, OLD_RUNS, REPO, file_hash, read, save
from ..qwen3_mxfp4.kernels import activation_qdq


def definition(path, name):
    tree = ast.parse(path.read_text())
    return ast.dump(next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == name), include_attributes=False)


def validate(output):
    before = BASE / "fht32_tp4_20261110T032038Z/source_before"
    current = HERE.parent / "qwen3_mxfp4"
    original = read(OLD_RUNS["direct"] / "manifest.json")["source_sha256"]
    hashes = {}
    for name in ("kernels.py", "runtime.py", "format.py"):
        expected = original[str((current / name).relative_to(REPO))]
        if file_hash(before / name) != expected:
            raise ValueError("Original source snapshot does not match the completed run")
        hashes[str(before / name)] = expected
        hashes[str(current / name)] = file_hash(current / name)
    for filename, name in (("kernels.py", "packed_gemm"), ("runtime.py", "routed_experts"),
                           ("format.py", "quantize"), ("format.py", "dequantize")):
        if definition(before / filename, name) != definition(current / filename, name):
            raise ValueError(f"Original shared computation changed: {name}")
    spec = importlib.util.spec_from_file_location("original_direct_r64_kernels", before / "kernels.py")
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    torch.manual_seed(1234)
    rows = []
    for variant in ("direct", "r64"):
        for tokens in (1, 256, 8192):
            for width in (2048, 192):
                x = torch.randn(tokens, width, device="cuda", dtype=torch.bfloat16)
                x[0, :32] = 0
                expected = old.activation_qdq(x, variant, diagnostics=True)
                got = activation_qdq(x, variant, diagnostics=True)
                for a, b in zip(got, expected):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                rows.append({"variant": variant, "shape": [tokens, width], "codes_scales_rotation_qdq_exact": True})
    save(output, {"status": "passed", "device": torch.cuda.get_device_name(), "checks": rows,
                  "packed_gemm_and_routed_experts_ast_unchanged": True, "source_sha256": hashes})
    print(f"Preserved direct/R64: {len(rows)} real GPU shape checks; GEMM and MoE source unchanged", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    validate(Path(parser.parse_args().output))
