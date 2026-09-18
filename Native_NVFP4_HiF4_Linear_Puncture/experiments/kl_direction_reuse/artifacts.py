"""Hash-bound artifacts; incomplete files are never treated as completed work."""
from dataclasses import fields, is_dataclass, replace
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import time

import torch


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tensor_hash(x):
    x = x.detach().cpu().contiguous()
    h = hashlib.sha256(str((str(x.dtype), tuple(x.shape))).encode())
    h.update(x.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def parameters_hash(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(tensor_hash(value).encode())
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    torch.save(value, tmp)
    os.replace(tmp, path)
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def checked_load(entry):
    if sha256(entry["path"]) != entry["sha256"]:
        raise RuntimeError(f"artifact changed: {entry['path']}")
    return load(entry["path"])


def move(value, device):
    if torch.is_tensor(value):
        return value.detach().to(device).clone()
    if is_dataclass(value):
        return replace(value, **{f.name: move(getattr(value, f.name), device) for f in fields(value)})
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [move(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(move(v, device) for v in value)
    return value


def _source_entries(experiment_sources=None):
    """Include imported scientific/runtime code, not just the new entrypoint."""
    base = Path(__file__).resolve().parents[1]
    paths = list(Path(__file__).parent.glob("*.py"))
    for directory in (base / "non_equivalent_reconstruction", base / "e2e_diag_reconstruction/core",
                      base / "e2e_diag_reconstruction/training", base / "e2e_diag_reconstruction/evaluation",
                      base / "e2e_diag_reconstruction/data", base / "diag_gradient"):
        paths.extend(directory.glob("*.py"))
    repo = base.parent
    paths.extend((repo / "src").glob("*.py"))
    # vLLM is modified locally in this workspace and participates in production math.
    vllm = base.parents[1] / "3rdparty/vllm/vllm"
    for relative in ("model_executor/layers/quantization", "model_executor/layers/fused_moe",
                     "model_executor/models", "v1/attention/backends", "model_executor/layers/rotary_embedding"):
        paths.extend((vllm / relative).rglob("*.py"))
    paths.extend((vllm / "model_executor/layers").glob("*.py"))
    if experiment_sources is not None:
        # Reconstruct the original keys while hashing the preserved source files.
        paths = [p for p in paths if p.parent != Path(__file__).parent]
    entries = {str(p): sha256(p) for p in sorted(set(paths))}
    if experiment_sources is not None:
        entries.update({str(Path(__file__).parent / p.name): sha256(p)
                        for p in Path(experiment_sources).glob('*.py')})
    entries["packages"] = {name: version(name) for name in ("torch", "transformers", "vllm", "triton")}
    entries["torch_cuda_build"] = torch.version.cuda
    return entries


def source_fingerprint(experiment_sources=None):
    return hashlib.sha256(json.dumps(_source_entries(experiment_sources), sort_keys=True).encode()).hexdigest()


def capture_fingerprint(experiment_sources=None):
    """Bind capture producers independently of the differentiable training code.

    All external scientific code and package versions remain bound. Inside this
    experiment only the actual capture producer and its called helpers matter;
    changing a loss used solely in training cannot change saved teacher tensors.
    """
    import ast
    here = Path(__file__).parent
    entries = {k: v for k, v in _source_entries(experiment_sources).items()
               if str(here) + '/' not in k}
    sources = Path(experiment_sources) if experiment_sources is not None else here
    for filename in ('config.py', 'data.py', 'materialize.py'):
        entries[filename] = sha256(sources / filename)
    nodes = {
        'capture.py': {'Logits', 'Install', 'Begin', 'Finish', 'capture'},
        'losses.py': {'kl_sum', 'distribution_metrics'},
        'artifacts.py': {'sha256', 'tensor_hash', 'read_json', 'write_json', 'save', 'load', 'checked_load'},
    }
    for filename, names in nodes.items():
        tree = ast.parse((sources / filename).read_text())
        found = {node.name: ast.dump(node, include_attributes=False) for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names}
        if set(found) != names:
            raise RuntimeError(f'incomplete capture producer dependency list: {filename}')
        entries[filename] = found
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()


class ActiveClock:
    def __init__(self, elapsed=0., now=time.monotonic):
        self.now = now
        self.started = now()
        self.prior = elapsed

    @property
    def elapsed(self):
        return self.prior + self.now() - self.started


class BudgetExpired(RuntimeError):
    pass


def check_budget(clock, seconds):
    if clock.elapsed >= seconds:
        raise BudgetExpired("training budget exhausted")
