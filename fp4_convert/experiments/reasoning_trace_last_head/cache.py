"""Independent full prefixes; only final student inputs and sparse teacher targets persist."""
import argparse
import gc
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from ..e2e_diag_reconstruction.core.modelopt_moe_checkpoint import load_qwen3_moe_layer_state
from ..e2e_diag_reconstruction.core.moe_semantic_hif4 import NativeQwen3MoELayerRuntime, _rms_norm
from ..e2e_diag_reconstruction.data.batching import build_validation_batches
from ..e2e_diag_reconstruction.training.moe_layer_runtime import build_qwen3_moe_layer_call
from ..non_equivalent_reconstruction.student import causal_mask
from ..non_equivalent_reconstruction.train import assemble
from ..reasoning_trace_calibration.training import context
from ..reasoning_trace_router_bf16.student import load_layer, RouterStudent
from .artifacts import atomic_save, digest, load, read_json, selected_samples, sha256, state_digest, verify, write_json
from .config import NATIVE
from .losses import target
from .model import non_layer


@torch.no_grad()
def prepare(root, role):
    root, m = verify(root)
    train, val = selected_samples(m)
    samples = train + val
    directory = root / "cache" / role
    directory.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda:0")
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _, _, hidden = context(samples)
    started = time.monotonic()
    layers = 48 if role == "teacher" else 47
    for layer in range(layers):
        if role == "teacher":
            base = load_qwen3_moe_layer_state(NATIVE, layer, device)
            runner = NativeQwen3MoELayerRuntime(base).to(device).eval()
        else:
            base = load_layer(m["initialization"], layer, device)
            runner = RouterStudent(base).to(device).eval().requires_grad_(False)
        outputs = {}
        for batch in build_validation_batches(samples, 4):
            x = assemble(hidden, batch, device)
            call = build_qwen3_moe_layer_call(str(NATIVE), x)
            if role == "teacher":
                y = runner(x, attention_mask=causal_mask(x), position_embeddings=call.position_embeddings).output
            else:
                y = runner(x, position_embeddings=call.position_embeddings, use_ste=False).output
            for i, sample in enumerate(batch):
                value = y[i, :sample.input_ids.numel()].cpu().contiguous()
                if not torch.isfinite(value).all():
                    raise RuntimeError("non-finite independent prefix")
                outputs[sample.sample_id] = value
        hidden = outputs
        print({"event": "prefix_layer", "role": role, "layer": layer,
               "samples": len(hidden), "seconds": round(time.monotonic() - started, 2)}, flush=True)
        del runner, base, x, y, call
        gc.collect()
        torch.cuda.empty_cache()
    if role == "teacher":
        values = non_layer(m["initialization"], device)
        # B's non-layer weights are copied from native; bind/check that fact explicitly.
        from safetensors import safe_open
        index = read_json(NATIVE / "model.safetensors.index.json")["weight_map"]
        for name in ("model.norm.weight", "lm_head.weight", "model.embed_tokens.weight"):
            with safe_open(str(NATIVE / index[name]), framework="pt", device="cpu") as handle:
                native_value = handle.get_tensor(name)
            if not torch.equal(native_value, values[name].cpu()):
                raise RuntimeError(f"B's initial output weight differs from teacher: {name}")
    records = []
    for number, sample in enumerate(samples):
        value = hidden[sample.sample_id]
        if role == "teacher":
            h = _rms_norm(value.to(device), values["model.norm.weight"], 1e-6)
            blocks = [target(F.linear(h[i:i + 128], values["lm_head.weight"]))
                      for i in range(0, h.shape[0], 128)]
            payload = {key: torch.cat([b[key] for b in blocks]).cpu() for key in ("ids", "top", "tail")}
            mass = payload["top"].exp().sum(-1) + payload["tail"].exp()
            if not torch.allclose(mass, torch.ones_like(mass), atol=1e-5, rtol=1e-5):
                raise RuntimeError("teacher 65-bin probability mass is not conserved")
        else:
            payload = {"hidden": value}
        payload.update(sample_id=sample.sample_id, token_sha256=state_digest({"ids": sample.input_ids, "mask": sample.loss_mask}))
        path = directory / f"{number:03d}.pt"
        atomic_save(payload, path)
        records.append({"id": sample.sample_id, "path": path.name, "tokens": sample.input_ids.numel(),
                        "sha256": sha256(path), "token_sha256": payload["token_sha256"]})
    write_json({"complete": True, "role": role, "independent_prefix": True,
                "prefix_layers": layers, "protocol_sha256": m["protocol_sha256"],
                "source_hashes_sha256": digest(read_json(root / "source_hashes.json")),
                "calibration_sha256": m["calibration_sha256"], "initialization_export_sha256": m["initialization_export_sha256"],
                "teacher_index_sha256": m["teacher_index_sha256"], "samples": records,
                "seconds": time.monotonic() - started,
                "peak_memory_allocated": torch.cuda.max_memory_allocated()}, directory / "manifest.json")


class FixedCache:
    def __init__(self, root, manifest):
        self.root = Path(root)
        self.records = {}
        for role in ("student", "teacher"):
            directory = self.root / "cache" / role
            bound = read_json(directory / "manifest.json")
            if (not bound["complete"] or not bound["independent_prefix"]
                    or bound["prefix_layers"] != (47 if role == "student" else 48)
                    or bound["source_hashes_sha256"] != digest(read_json(self.root / "source_hashes.json"))
                    or bound["calibration_sha256"] != manifest["calibration_sha256"]
                    or bound["protocol_sha256"] != manifest["protocol_sha256"]
                    or bound["initialization_export_sha256"] != manifest["initialization_export_sha256"]
                    or bound["teacher_index_sha256"] != manifest["teacher_index_sha256"]):
                raise ValueError("stale or incomplete independent cache")
            if len(bound["samples"]) != 160 or len({r["id"] for r in bound["samples"]}) != 160:
                raise ValueError("cache must cover all 160 complete trajectories")
            self.records[role] = {r["id"]: r for r in bound["samples"]}
            for r in bound["samples"]:
                if sha256(directory / r["path"]) != r["sha256"]:
                    raise ValueError("cached payload changed")

    def batch(self, samples, device):
        hidden, targets = {}, {k: [] for k in ("ids", "top", "tail")}
        for sample in samples:
            token_sha = state_digest({"ids": sample.input_ids, "mask": sample.loss_mask})
            for role in ("student", "teacher"):
                r = self.records[role][sample.sample_id]
                if r["token_sha256"] != token_sha or r["tokens"] != sample.input_ids.numel():
                    raise ValueError("cache token identity/coverage differs from B")
                payload = load(self.root / "cache" / role / r["path"])
                if payload["sample_id"] != sample.sample_id or payload["token_sha256"] != token_sha:
                    raise ValueError("cache sample binding changed")
                if role == "student":
                    hidden[sample.sample_id] = payload["hidden"]
                else:
                    valid = sample.loss_mask.bool()
                    for key in targets:
                        targets[key].append(payload[key][valid])
        return assemble(hidden, samples, device), {k: torch.cat(v).to(device) for k, v in targets.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--role", choices=("student", "teacher"), required=True)
    args = parser.parse_args()
    prepare(args.run, args.role)
