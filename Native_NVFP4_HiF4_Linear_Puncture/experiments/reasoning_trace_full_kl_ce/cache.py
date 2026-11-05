"""Cache independent native final normalized hidden states, not giant vocabulary targets."""
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
from ..reasoning_trace_last_head.losses import log_bins
from ..reasoning_trace_last_head.model import non_layer
from .artifacts import atomic_save, load, read_json, selected_samples, sha256, state_digest, verify, write_json
from .config import NATIVE


@torch.no_grad()
def prepare(root):
    root, m = verify(root)
    train, val = selected_samples(m)
    samples = train + val
    directory = root / "cache/teacher_hidden"
    directory.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _, device, hidden = context(samples)
    started = time.monotonic()
    for layer in range(48):
        base = load_qwen3_moe_layer_state(NATIVE, layer, device)
        runner = NativeQwen3MoELayerRuntime(base).to(device).eval()
        output = {}
        for batch in build_validation_batches(samples, 4):
            x = assemble(hidden, batch, device)
            call = build_qwen3_moe_layer_call(str(NATIVE), x)
            y = runner(x, attention_mask=causal_mask(x), position_embeddings=call.position_embeddings).output
            for i, sample in enumerate(batch):
                value = y[i, :sample.input_ids.numel()].cpu().contiguous()
                if not torch.isfinite(value).all():
                    raise RuntimeError("non-finite native prefix")
                output[sample.sample_id] = value
        hidden = output
        print({"event": "teacher_prefix", "layer": layer, "samples": len(hidden),
               "seconds": round(time.monotonic() - started, 2)}, flush=True)
        del base, runner, x, y, call
        gc.collect()
        torch.cuda.empty_cache()
    values = non_layer(m["initialization"], device)
    old_dir = Path(m["cache_run"]) / "cache/teacher"
    old_records = {row["id"]: row for row in read_json(old_dir / "manifest.json")["samples"]}
    rows, errors = [], []
    for number, sample in enumerate(samples):
        h = _rms_norm(hidden[sample.sample_id].to(device), values["model.norm.weight"], 1e-6)
        if h.dtype != torch.bfloat16 or not torch.isfinite(h).all():
            raise RuntimeError("teacher normalized hidden cache must remain finite BF16")
        old = old_records[sample.sample_id]
        if sha256(old_dir / old["path"]) != old["sha256"]:
            raise ValueError("old sparse teacher target changed")
        targets = load(old_dir / old["path"])
        max_error = 0.
        for start in range(0, len(h), 128):
            logits = F.linear(h[start:start + 128], values["lm_head.weight"])
            top, tail = log_bins(logits, targets["ids"][start:start + 128].to(device))
            expected_top, expected_tail = targets["top"][start:start + 128].to(device), targets["tail"][start:start + 128].to(device)
            torch.testing.assert_close(top, expected_top, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(tail, expected_tail, atol=1e-6, rtol=1e-5)
            max_error = max(max_error, float((top - expected_top).abs().max()), float((tail - expected_tail).abs().max()))
        errors.append(max_error)
        path = directory / f"{number:03d}.pt"
        token_sha = state_digest({"ids": sample.input_ids, "mask": sample.loss_mask})
        atomic_save({"hidden": h.cpu().contiguous(), "sample_id": sample.sample_id, "token_sha256": token_sha}, path)
        rows.append({"id": sample.sample_id, "path": path.name, "tokens": len(h),
                     "sha256": sha256(path), "token_sha256": token_sha})
    write_json({"complete": True, "role": "teacher_hidden", "prefix_layers": 48,
                "independent_prefix": True, "final_norm_applied": True, "dtype": "bfloat16",
                "samples": rows, "calibration_sha256": m["calibration_sha256"],
                "initialization_export_sha256": m["initialization_export_sha256"],
                "teacher_index_sha256": m["teacher_index_sha256"],
                "old_sparse_target_replay_max_log_probability_error": max(errors),
                "seconds": time.monotonic() - started, "peak_allocated": torch.cuda.max_memory_allocated()}, directory / "manifest.json")


class FixedCache:
    def __init__(self, root, m, objective):
        self.root, self.objective = Path(root), objective
        self.records = {}
        for role in (("student", "teacher_hidden") if objective == "full_kl" else ("student",)):
            d = self.root / "cache" / role
            bound = read_json(d / "manifest.json")
            if (not bound["complete"] or not bound["independent_prefix"]
                    or bound["prefix_layers"] != (47 if role == "student" else 48)
                    or bound["calibration_sha256"] != m["calibration_sha256"]
                    or bound["initialization_export_sha256"] != m["initialization_export_sha256"]):
                raise ValueError("cache source contract changed")
            if len(bound["samples"]) != 160 or len({r["id"] for r in bound["samples"]}) != 160:
                raise ValueError("cache must cover all 160 complete trajectories")
            for row in bound["samples"]:
                if sha256(d / row["path"]) != row["sha256"]:
                    raise ValueError("cached full hidden input changed")
            self.records[role] = {row["id"]: row for row in bound["samples"]}

    def batch(self, samples, device):
        result = {}
        for role, records in self.records.items():
            values = {}
            for sample in samples:
                row = records[sample.sample_id]
                token_sha = state_digest({"ids": sample.input_ids, "mask": sample.loss_mask})
                value = load(self.root / "cache" / role / row["path"])
                if (row["tokens"] != sample.input_ids.numel() or row["token_sha256"] != token_sha
                        or value["sample_id"] != sample.sample_id or value["token_sha256"] != token_sha):
                    raise ValueError("cached token/sample identity changed")
                values[sample.sample_id] = value["hidden"]
            result[role] = assemble(values, samples, device)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    prepare(parser.parse_args().run)
