"""Layer-sequential GPTQ, AWQ and SmoothQuant on complete calibration sequences."""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

from .checkpoint import Checkpoint
from .fit import calibrate_block, fit
from .layers import Projection, TextBlocks


def corpus(source, smoke_text):
    tokenizer = AutoTokenizer.from_pretrained(source.root, local_files_only=True)
    if smoke_text is not None:
        if not smoke_text.strip():
            raise ValueError("Provide a nonempty qualification prompt")
        ids = tokenizer(smoke_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        return [ids], ["smoke"], {"qualification_text": smoke_text}
    from datasets import Dataset
    from ..e2e_diag_reconstruction.data.calibration import build_s1k_original_sample
    from ..reasoning_trace_calibration.config import S1K_ARROW
    from ..reasoning_trace_calibration.training import samples_for
    from ..matrix_norm_efficiency.config import SOURCE_RUN

    train, _, hashes = samples_for(SOURCE_RUN, "matrix_norm_s1k")
    if source.config["model_type"] != "qwen3_moe":
        data = Dataset.from_file(str(S1K_ARROW))
        train = [build_s1k_original_sample(tokenizer, data[s.source_index], s.source_index) for s in train]
    if any(not s.loss_mask.bool().all() for s in train):
        raise ValueError("Selected S1K train rows must contain only valid, unpadded tokens")
    return [s.input_ids for s in train], [s.sample_id for s in train], hashes


@torch.inference_mode()
def calibrate(source, output, target, method, device, smoke_text=None, smoke_layers=None,
              smoke_sequence_index=None):
    if output.exists():
        raise FileExistsError(output)
    if smoke_text is not None and smoke_sequence_index is not None:
        raise ValueError("Choose either --smoke-text or --smoke-sequence-index")
    qualification = smoke_text is not None or smoke_sequence_index is not None
    if smoke_layers is not None and (not qualification or smoke_layers < 1):
        raise ValueError("--smoke-layers requires a qualification input and a positive layer count")
    output.parent.mkdir(parents=True, exist_ok=True)
    with Checkpoint(source) as checkpoint, TemporaryDirectory(dir=output.parent) as temp:
        checkpoint.verify_headers()
        folder = Path(temp) / "calibration"
        folder.mkdir()
        blocks = TextBlocks(checkpoint, device)
        sequences, ids, corpus_meta = corpus(checkpoint, smoke_text)
        if smoke_sequence_index is not None:
            if not 0 <= smoke_sequence_index < len(sequences):
                raise ValueError(f"Invalid calibration sequence index: {smoke_sequence_index}")
            sequences, ids = [sequences[smoke_sequence_index]], [ids[smoke_sequence_index]]
            corpus_meta = {**corpus_meta, "qualification_sequence_index": smoke_sequence_index}
        hidden = blocks.embed(sequences)
        index = {}
        coverage = []
        layers = blocks.cfg.num_hidden_layers
        if smoke_layers is not None and smoke_layers > layers:
            raise ValueError(f"Model has only {layers} layers")
        def save_group(filename, nodes, records):
            tensors = {}
            for node in nodes:
                if node.name in index or node.target != target:
                    raise ValueError(f"Missing or duplicate calibration for {node.name}")
                tensors[node.name + ".weight"] = node.weight.detach().cpu().contiguous()
                if node.channel_scale is not None:
                    tensors[node.name + ".input_divisor"] = node.channel_scale.detach().cpu().clone().contiguous()
                index[node.name] = filename
            if tensors:
                save_file(tensors, folder / filename)
            coverage.extend(records)
        for i in range(smoke_layers or layers):
            layer = blocks.load(i)
            def kw(x):
                return blocks.kwargs(i, x)
            records = calibrate_block(layer, hidden, kw, target, method)
            for j, states in enumerate(hidden):
                x = states.to(device)
                hidden[j] = layer(x, **kw(x)).cpu()
                if not torch.isfinite(hidden[j]).all():
                    raise ValueError(f"Nonfinite states after layer {i} in sequence {ids[j]}")
            nodes = (m for m in layer.modules() if isinstance(m, Projection) and not m.fp8)
            save_group(f"layer-{i:03}.safetensors", nodes, records)
            print(f"Layer {i+1}/{smoke_layers or layers}, {len(index)} projections fitted", flush=True)
            del layer
        if (smoke_layers or layers) == layers and "lm_head" in checkpoint.native:
            if blocks.kind != "qwen3_5":
                raise ValueError("Unknown final normalization for quantized output head")
            norm = Qwen3_5RMSNorm(blocks.cfg.hidden_size, eps=blocks.cfg.rms_norm_eps).to(device)
            norm.weight.data = checkpoint.tensor(blocks.root + ".norm.weight").to(device)
            xs = [norm(x.to(device)).squeeze(0).cpu() for x in hidden]
            head = Projection(checkpoint, "lm_head", device)
            record = fit([head], xs, target, method)
            save_group("head.safetensors", [head], [record])
        complete = checkpoint.native == set(index)
        if smoke_layers is None and not complete:
            raise ValueError("Incomplete full-model calibration")
        settings = {
            "gptq": {"block_size": 128, "damping_fraction": .01, "activation_order": False},
            "awq": {"channel_exponents": [i / 20 for i in range(20)], "search_activation_qdq": False},
            "smoothquant": {"alpha": .5},
        }[method]
        recipe = {"source": checkpoint.source_id(), "method": method, "target": target,
                  "group_size": 64 if target == "hif4" else 32, "settings": settings,
                  "scope_complete": complete, "purpose": "qualification" if qualification else "paper",
                  "sample_ids": ids, "tokens": [len(row) for row in sequences],
                  "token_sha256": [hashlib.sha256(row.numpy().tobytes()).hexdigest() for row in sequences],
                  "corpus": corpus_meta, "weights": index, "groups": coverage}
        (folder / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
        if output.exists():
            raise FileExistsError(output)
        folder.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--method", choices=["gptq", "awq", "smoothquant"], required=True)
    parser.add_argument("--target", choices=["hif4", "mxfp4"], required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--smoke-text", help="Use a single prompt for engineering qualification only")
    parser.add_argument("--smoke-sequence-index", type=int,
                        help="Use one complete formal S1K sequence for qualification; no token truncation")
    parser.add_argument("--smoke-layers", type=int, help="Stop qualification after the given prefix")
    calibrate(**vars(parser.parse_args()))
