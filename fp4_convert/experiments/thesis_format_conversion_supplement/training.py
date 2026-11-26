"""One-sided Cayley constraint; fixed activation R64 and the clean A20 recipe."""
import argparse
import gc
from pathlib import Path
import time
from types import SimpleNamespace

import torch
from torch import nn

from ..module_design_ablation.parameters import LayerParameters
from ..module_design_ablation.rotation import CayleyRotation, apply_group_matrix
from ..module_design_ablation.student import Student
from ..non_equivalent_reconstruction.transforms import projection_weights
from ..non_equivalent_reconstruction.train import teacher_targets, run_epoch, propagate
from ..e2e_diag_reconstruction.core.modelopt_moe_checkpoint import load_qwen3_moe_layer_state
from ..e2e_diag_reconstruction.core.moe_semantic_hif4 import NativeQwen3MoELayerRuntime
from ..e2e_diag_reconstruction.core.moe_transforms import apply_r64_no_cross_head
from ..e2e_diag_reconstruction.data.batching import build_length_bucket_batches
from ..reasoning_trace_calibration.training import samples_for, context, config
from ...src.formats import qdq_hif4_direct
from .artifacts import *
from .config import NATIVE, SOURCE_RUN
from .export import save_layer, finish, _copy_json_without_quantization, _copy_tokenizer_assets, _copy_non_layer_tensors


class WeightOrthogonalParameters(LayerParameters):
    def __init__(self, base):
        # Construct only frozen-R64/norm state, never allocate the free matrices.
        super().__init__(base, "norm")
        for name, weight in projection_weights(base):
            if name != "router":
                self.rotations[name] = CayleyRotation(weight.shape[-1]//64, device=weight.device)

    def weight(self, name, weight, *, head_dim=None):
        if name == "router":
            return weight.float()
        return apply_group_matrix(weight.float(), self.rotations[name](), head_dim=head_dim)

    def projection(self, x, name, weight, *, head_dim=None):
        return (apply_r64_no_cross_head(x.float(), head_dim=head_dim),
                self.weight(name, weight, head_dim=head_dim))


class OrthogonalStudent(Student):
    def __init__(self, base):
        nn.Module.__init__(self)
        self.base = base
        self.learned = WeightOrthogonalParameters(base)
        self.eps = 1e-6
        self.attention_view = SimpleNamespace(num_key_value_groups=base.spec.num_attention_heads//base.spec.num_key_value_heads,
                                              training=False)


@torch.no_grad()
def orthogonality(learned):
    error = max(float((q() @ q().transpose(-1,-2) - q.identity).abs().max()) for q in learned.rotations.values())
    if error > 1e-5:
        raise ValueError(f"FP32 orthogonality exceeds existing 1e-5 tolerance: {error}")
    return error


def train(root, *, qualification=False):
    root = run_root(root); verify_sources(root)
    training, validation, hashes = samples_for(SOURCE_RUN, "matrix_norm_s1k")
    if qualification:
        # Full longest sequences, formal batch, 2 updates/validation cycles and 2 layers.
        training = sorted(training, key=lambda s:s.input_ids.numel(), reverse=True)[:4]
        validation = sorted(validation, key=lambda s:s.input_ids.numel(), reverse=True)[:4]
    output = root / ("qualification/training" if qualification else "training") / "orthogonal_a20"
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    samples = training+validation
    collator, device, hidden = context(samples)
    cfg = config()
    manifest = {"complete": False, "qualification": qualification, "calibration_sha256": hashes,
                "completed_layers": [], "sample_ids": {"train": [s.sample_id for s in training],
                "val": [s.sample_id for s in validation]}, "initialization": str(NATIVE)}
    write_json(manifest, output/"manifest.json")
    for layer in range(2 if qualification else 48):
        torch.cuda.reset_peak_memory_stats()
        layer_start = time.monotonic()
        base = load_qwen3_moe_layer_state(NATIVE, layer, device)
        teacher = NativeQwen3MoELayerRuntime(base).to(device).eval()
        targets, logits = teacher_targets(teacher, NATIVE, samples, collator, hidden, device, 4)
        del teacher
        student = OrthogonalStudent(base).to(device)
        frozen = state_digest({n:w for n,w in projection_weights(base)})
        initial = state_digest(student.learned.snapshot())
        epochs = 2 if qualification else 20
        optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4, weight_decay=0.)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                    T_max=epochs*len(build_length_bucket_batches(training, 4, 42)), eta_min=0.)
        best = None
        for epoch in range(epochs):
            t = run_epoch(student,cfg,NATIVE,training,collator,hidden,targets,logits,device,
                          epoch=epoch,optimizer=optimizer,scheduler=scheduler)
            v = run_epoch(student,cfg,NATIVE,validation,collator,hidden,targets,logits,device,epoch=epoch)
            if best is None or v["reconstruction_nmse"] < best["metrics"]["reconstruction_nmse"]:
                best = {"epoch":epoch+1,"metrics":v,"parameters":student.learned.snapshot()}
            write_json({"train":t,"validation":v},output/f"layers/{layer:03d}/epochs/{epoch+1:03d}.json")
        selected = output/f"layers/{layer:03d}/selected.pt"
        atomic_save(best, selected)
        student.learned.load_state_dict(load(selected)["parameters"],strict=True)
        if state_digest(student.learned.snapshot()) != state_digest(best["parameters"]) or initial == state_digest(best["parameters"]):
            raise ValueError("selected checkpoint did not update or exact reload failed")
        if state_digest({n:w for n,w in projection_weights(base)}) != frozen:
            raise ValueError("frozen effective weights/router changed")
        constraint = orthogonality(student.learned)
        hidden = propagate(student,NATIVE,samples,hidden,device,4)
        if any(not torch.isfinite(x).all() for x in hidden.values()):
            raise ValueError("nonfinite propagated student state")
        write_json({"seconds":time.monotonic()-layer_start,"orthogonality_max_abs":constraint,
                    "peak_allocated_bytes":torch.cuda.max_memory_allocated(),
                    "selected_epoch":best["epoch"],"validation":best["metrics"]},output/f"layers/{layer:03d}/metrics.json")
        manifest["completed_layers"].append(layer)
        manifest["seconds"] = time.monotonic()-started
        write_json(manifest,output/"manifest.json")
        del student,base,targets,logits,optimizer,scheduler,best
        gc.collect();torch.cuda.empty_cache()
    manifest["complete"]=True
    write_json(manifest,output/"manifest.json");verify_sources(root)


@torch.no_grad()
def export(root):
    root=run_root(root);verify_sources(root)
    directory=root/"training/orthogonal_a20"
    saved=read_json(directory/"manifest.json")
    if not saved["complete"] or saved["qualification"] or saved["completed_layers"] != list(range(48)):
        raise ValueError("formal training incomplete")
    output=root/"models/orthogonal_a20";output.mkdir(parents=True,exist_ok=False)
    _copy_json_without_quantization(NATIVE/"config.json",output/"config.json")
    _copy_tokenizer_assets(NATIVE,output)
    weight_map={};_copy_non_layer_tensors(NATIVE,output,weight_map)
    for layer in range(48):
        base=load_qwen3_moe_layer_state(NATIVE,layer,"cuda:0")
        learned=WeightOrthogonalParameters(base)
        learned.load_state_dict(load(directory/f"layers/{layer:03d}/selected.pt")["parameters"],strict=True)
        quantized={n:qdq_hif4_direct(learned.weight(n,w,head_dim=128 if n=="o_proj" else None),output_dtype=torch.bfloat16)
                   for n,w in projection_weights(base) if n!="router"}
        base.input_layernorm_weight=learned.input_norm.to(torch.bfloat16)
        base.post_attention_layernorm_weight=learned.moe_norm.to(torch.bfloat16)
        weight_map.update(save_layer(base,quantized,output))
        del base,learned,quantized;gc.collect();torch.cuda.empty_cache()
    finish(output,weight_map,"orthogonal_a20");verify_sources(root)


if __name__ == "__main__":
    p=argparse.ArgumentParser(__doc__);p.add_argument("action",choices=("train","export"))
    p.add_argument("--run",type=Path,required=True);p.add_argument("--qualification",action="store_true")
    a=p.parse_args()
    if a.action=="train":train(a.run,qualification=a.qualification)
    else:export(a.run)
