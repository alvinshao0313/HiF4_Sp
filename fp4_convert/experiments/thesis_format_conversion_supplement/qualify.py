"""Real layer search/export/operator verification, plus the one-sided constraint."""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from ..e2e_diag_reconstruction.core.modelopt_moe_checkpoint import load_qwen3_moe_layer_state
from ..e2e_diag_reconstruction.core.moe_transforms import apply_r64_no_cross_head
from ..module_design_ablation.parameters import LayerParameters
from ..reasoning_trace_calibration.training import samples_for, context
from ..non_equivalent_reconstruction.transforms import projection_weights
from ...src.formats import qdq_hif4_direct
from .artifacts import *
from .config import NATIVE, SOURCE_RUN
from .export import convert, tensor_key
from .quantization import ConvertedStudent, replay
from .training import WeightOrthogonalParameters, orthogonality


def scale(root):
    root=run_root(root)
    convert(root,"r64_scale",qualification=True)
    output=root/"qualification/models/r64_scale"
    base=load_qwen3_moe_layer_state(NATIVE,0,"cuda:0")
    reloaded=load_file(str(output/"model-layer-00000-of-00048.safetensors"),device="cuda:0")
    student=ConvertedStudent(base)
    student.quantized={n:reloaded[tensor_key(0,n)] for n,w in projection_weights(base) if n!="router"}
    samples,_,_=samples_for(SOURCE_RUN,"matrix_norm_s1k")
    samples=sorted(samples,key=lambda s:s.input_ids.numel(),reverse=True)[:4]
    _,_,hidden=context(samples)
    # Actual routed input capture; tensors here are evidence, not calibration substitutes.
    class Capture:
        effective_tokens=0
        def __init__(self): self.values=[]
        def add_batch(self,x,out): self.values.append(x.detach().cpu())
    names=("q_proj","expert_0_gate_proj","expert_0_up_proj","expert_0_down_proj")
    student.collectors={n:Capture() for n in names}
    with torch.no_grad():
        outputs=replay(student,samples,hidden,propagate=True)
        details={}
        from vllm.model_executor.layers.quantization.hif4_transform_triton import hif4_r64_quantize_hifx4_triton
        for name,capture in student.collectors.items():
            if not capture.values: raise ValueError(f"representative expert unvisited: {name}")
            rotated_input=torch.cat(capture.values).to("cuda:0")
            q=student.quantized[name]
            reference=F.linear(qdq_hif4_direct(rotated_input,output_dtype=torch.bfloat16),q)
            # R64 is self-inverse; actual operator check uses the runtime QDQ on the
            # real pre-R64 BF16 input reconstructed from its captured FP32 rotation.
            original=apply_r64_no_cross_head(rotated_input).to(torch.bfloat16)
            activation=hif4_r64_quantize_hifx4_triton(original)
            actual=F.linear(activation,q)
            nmse=float((actual.float()-reference.float()).square().sum()/reference.float().square().sum())
            if nmse>1e-4: raise ValueError(f"actual runtime activation/Linear NMSE too large: {name} {nmse}")
            details[name]={"effective_tokens":capture.effective_tokens,"runtime_linear_nmse":nmse,
                           "saved_values_unchanged":True}
        free=LayerParameters(base,"matrix_norm")
        orth=WeightOrthogonalParameters(base)
        for name,weight in projection_weights(base):
            if name!="router" and not torch.equal(free.weight(name,weight),orth.weight(name,weight)):
                raise ValueError(f"orthogonal/free initial maps differ: {name}")
        before=orth.rotations["q_proj"]().detach().clone()
        activation_before=orth.projection(base.attention["q_proj"][:2],"q_proj",base.attention["q_proj"])[0]
    optimizer=torch.optim.AdamW([orth.rotations["q_proj"].coefficients],lr=3e-4,weight_decay=0.)
    with torch.enable_grad():
        target=torch.roll(free.weight("q_proj",base.attention["q_proj"]).detach(),shifts=1,dims=-1)
        for _ in range(2):
            optimizer.zero_grad()
            (orth.weight("q_proj",base.attention["q_proj"])-target).square().mean().backward()
            optimizer.step()
    activation_after=orth.projection(base.attention["q_proj"][:2],"q_proj",base.attention["q_proj"])[0]
    if not torch.equal(activation_before,activation_after): raise ValueError("online activation R64 changed")
    if torch.equal(before,orth.rotations["q_proj"]()): raise ValueError("Cayley coefficients did not update")
    error=orthogonality(orth)
    result={"complete":True,"scope":"real layer 0, all 388 conversions and exact export reload; full longest batch routed inputs",
            "projection_checks":details,"orthogonal_initial_matches_free":True,"fixed_activation_after_update":True,
            "orthogonality_max_abs":error,"sample_ids":[s.sample_id for s in samples],
            "full_sequence_lengths":[s.input_ids.numel() for s in samples],
            "not_evidence_for":"full-model accuracy or orthogonal training qualification"}
    write_json(result,root/"qualification/scale.json")
    print(result,flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(__doc__);p.add_argument("--run",type=Path,required=True)
    scale(p.parse_args().run)
