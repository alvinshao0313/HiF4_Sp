"""Same-input, real-tensor diagnostics. These are not formal inference results."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F

from . import selected_layer_objective_trainer as tr
from .path_audit import selected_states, difference
from .corrected_phase import phase_root
from .capture_states import load_rank_records, load_raw_logits
from .residual_ledger import index_capture_records, extract_layer_tensors
from .run_state import atomic_write_json, write_jsonl
from .config import DEFAULT_MODEL_PATH
from ..e2e_diag_reconstruction.core.moe_semantic_hif4 import (
    native_nvfp4_linear,forward_student_attention_proj,StudentStepCache,
)


def local_worker(root: Path, variant: str, model_path: str):
    from vllm import _custom_ops as ops
    torch.set_num_threads(4)
    device=torch.device('cuda:0');snapshot=Path(tr.resolve_local_snapshot(model_path))
    out=phase_root(root)/'alignment'/f'local_{variant}';out.mkdir(parents=True,exist_ok=True)
    refs=root/'60_objective/actual_holdout'/variant
    states=selected_states(root)
    captures={}
    for meta in states:
        key=meta['sample_key'];di=meta['decode_index']
        idx=index_capture_records(sum((load_rank_records(refs/'hooks',variant,key,rank) for rank in (0,1)),[]))
        captures[key]=(idx,extract_layer_tensors(idx,sample_key=key,decode_index=di))
    rows=[]
    with torch.no_grad():
        for layer in (0,1,19,31,39,41,43,47):
            st=tr.load_qwen3_moe_layer_state(snapshot,layer,device)
            try:
                native=tr.NativeQwen3MoELayerRuntime(st,is_causal=True).to(device).eval()
                diag=tr.build_moe_diag_state(st.spec,'fusable').to(device)
                for p in diag.parameters():p.requires_grad_(False)
                student=tr.StudentQwen3MoELayerRuntime(st,diag,use_r64=False,rot_order='diag_then_r64',is_causal=True).to(device).eval()
                for meta in states:
                    key=meta['sample_key'];di=meta['decode_index'];idx,t=captures[key]
                    def ref(boundary,role,rank=0):return idx[(key,di,layer,boundary,role,rank)].reshape(1,-1).to(device)
                    row={k:meta[k] for k in ('sample_key','calibration_sample_id','source','prefix_length_j')}
                    row.update(layer=layer,variant=variant)
                    # Compare the training norm and production kernel on the SAME rounded residual.
                    for tag,hidden,weight,truth in [
                        ('input_norm',t['R'][layer],st.input_layernorm_weight,ref('input_norm','normalized')),
                        ('post_attn_norm',t['R_A'][layer],st.post_attention_layernorm_weight,ref('post_attn_norm','normalized'))]:
                        x=hidden.reshape(1,-1).to(device)
                        training=tr._rms_norm(x,weight,1e-6)
                        cuda=torch.empty_like(x);ops.rms_norm(cuda,x,weight,1e-6)
                        float_weight=(x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)*weight.float()).to(x.dtype)
                        row[tag]={'training_vs_actual':difference(training,truth),
                                  'cuda_norm_vs_actual':difference(cuda,truth),
                                  'float_weight_vs_actual':difference(float_weight,truth)}
                    # Reproduce the actual fused residual+norm using its TWO captured inputs.
                    a=ref('o_proj','tp_reduced').clone();r=t['R'][layer].reshape(1,-1).to(device).clone()
                    ops.fused_add_rms_norm(a,r,st.post_attention_layernorm_weight,1e-6)
                    row['fused_post_attn_norm']={'normalized':difference(a,ref('post_attn_norm','normalized')),
                                                 'residual':difference(r,t['R_A'][layer])}
                    x=ref('post_attn_norm','normalized')
                    moe=(native.routed_moe(x) if variant=='E0' else tr._student_local_moe(student,x,use_ste=False))
                    row['same_input_router']=difference(moe.router_logits,ref('router_logits','logits'))
                    row['same_input_moe']=difference(moe.output,ref('moe_out','tp_reduced'))
                    # O projection receives TP-local attention outputs; concatenate in TP rank order.
                    x_o=torch.cat([ref('attention_core','rank_local',rank) for rank in (0,1)],dim=-1)
                    if variant=='E0':
                        y=native_nvfp4_linear(x_o,st.attention['o_proj'],st.attention_metadata['o_proj'].input_global_scale_inv)
                    else:
                        y=forward_student_attention_proj('o_proj',x_o,st.attention['o_proj'],diag,use_r64=False,
                            rot_order='diag_then_r64',step_cache=StudentStepCache.new(),use_ste=False,head_dim=st.spec.head_dim)
                    row['same_input_o_proj']=difference(y,ref('o_proj','tp_reduced'))
                    rows.append(row)
                print(f'{variant}: local layer {layer} complete',flush=True)
            finally:tr.release_qwen3_moe_layer_state(st)
        norm,lm=tr._load_final_norm_and_lm_head(snapshot,device)
        logits_rows=[]
        for meta in states:
            key=meta['sample_key'];di=meta['decode_index'];idx,t=captures[key]
            x=t['final_hidden'].reshape(1,-1).to(device)
            actual=load_raw_logits(refs/'raw_logits',variant,key,di)
            logits_rows.append({'sample_key':key,'bf16_linear':difference(F.linear(x,lm),actual),
                               'fp32_linear':difference(F.linear(x.float(),lm.float()),actual)})
        write_jsonl(out/'local_boundaries.jsonl',rows);write_jsonl(out/'lm_head_diagnostic.jsonl',logits_rows)
    summary={'variant':variant,'status':'COMPLETE','states':len(states),'layers':[0,1,19,31,39,41,43,47],
             'interpretation':'diagnostic only: same actual input; no new attention history or semantic inference replay'}
    atomic_write_json(out/'complete.json',summary)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--variant',choices=['E0','E1'],required=True);p.add_argument('--model-path',default=DEFAULT_MODEL_PATH)
    a=p.parse_args();local_worker(a.root,a.variant,a.model_path)
