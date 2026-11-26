"""Real-GPU regression checks for corrected objective boundaries and causal SDPA."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from . import selected_layer_objective_trainer as tr
from .corrected_phase import phase_root
from .run_state import read_json,atomic_write_json
from .path_audit import difference
from .config import DEFAULT_MODEL_PATH


def run(root:Path,kind:str,model_path:str):
    torch.set_num_threads(4)
    out=phase_root(root);protocol=read_json(out/'protocol.json')
    sid=protocol['longest_sample']['sample_id']
    sample=tr.load_objective_calibration_samples([sid])[0]
    snapshot=Path(tr.resolve_local_snapshot(model_path));device=torch.device('cuda:0')
    tokenizer=tr.AutoTokenizer.from_pretrained(snapshot,trust_remote_code=True)
    collator=tr.DynamicCalibrationCollator(tokenizer.pad_token_id)
    cache=tr.build_initial_moe_hidden_cache(snapshot,[sample],collator,device,1)
    h=cache.get(sid).unsqueeze(0).to(device)
    state=tr.load_qwen3_moe_layer_state(snapshot,0,device)
    result={'kind':kind,'longest_sample':sid,'length':h.shape[1]}
    try:
        if kind=='native':
            runtime=tr.NativeQwen3MoELayerRuntime(state,is_causal=True).to(device).eval()
            def parts(x,ste=False):
                return tr._native_layer_parts(runtime,x,tr.build_qwen3_moe_layer_call(str(snapshot),x).position_embeddings,attention_mask=None)
        else:
            diag=tr.build_moe_diag_state(state.spec,'fusable').to(device)
            runtime=tr.StudentQwen3MoELayerRuntime(state,diag,use_r64=False,rot_order='diag_then_r64',is_causal=True).to(device).eval()
            def parts(x,ste=False):
                return tr._student_layer_parts(runtime,x,tr.build_qwen3_moe_layer_call(str(snapshot),x).position_embeddings,attention_mask=None,use_ste=ste)
        with torch.no_grad():
            x=h[:,:32];changed=x.clone();changed[:,16:]=changed[:,16:].flip(1)
            y=parts(x)
            future=difference(parts(changed)['output'][:,:16],y['output'][:,:16])
            if not future['equal']:raise RuntimeError('future token leakage')
            result['future_invariance']=future
            # Same padded batch layout, vary only the first sample's padding.
            batch=x.repeat(2,1,1);batch[0,16:]=0
            padding_changed=batch.clone();padding_changed[0,16:]=h[0,32:48]
            a,b=parts(batch),parts(padding_changed)
            pad=difference(a['output'][0,:16],b['output'][0,:16])
            if not pad['equal']:raise RuntimeError('padding influenced valid outputs')
            result['right_padding_invariance']=pad
            # Batch/sequence GEMM rounding differences are measured separately.
            result['single_vs_batch']=difference(y['output'][0,:16],a['output'][0,:16])
            if kind=='student':
                norm=tr._rms_norm(x,state.input_layernorm_weight,1e-6)
                raw=runtime.attention(norm,None,tr.build_qwen3_moe_layer_call(str(snapshot),x).position_embeddings,tr.StudentStepCache.new(),use_ste=False)
                if not torch.equal(raw,y['attn_branch']):raise RuntimeError('O1_A is not the raw attention output')
                result['o1_a_raw_branch_equal']=True
                result['old_subtraction_difference']=difference(y['post_attn']-x,raw)
                e0=tr.NativeQwen3MoELayerRuntime(state,is_causal=True).to(device).eval()
                e0parts=tr._native_layer_parts(e0,x,tr.build_qwen3_moe_layer_call(str(snapshot),x).position_embeddings,attention_mask=None)
                shared=e0parts['moe_input']
                first=tr._student_local_moe(runtime,shared,use_ste=False).output
                diag.z_qkv.fill_(0.125);diag.z_vo.fill_(-0.125)
                second=tr._student_local_moe(runtime,shared,use_ste=False).output
                if not torch.equal(first,second):raise RuntimeError('O1_M depends on student attention parameters')
                diag.z_qkv.zero_();diag.z_vo.zero_()
                result['o1_m_attention_independent']=True
            torch.cuda.reset_peak_memory_stats()
            long_out=parts(h)['output']
            if long_out.shape!=h.shape or not torch.isfinite(long_out).all():raise RuntimeError('longest sequence forward failed')
            result['longest_forward']={'finite':True,'peak_bytes':torch.cuda.max_memory_allocated()}
            del long_out
    finally:tr.release_qwen3_moe_layer_state(state)
    if kind=='student':
        gradients=[];outputs=[]
        for checkpoint in (False,True):
            x=h[:,:16].detach().clone().requires_grad_(True)
            y=tr._o4_subsequent_hif4(snapshot=snapshot,hidden=x,start_layer=42,num_layers=48,device=device,use_checkpoint=checkpoint)
            y.float().square().mean().backward()
            if x.grad is None or not torch.isfinite(x.grad).all() or not x.grad.abs().sum()>0:raise RuntimeError('O4 suffix gradient invalid')
            outputs.append(y.detach().cpu());gradients.append(x.grad.detach().cpu())
            del x,y
        if not torch.equal(outputs[0],outputs[1]) or not torch.equal(gradients[0],gradients[1]):raise RuntimeError('O4 checkpoint changed forward/backward')
        result['o4_six_layer_checkpoint_gradient']={'equal':True,'norm':float(gradients[0].float().norm())}
    result['status']='PASS'
    atomic_write_json(out/f'smoke_{kind}.json',result)
    print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--kind',choices=['native','student'],required=True);p.add_argument('--model-path',default=DEFAULT_MODEL_PATH)
    a=p.parse_args();run(a.root,a.kind,a.model_path)
