"""Execute the next authorized correctness checks after actual teacher capture."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import torch
from . import selected_layer_objective_trainer as tr
from .actual_teacher import load_actual_final_logits
from .candidate_runtime import sha256
from .config import DEFAULT_MODEL_PATH, DEFAULT_PHASEA_ROOT
from .corrected_phase import write_frozen
from .mechanism_phase import phase_root
from .path_audit import difference
from .run_state import read_json, atomic_write_json, read_jsonl


def layer_teacher(root, sid, variant, layer):
    manifest=read_json(phase_root(root)/'actual_teacher/manifest.json')
    if manifest['status']!='COMPLETE':raise RuntimeError('actual teacher incomplete')
    entry=manifest['samples'][sid][variant]['layers'][str(layer)]
    if sha256(Path(entry['path']))!=entry['sha256']:raise RuntimeError('teacher hash differs')
    return torch.load(entry['path'],map_location='cpu',weights_only=False,mmap=True)


def training_checks(root, model_path):
    torch.set_num_threads(4)
    torch.manual_seed(20260909)
    torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device('cuda:0')  # physical GPU 2 under the authorized mapping
    out=phase_root(root)/'validation';out.mkdir(exist_ok=True)
    protocol=read_json(phase_root(root)/'protocol.json')
    sid=protocol['longest_sample']['sample_id']
    snapshot=Path(tr.resolve_local_snapshot(model_path))
    e1=layer_teacher(root,sid,'E1',41)
    state=tr.load_qwen3_moe_layer_state(snapshot,41,device)
    diag=tr.build_moe_diag_state(state.spec,'fusable').to(device)
    tr._configure_diag_from_params(diag,['D_QKV','D_VO'])
    with torch.no_grad():
        diag.z_qkv.uniform_(-.03,.03);diag.z_vo.uniform_(-.03,.03)
    student=tr.StudentQwen3MoELayerRuntime(state,diag,use_r64=False,rot_order='diag_then_r64',
                        is_causal=True,o_proj_tp_size=2,moe_tp_size=2).to(device)
    short=e1['input'][:32].unsqueeze(0).to(device)
    def attention(x):
        return tr._student_local_attention(student,x,tr.build_qwen3_moe_layer_call(str(snapshot),x).position_embeddings,use_ste=False)
    with torch.no_grad():
        changed=short.clone();changed[:,16:]=changed[:,16:].flip(1)
        a,b=attention(short),attention(changed)
        if not torch.equal(a[:,:16],b[:,:16]):raise RuntimeError('Attention future leakage')
        batch=short.repeat(2,1,1);batch[0,16:]=0
        altered=batch.clone();altered[0,16:]=short[0,16:]
        pa,pb=attention(batch),attention(altered)
        if not torch.equal(pa[0,:16],pb[0,:16]):raise RuntimeError('right padding affects valid positions')
        padding_difference=difference(a[0,:16],pa[0,:16])
    atomic_write_json(out/'attention_semantics.json',{'status':'PASS','future_invariance':True,
        'padding_invariance':True,'single_vs_batch_numerical_difference':padding_difference,
        'source_sha256':sha256(Path(tr.__file__))})
    print('PASS Attention causality and right padding',flush=True)

    # Compare identical frozen suffixes with checkpoint on/off; no parameter
    # is optimized here, and the input gradient must remain connected.
    values=[];grads=[]
    for enabled in (False,True):
        h=e1['output'][:32].unsqueeze(0).to(device).requires_grad_(True)
        y=tr._o4_subsequent_hif4(snapshot=snapshot,hidden=h,start_layer=42,num_layers=48,
                                device=device,use_checkpoint=enabled)
        y.float().square().mean().backward()
        if h.grad is None or not torch.isfinite(h.grad).all() or h.grad.abs().sum()==0:
            raise RuntimeError('O4 frozen suffix lost its input gradient')
        values.append(y.detach().cpu());grads.append(h.grad.detach().cpu())
        del h,y
        torch.cuda.empty_cache()
    if not torch.equal(values[0],values[1]) or not torch.equal(grads[0],grads[1]):
        raise RuntimeError('checkpoint changed identical suffix forward or gradients')
    atomic_write_json(out/'o4_checkpoint.json',{'status':'PASS','layers':[42,43,44,45,46,47],
        'forward_equal':True,'input_gradient_equal':True,'input_gradient_norm':float(grads[0].float().norm()),
        'source_sha256':sha256(Path(tr.__file__))})
    print('PASS O4 six-layer checkpoint forward/gradient equivalence',flush=True)
    del values,grads

    teacher=load_actual_final_logits(phase_root(root)/'actual_teacher',[sid])
    norm,lm=tr._load_final_norm_and_lm_head(snapshot,device)
    for p in diag.parameters():p.grad=None
    torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    print(f'START O4 full longest sample: {sid} tokens={len(e1["input"])}',flush=True)
    loss,n=tr._o4_one_sample_kl(student=student,snapshot=snapshot,x_store={sid:e1['input']},
        sample_id=sid,n_tokens=len(e1['input']),layer=41,num_layers=48,device=device,
        train_mode=True,norm_w=norm,lm_w=lm,e0_final_logits=teacher)
    if not torch.isfinite(loss):raise RuntimeError('O4 longest loss nonfinite')
    print(f'O4 longest forward KL={float(loss.detach())}; backward',flush=True)
    loss.backward()
    gradients={}
    for name,p in diag.named_parameters():
        if p.requires_grad:
            if p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().sum()==0:
                raise RuntimeError(f'invalid longest O4 gradient: {name}')
            gradients[name]=float(p.grad.norm())
        elif p.grad is not None:raise RuntimeError(f'gradient reached frozen parameter: {name}')
    result={'status':'PASS','sample_id':sid,'tokens':n,'loss':float(loss.detach()),
        'gradients':gradients,'seconds':time.monotonic()-started,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),
        'input_source':'actual E1 full causal history','teacher_source':'actual E0 final logits',
        'teacher_manifest_sha256':sha256(phase_root(root)/'actual_teacher/manifest.json'),
        'trainer_sha256':sha256(Path(tr.__file__))}
    atomic_write_json(out/'o4_longest.json',result)
    print(result,flush=True)


def deployment_capture(root, layer, model_path, phasea_root):
    from .capture_states import capture_variant_states
    from .objective_holdout import validate_capture
    from ..long_trajectory_stability.real_vllm_hooks.build_llm import build_real_vllm
    out=phase_root(root)/'validation'/f'deployment_L{layer}'
    model=phase_root(root)/'export_checks'/f'nonzero_L{layer}_model'
    manifest=read_json(model/'candidate_manifest.json')
    for name,digest in manifest['exported_sha256'].items():
        if sha256(model/name)!=digest:raise RuntimeError('candidate export changed')
    cohort=Path(root)/'60_objective/corrected_objectives_v3/alignment/cohort.jsonl'
    llm,runtime=build_real_vllm('E1',model_path=model_path,phasea_root=phasea_root,
            materialized_model_path=model,gpu_memory_utilization=.60,max_num_seqs=1)
    if Path(runtime['model_path']).resolve()!=model.resolve():
        raise RuntimeError('vLLM loaded a different export')
    capture_variant_states(variant='E1',cohort_path=cohort,output_root=out,capture_level='core_qkv',
                          llm_and_runtime=(llm,runtime),capture_all_predictors=True)
    states=read_jsonl(cohort);validate_capture(out,'E1',states)
    atomic_write_json(out/'actual_load_complete.json',{'status':'PASS','runtime':runtime,
        'export_manifest_sha256':sha256(model/'candidate_manifest.json'),
        'cohort_sha256':sha256(cohort),'states':len(states),'tp_replica_and_residual_closure':True,
        'scope':'actual nonzero export load/capture; training-folded boundary comparison still separate'})


def run_validation(*,run_root,model_path,phasea_root):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='2,3':raise RuntimeError('use authorized GPU 2,3 mapping')
    root=Path(run_root).resolve();out=phase_root(root)/'validation';out.mkdir(exist_ok=True)
    teacher=read_json(phase_root(root)/'actual_teacher/manifest.json')
    if teacher['status']!='COMPLETE':raise RuntimeError('actual teacher incomplete')
    write_frozen(out/'execution.json',{'protocol_sha256':sha256(phase_root(root)/'protocol.json'),
        'teacher_manifest_sha256':sha256(phase_root(root)/'actual_teacher/manifest.json'),
        'source_sha256':{str(p):sha256(p) for p in (Path(__file__),Path(tr.__file__),
            Path(tr.__file__).parent.parent/'e2e_diag_reconstruction/core/moe_semantic_hif4.py')},
        'jobs':['training','deployment31','deployment41'],'gpus':[2,3],'automatic_s5':False})
    for job in ('training','deployment31','deployment41'):
        with (out/f'{job}.log').open('x') as log:
            subprocess.run([sys.executable,'-u','-m',__name__,'--root',str(root),'--job',job,
                            '--model-path',model_path,'--phasea-root',str(phasea_root)],
                            stdout=log,stderr=subprocess.STDOUT,check=True)
    atomic_write_json(out/'complete.json',{'status':'COMPLETE','jobs':['training','deployment31','deployment41'],
        'training_gate':'not automatically granted: inspect remaining objective/folding evidence',
        'no_automatic_s5':True})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--job',choices=['training','deployment31','deployment41'],required=True)
    p.add_argument('--model-path',default=DEFAULT_MODEL_PATH)
    p.add_argument('--phasea-root',type=Path,default=DEFAULT_PHASEA_ROOT)
    a=p.parse_args()
    if a.job=='training':training_checks(a.root,a.model_path)
    else:deployment_capture(a.root,int(a.job.removeprefix('deployment')),a.model_path,a.phasea_root)
