"""Training-runtime diagnostics against immutable actual-vLLM captures.

This is not an inference implementation or a replacement for formal actual-path
measurements. Two isolated workers use the existing training runtime unchanged
except for the explicit causal mask. No optimization or new checkpoint occurs.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from . import selected_layer_objective_trainer as tr
from .attention_semantics import causal_attention_mask, TRAINING_PATH_VERSION
from .candidate_runtime import sha256
from .capture_states import load_rank_records, load_raw_logits
from .config import DEFAULT_MODEL_PATH
from .math_utils import exact_kl
from .objective_holdout import _nmse
from .residual_ledger import index_capture_records, extract_layer_tensors
from .run_state import atomic_write_json, read_json, read_jsonl, write_jsonl

CANDIDATES = {'L31_O2_M':31, 'L31_O3_full':31, 'L41_O2_A':41, 'L41_O4':41}


def audit_root(root):
    return root / '60_objective/path_mechanism_audit'


def checkpoint_path(root, label):
    layer, loss = label.split('_', 1)
    return root/'60_objective/objective_candidates'/layer/loss/'checkpoint.pt'


def selected_states(root):
    states = read_jsonl(root/'60_objective/actual_holdout/cohort.jsonl')
    chosen = []
    for source in ('wikitext2', 's1k_original'):
        ids = sorted({r['calibration_sample_id'] for r in states if r['source']==source})
        if len(ids) != 8:
            raise RuntimeError('expected eight existing validation samples per source')
        chosen.extend(ids[:2])
    return [r for r in states if r['calibration_sample_id'] in chosen]


def prepare(root, model_path):
    out = audit_root(root)
    out.mkdir(parents=True, exist_ok=True)
    states = selected_states(root)
    protocol = {'schema':1, 'training_path_version':TRAINING_PATH_VERSION,
                'model_path':model_path, 'sample_selection':'first two sorted IDs per source; no outcome selection',
                'sample_ids':sorted({r['calibration_sample_id'] for r in states}),
                'prefix_lengths':[8,32,64,128], 'diagnostic_sequence_length':128,
                'modes':['legacy_unmasked','causal'], 'gpus':[2,3],
                'candidates':{k:sha256(checkpoint_path(root,k)) for k in CANDIDATES},
                'actual_cohort_sha256':sha256(root/'60_objective/actual_holdout/cohort.jsonl'),
                'scope':'training runtime diagnostic only; existing actual TP2 captures are reference; no retraining or S5',
                'source_sha256':{name:sha256(Path(__file__).parent/name) for name in (
                    'path_audit.py','attention_semantics.py','selected_layer_objective_trainer.py','router_teacher_cache.py')},
                'caveat':'128-token legacy diagnostic does not reproduce original full-sequence training; backend differences remain possible'}
    path = out/'protocol.json'
    if path.exists() and read_json(path) != protocol:
        raise RuntimeError('audit protocol changed; do not silently resume')
    atomic_write_json(path,protocol)
    write_jsonl(out/'cohort.jsonl',states)
    return protocol,states


def initial_cache(root, model_path):
    states = selected_states(root)
    ids = sorted({r['calibration_sample_id'] for r in states})
    samples = [dataclasses.replace(s,input_ids=s.input_ids[:128],loss_mask=s.loss_mask[:128])
               for s in tr.load_objective_calibration_samples(ids)]
    for sample in samples:
        reference = next(r for r in states if r['calibration_sample_id']==sample.sample_id)
        if sample.input_ids.tolist() != reference['full_calibration_token_ids'][:128]:
            raise RuntimeError('diagnostic token history differs from actual capture')
    snapshot = Path(tr.resolve_local_snapshot(model_path))
    tokenizer = tr.AutoTokenizer.from_pretrained(snapshot, trust_remote_code=True)
    collator = tr.DynamicCalibrationCollator(tokenizer.pad_token_id)
    cache = tr.build_initial_moe_hidden_cache(snapshot,samples,collator,torch.device('cuda:0'),1)
    return snapshot,states,cache


def difference(x, ref):
    a,b = x.detach().cpu().reshape(-1),ref.detach().cpu().reshape(-1)
    if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise RuntimeError('nonfinite or incompatible audit tensors')
    return {'equal':torch.equal(a,b),'max_abs':float((a.double()-b.double()).abs().max()),'nmse':_nmse(a,b)}


def smoke(root, model_path, kind):
    torch.set_num_threads(4)
    snapshot,states,cache = initial_cache(root,model_path)
    h = cache.get(states[0]['calibration_sample_id'])[:16].unsqueeze(0).cuda()
    st = tr.load_qwen3_moe_layer_state(snapshot,0,torch.device('cuda:0'))
    result = {'kind':kind,'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')}
    try:
        if kind == 'native':
            runtime = tr.NativeQwen3MoELayerRuntime(st).cuda().eval()
            forward = lambda x,mask: tr._native_layer_parts(runtime,x,tr.build_qwen3_moe_layer_call(str(snapshot),x).position_embeddings,attention_mask=mask)['output']
        else:
            diag = tr.build_moe_diag_state(st.spec,'fusable').cuda()
            for p in diag.parameters(): p.requires_grad_(False)
            runtime = tr.StudentQwen3MoELayerRuntime(st,diag,use_r64=False,rot_order='diag_then_r64').cuda().eval()
            forward = lambda x,mask: tr._student_layer_parts(runtime,x,tr.build_qwen3_moe_layer_call(str(snapshot),x).position_embeddings,attention_mask=mask,use_ste=False)['output']
        perturbed = h.clone()
        perturbed[:,8:] = torch.flip(h[:,8:],dims=[1])
        with torch.no_grad():
            for name,mask in [('legacy_unmasked',None),('causal',causal_attention_mask(h))]:
                original = forward(h,mask)[:,:8]
                repeat = forward(h,mask)[:,:8]
                changed = forward(perturbed,mask)[:,:8]
                result[name] = {'repeat':difference(repeat,original),'future_change':difference(changed,original)}
        if not result['causal']['repeat']['equal'] or not result['causal']['future_change']['equal']:
            raise RuntimeError('causal future-invariance failed')
        if result['legacy_unmasked']['future_change']['equal']:
            raise RuntimeError('legacy future leakage not reproduced')
    finally:
        tr.release_qwen3_moe_layer_state(st)
    if kind == 'student':
        x=h.detach().clone().requires_grad_(True)
        y=tr._o4_subsequent_hif4(snapshot=snapshot,hidden=x,start_layer=47,num_layers=48,device=torch.device('cuda:0'))
        y.float().square().mean().backward()
        if x.grad is None or not torch.isfinite(x.grad).all() or not x.grad.abs().sum()>0:
            raise RuntimeError('corrected O4 downstream input gradient failed')
        result['o4_checkpointed_backward']={'finite':True,'grad_norm':float(x.grad.float().norm())}
    result['status']='PASS'
    atomic_write_json(audit_root(root)/f'smoke_{kind}.json',result)
    print(result,flush=True)


def worker(root, model_path, mode):
    torch.set_num_threads(4)
    out=audit_root(root)/mode
    out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():
        raise RuntimeError('refusing to overwrite completed diagnostic worker')
    snapshot,states,initial=initial_cache(root,model_path)
    ids=sorted({r['calibration_sample_id'] for r in states})
    hidden={v:{sid:initial.get(sid) for sid in ids} for v in ('E0','E1')}
    actual=root/'60_objective/actual_holdout'
    references={}
    for label in ['E0','E1',*CANDIDATES]:
        references[label]={}
        for meta in states:
            key,di=meta['sample_key'],meta['decode_index']
            variant='E0' if label=='E0' else 'E1'
            index=index_capture_records(load_rank_records(actual/label/'hooks',variant,key,0))
            tensors=extract_layer_tensors(index,sample_key=key,decode_index=di)
            references[label][key]=(tensors,index)
    rows=[]
    with torch.no_grad():
        for layer in range(48):
            st=tr.load_qwen3_moe_layer_state(snapshot,layer,torch.device('cuda:0'))
            try:
                native=tr.NativeQwen3MoELayerRuntime(st).cuda().eval()
                for label,start in CANDIDATES.items():
                    if layer==start:
                        hidden[label]=dict(hidden['E1'])
                if layer==42:
                    # Identical L41 output, only subsequent quantization differs.
                    hidden['L41_O4_native_suffix']=dict(hidden['L41_O4'])
                nxt={}
                for label,inputs in hidden.items():
                    is_native=label=='E0' or label.endswith('_native_suffix')
                    if not is_native:
                        diag=tr.build_moe_diag_state(st.spec,'fusable').cuda()
                        if CANDIDATES.get(label)==layer:
                            payload=torch.load(checkpoint_path(root,label),map_location='cpu',weights_only=False)
                            diag.load_snapshot(payload['diag'])
                        for p in diag.parameters(): p.requires_grad_(False)
                        student=tr.StudentQwen3MoELayerRuntime(st,diag,use_r64=False,rot_order='diag_then_r64').cuda().eval()
                    nxt[label]={}
                    ref_label='L41_O4' if label.endswith('_native_suffix') else label
                    for sid in ids:
                        h=inputs[sid].unsqueeze(0).cuda()
                        mask=causal_attention_mask(h) if mode=='causal' else None
                        pe=tr.build_qwen3_moe_layer_call(str(snapshot),h).position_embeddings
                        parts=(tr._native_layer_parts(native,h,pe,attention_mask=mask) if is_native else
                               tr._student_layer_parts(student,h,pe,attention_mask=mask,use_ste=False))
                        nxt[label][sid]=parts['output'][0].detach().cpu()
                        router=parts['router_logits'].reshape(1,128,-1)
                        for meta in states:
                            if meta['calibration_sample_id']!=sid: continue
                            key,di=meta['sample_key'],meta['decode_index']
                            ref,index=references[ref_label][key]
                            row={k:meta[k] for k in ('sample_key','calibration_sample_id','source','prefix_length_j')}
                            row.update(mode=mode,label=label,layer=layer)
                            for boundary,value,target in [
                                ('input',h[0,di],ref['R'][layer]),
                                ('post_attn',parts['post_attn'][0,di],ref['R_A'][layer]),
                                ('output',parts['output'][0,di],ref['R'][layer+1]),
                                ('router',router[0,di],index[(key,di,layer,'router_logits','logits',0)])]:
                                row[boundary]=difference(value,target)
                            rows.append(row)
                    if not is_native: del student,diag
                hidden=nxt
                print(f'{mode} layer={layer} labels={list(hidden)}',flush=True)
                atomic_write_json(out/'progress.json',{'last_completed_layer':layer,'labels':list(hidden)})
            finally:
                tr.release_qwen3_moe_layer_state(st)
        write_jsonl(out/'boundary_comparisons.jsonl',rows)
        norm,lm=tr._load_final_norm_and_lm_head(snapshot,torch.device('cuda:0'))
        logits_rows=[]
        saved={}
        for label,inputs in hidden.items():
            ref_label='L41_O4' if label.endswith('_native_suffix') else label
            variant='E0' if label=='E0' else 'E1'
            for sid in ids:
                h=inputs[sid][[7,31,63,127]].unsqueeze(0).cuda()
                logits=tr._final_logits_from_hidden(h,torch.tensor([4]),norm,lm).cpu()
                for i,j in enumerate((8,32,64,128)):
                    meta=next(r for r in states if r['calibration_sample_id']==sid and r['prefix_length_j']==j)
                    key,di=meta['sample_key'],meta['decode_index']
                    ref=load_raw_logits(actual/ref_label/'raw_logits',variant,key,di)
                    e0=load_raw_logits(actual/'E0/raw_logits','E0',key,di)
                    saved[(label,key)]=logits[i]
                    logits_rows.append({'label':label,'sample_key':key,'source':meta['source'],
                        'calibration_sample_id':sid,'prefix_length_j':j,'mode':mode,
                        'vs_actual_same_candidate':difference(logits[i],ref),
                        'kl_actual_candidate_to_diagnostic':exact_kl(ref,logits[i]),
                        'kl_actual_e0_to_diagnostic':exact_kl(e0,logits[i]),
                        'actual_candidate_kl':exact_kl(e0,ref)})
        torch.save(saved,out/'selected_logits.pt')
        write_jsonl(out/'logit_comparisons.jsonl',logits_rows)
    atomic_write_json(out/'complete.json',{'status':'COMPLETE','mode':mode,'boundary_rows':len(rows),'logit_rows':len(logits_rows)})


def teacher_cache_review(root):
    states=read_jsonl(root/'60_objective/actual_holdout/cohort.jsonl')
    rows=[]
    for layer in (19,31):
        path=root/f'60_objective/router_teacher_cache/L{layer}_val.pt'
        cache=torch.load(path,map_location='cpu',weights_only=False)
        for meta in states:
            key,di=meta['sample_key'],meta['decode_index']
            index=index_capture_records(load_rank_records(root/'60_objective/actual_holdout/E0/hooks','E0',key,0))
            ref=index[(key,di,layer,'router_logits','logits',0)]
            old=cache[meta['calibration_sample_id']]['router_logits'][di]
            rows.append({k:meta[k] for k in ('sample_key','calibration_sample_id','source','prefix_length_j')} |
                        {'layer':layer,'difference':difference(old,ref),'kl_actual_to_cached':exact_kl(ref,old)})
    write_jsonl(audit_root(root)/'original_teacher_cache_comparison.jsonl',rows)
    return rows


def summarize(root):
    out=audit_root(root)
    summary={'status':'COMPLETE','training_equivalence_claim':'NOT_ESTABLISHED',
             'limitations':['four existing samples; not independent validation',
                            'training runtime vs actual TP2 mixes backend and arithmetic differences',
                            'no claim of ideal objective effectiveness from legacy checkpoints'], 'modes':{}}
    lines=['# 训练路径与实际推理审计','','旧 checkpoint 未重训，原有 gate 未修改。',
           '未来 token 泄漏和 O4 后续格式已在代码中修正；下表量化剩余路径差异，不预设已经等价。','',
           '| 模式 | 模型 | 与对应真实推理输出 KL 均值 |', '|---|---|---|']
    for mode in ('legacy_unmasked','causal'):
        rows=read_jsonl(out/mode/'logit_comparisons.jsonl')
        sm={}
        for label in sorted({r['label'] for r in rows}):
            group=[r for r in rows if r['label']==label]
            sm[label]={'kl_actual_candidate_to_diagnostic_mean':sum(r['kl_actual_candidate_to_diagnostic'] for r in group)/len(group),
                       'exact_logit_matches':sum(r['vs_actual_same_candidate']['equal'] for r in group),'states':len(group)}
            lines.append(f"| {mode} | {label} | {sm[label]['kl_actual_candidate_to_diagnostic_mean']:.7g} |")
        summary['modes'][mode]=sm
        logits=torch.load(out/mode/'selected_logits.pt',map_location='cpu',weights_only=False)
        contrasts=[]
        for meta in selected_states(root):
            key=meta['sample_key']
            contrasts.append({'sample_key':key,'hif4_vs_native_suffix':difference(logits[('L41_O4',key)],logits[('L41_O4_native_suffix',key)]),
                              'kl_hif4_to_native_suffix':exact_kl(logits[('L41_O4',key)],logits[('L41_O4_native_suffix',key)])})
        write_jsonl(out/mode/'o4_suffix_contrast.jsonl',contrasts)
    lines += ['','## 结论边界','',
              '- 旧 Router teacher 缓存也来自无因果掩码路径，不能用于修正后的 O3 训练。',
              '- 本次只做路径诊断与机制复盘；没有重新训练，不开启 S5。',
              '- 真实推理结果继续有效，但不能将旧 checkpoint 的成败解释为理想 O2/O3/O4 的成败。',
              '- 因果掩码修正不保证 SDPA/TP1 与 vLLM/TP2 数值等价。剩余偏差必须单独判断。']
    atomic_write_json(out/'summary.json',summary)
    (out/'PATH_AUDIT.md').write_text('\n'.join(lines)+'\n')
    return summary


def run_path_audit(*,run_root,model_path):
    from .mechanism_review import run_mechanism_review
    root=Path(run_root)
    torch.set_num_threads(4)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='2,3':
        raise RuntimeError('this authorized diagnostic requires CUDA_VISIBLE_DEVICES=2,3')
    prepare(root,model_path)
    for kind in ('native','student'):
        if read_json(audit_root(root)/f'smoke_{kind}.json')['status']!='PASS':
            raise RuntimeError('GPU smoke required before detached audit')
    teacher_cache_review(root)
    run_mechanism_review(root)
    processes=[]
    try:
        for gpu,mode in zip(('2','3'),('legacy_unmasked','causal')):
            log=(audit_root(root)/f'{mode}.log').open('w')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4')
            p=subprocess.Popen([sys.executable,'-u','-m',__name__,'--root',str(root),'--model-path',model_path,'--worker',mode],stdout=log,stderr=subprocess.STDOUT,env=env)
            processes.append((p,log,mode))
        atomic_write_json(audit_root(root)/'workers.json',{mode:p.pid for p,_,mode in processes})
        while any(p.poll() is None for p,_,_ in processes):
            for p,_,mode in processes:
                if p.poll() not in (None,0): raise RuntimeError(f'audit worker {mode} exited {p.returncode}')
            time.sleep(2)
        for p,_,mode in processes:
            if p.returncode!=0: raise RuntimeError(f'audit worker {mode} exited {p.returncode}')
    finally:
        for p,log,_ in processes:
            if p.poll() is None: p.terminate();p.wait()
            log.close()
    return summarize(root)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--model-path',default=DEFAULT_MODEL_PATH)
    actions=parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--smoke',choices=['native','student'])
    actions.add_argument('--worker',choices=['legacy_unmasked','causal'])
    args=parser.parse_args()
    if args.smoke: smoke(args.root,args.model_path,args.smoke)
    else: worker(args.root,args.model_path,args.worker)
