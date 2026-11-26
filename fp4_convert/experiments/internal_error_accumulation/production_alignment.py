"""Actual TP2 replay and precise MoE decomposition for the v3 alignment gate."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch

from .run_state import atomic_write_json,read_jsonl,write_jsonl
from .corrected_phase import phase_root
from .capture_states import capture_variant_states,load_rank_records,load_raw_logits
from .residual_ledger import index_capture_records,extract_layer_tensors
from .moe_decomposition import moe_three_way_decompose
from .config import DEFAULT_MODEL_PATH,DEFAULT_PHASEA_ROOT
from ..long_trajectory_stability.real_vllm_hooks.build_llm import build_real_vllm
from ..long_trajectory_stability.real_vllm_hooks.format_conversion.production_puncture import (
    _production_call,_both_ranks_pass,identity_metrics,
)

LAYERS=(0,1,19,31,39,41,43,47)


class ProductionAlignmentOp:
    def __init__(self,root: str,variant: str):self.root=root;self.variant=variant

    def __call__(self,model):
        from vllm.distributed import get_tensor_model_parallel_rank,get_tensor_model_parallel_world_size
        rank=int(get_tensor_model_parallel_rank())
        if get_tensor_model_parallel_world_size()!=2:raise RuntimeError('alignment requires TP2')
        root=Path(self.root);out=phase_root(root)/'alignment';states=read_jsonl(out/'cohort.jsonl')
        current=out/self.variant;reference=out/'E0';rows=[];lmrows=[]
        device=model.model.layers[0].input_layernorm.weight.device
        with torch.inference_mode():
            for meta in states:
                key,di=meta['sample_key'],int(meta['decode_index'])
                idx=index_capture_records(load_rank_records(current/'hooks',self.variant,key,rank))
                idx0=index_capture_records(load_rank_records(reference/'hooks','E0',key,rank))
                t=extract_layer_tensors(idx,sample_key=key,decode_index=di,rank=rank)
                t0=extract_layer_tensors(idx0,sample_key=key,decode_index=di,rank=rank)
                for layer in LAYERS:
                    def get(index,boundary,role):return index[(key,di,layer,boundary,role,rank)].reshape(1,-1)
                    for operator,inp,output in [
                        ('qkv',('input_norm','normalized'),('qkv_proj','rank_local')),
                        ('o_proj',('attention_core','rank_local'),('o_proj','tp_reduced')),
                        ('moe',('post_attn_norm','normalized'),('moe_out','tp_reduced'))]:
                        x=get(idx,*inp);actual=get(idx,*output)
                        repeats=[_production_call(model,layer,operator,x) for _ in range(3)]
                        identity=identity_metrics(actual,repeats)
                        identity['both_tp_ranks_pass']=_both_ranks_pass(model,identity['identity_pass'])
                        row={'sample_key':key,'layer':layer,'operator':operator,'variant':self.variant,'rank':rank,'identity':identity}
                        # A failed replay gate blocks conclusions for this operator.
                        if identity['both_tp_ranks_pass'] and operator=='moe':
                            r0=get(idx0,'router_logits','logits');r1=get(idx,'router_logits','logits')
                            noop=[_production_call(model,layer,'moe',x,r1) for _ in range(3)]
                            route_identity=identity_metrics(actual,noop)
                            route_identity['both_tp_ranks_pass']=_both_ranks_pass(model,route_identity['identity_pass'])
                            row['router_noop']=route_identity
                            if route_identity['both_tp_ranks_pass'] and self.variant=='E1':
                                x0=get(idx0,*inp);y0=get(idx0,*output)
                                y00=_production_call(model,layer,'moe',x0,r0)
                                y10=_production_call(model,layer,'moe',x,r0)
                                y11=actual
                                d_ra=t['R_A'][layer].double()-t0['R_A'][layer].double()
                                decomp=moe_three_way_decompose(y0=y0,y00=y00,y10=y10,y11=y11,delta_R_A=d_ra)
                                d_m=(y11.double()-y0.double()).reshape(-1)
                                d_after=(t['R'][layer+1].double()-t0['R'][layer+1].double()).reshape(-1)
                                d_ra=d_ra.reshape(-1)
                                eps=d_after-d_ra-d_m
                                correction=float(2*((d_ra+d_m)*eps).sum()+eps.square().sum())
                                actual_g=float(d_after.square().sum()-d_ra.square().sum())
                                reconstructed=decomp['G_M']+correction
                                tolerance=64*torch.finfo(torch.float64).eps*max(abs(actual_g),abs(reconstructed),float(d_after.square().sum()),1.0)
                                if abs(actual_g-reconstructed)>tolerance:raise RuntimeError('BF16-inclusive MoE energy closure failed')
                                row['three_way']={**decomp,'bf16_rounding_energy_correction':correction,'actual_G_M':actual_g,'runtime_energy_closure_error':abs(actual_g-reconstructed)}
                                vector_dir=out/'three_way_tensors'/key;vector_dir.mkdir(parents=True,exist_ok=True)
                                torch.save({'y0':y0,'y00':y00,'y10':y10,'y11':y11,'delta_R_A':d_ra,'delta_rounding':eps},vector_dir/f'L{layer}_rank{rank}.pt')
                        rows.append(row)
                # Use model.compute_logits: same production head, TP gather and scaling.
                hidden=t['final_hidden'].reshape(1,-1).to(device)
                copies=[model.compute_logits(hidden.clone()) for _ in range(3)]
                if copies[0] is not None:
                    actual_logits=load_raw_logits(current/'raw_logits',self.variant,key,di)
                    lm_identity=identity_metrics(actual_logits.reshape(-1),[v.detach().cpu().reshape(-1) for v in copies])
                else:
                    if rank==0:raise RuntimeError('production LM Head returned no logits on rank 0')
                    lm_identity={'identity_pass':True,'not_gather_rank':True}
                lm_identity['both_tp_ranks_pass']=_both_ranks_pass(model,lm_identity['identity_pass'])
                lmrows.append({'sample_key':key,'rank':rank,'variant':self.variant,'identity':lm_identity})
        write_jsonl(out/f'production_{self.variant}_rank{rank}.jsonl',rows)
        write_jsonl(out/f'lm_head_{self.variant}_rank{rank}.jsonl',lmrows)
        weight=model.lm_head.weight.detach().cpu().contiguous()
        import hashlib
        summary={'rank':rank,'variant':self.variant,'rows':len(rows),'replay_pass':all(r['identity']['both_tp_ranks_pass'] for r in rows),
                 'lm_head_pass':all(r['identity']['both_tp_ranks_pass'] for r in lmrows),
                 'lm_head_weight_sha256':hashlib.sha256(weight.view(torch.uint8).numpy().tobytes()).hexdigest(),
                 'lm_head_weight_shape':list(weight.shape)}
        atomic_write_json(out/f'production_{self.variant}_rank{rank}_summary.json',summary)
        return summary


def run_worker(root:Path,variant:str,model_path:str,phasea_root:Path):
    torch.set_num_threads(4)
    out=phase_root(root)/'alignment'
    llm,runtime=build_real_vllm(variant,model_path=model_path,phasea_root=phasea_root,gpu_memory_utilization=0.65,max_num_seqs=1)
    capture_variant_states(variant=variant,cohort_path=out/'cohort.jsonl',output_root=out/variant,
                          capture_level='core_qkv',llm_and_runtime=(llm,runtime),capture_all_predictors=True)
    replies=llm.apply_model(ProductionAlignmentOp(str(root),variant))
    if len(replies)!=2:raise RuntimeError('missing TP reply')
    atomic_write_json(out/f'production_{variant}_complete.json',{'status':'COMPLETE','replies':replies})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--variant',choices=['E0','E1'],required=True);p.add_argument('--model-path',default=DEFAULT_MODEL_PATH);p.add_argument('--phasea-root',type=Path,default=DEFAULT_PHASEA_ROOT)
    a=p.parse_args();run_worker(a.root,a.variant,a.model_path,a.phasea_root)
