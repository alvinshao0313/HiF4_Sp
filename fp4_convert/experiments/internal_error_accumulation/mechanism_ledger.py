"""Reuse actual E0/E1 val captures to separate branch energy and cross terms."""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
import torch
from .candidate_runtime import sha256
from .corrected_phase import token_hash
from .capture_states import load_rank_records
from .residual_ledger import index_capture_records, extract_layer_tensors
from .mechanism_phase import phase_root
from .mechanism_review import sample_statistics
from .run_state import read_json, read_jsonl, atomic_write_json, write_jsonl


def energy_terms(r0, r1, b0, b1, out0, out1):
    r0,r1,b0,b1,out0,out1=[x.detach().cpu().double() for x in (r0,r1,b0,b1,out0,out1)]
    e,d=r1-r0,b1-b0
    rho=(out1-r1-b1)-(out0-r0-b0)
    after=out1-out0
    c=d+rho
    if not torch.equal(after,e+c):
        raise RuntimeError('BF16-derived residual error identity did not close in FP64')
    local=float(d.square().sum());rounding=float(rho.square().sum())
    branch_rounding_cross=float(2*(d*rho).sum())
    upstream_branch_cross=float(2*(e*d).sum())
    upstream_rounding_cross=float(2*(e*rho).sum())
    injection=float(c.square().sum());cross=float(2*(e*c).sum())
    net=float(after.square().sum()-e.square().sum())
    tolerance=32*torch.finfo(torch.float64).eps*max(abs(net),injection,abs(cross),1.)
    if abs(net-injection-cross)>tolerance:
        raise RuntimeError('residual energy identity failed')
    return {'local_energy':local,'rounding_energy':rounding,
            'branch_rounding_cross':branch_rounding_cross,
            'upstream_branch_cross':upstream_branch_cross,
            'upstream_rounding_cross':upstream_rounding_cross,
            'injection_energy':injection,'upstream_cross':cross,'net_energy_change':net,
            'before_energy':float(e.square().sum()),'after_energy':float(after.square().sum()),
            'closure_error':abs(net-injection-cross)}


def run(root):
    torch.set_num_threads(4)
    out=phase_root(root)/'ledger';out.mkdir(parents=True,exist_ok=True)
    protocol=read_json(phase_root(root)/'protocol.json')
    old=Path(root)/'60_objective/actual_holdout'
    states=read_jsonl(old/'cohort.jsonl')
    grouped=defaultdict(list)
    for state in states:grouped[state['calibration_sample_id']].append(state)
    if set(grouped)!=set(protocol['val_ids']) or any(sorted(s['prefix_length_j'] for s in group)!=[8,32,64,128] for group in grouped.values()):
        raise RuntimeError('existing captures differ from the frozen development cohort')
    rows=[];provenance={}
    for meta in states:
        key=meta['sample_key'];sid=meta['calibration_sample_id'];pair=[]
        # The original cohort used JSON serialization; v4 uses int64 bytes.
        # Rehash the recorded tokens with the protocol's specified encoding.
        if token_hash(meta['full_calibration_token_ids'])!=protocol['sample_token_sha256'][sid]:
            raise RuntimeError('existing capture token provenance differs')
        for variant in ('E0','E1'):
            hooks=old/variant/'hooks'
            path=hooks/variant/key/'rank0.pt'
            provenance[str(path)]=sha256(path)
            pair.append(extract_layer_tensors(index_capture_records(load_rank_records(hooks,variant,key,0)),
                        sample_key=key,decode_index=meta['decode_index']))
        a,b=pair
        for layer in range(48):
            for branch,inputs,outputs,delta in [('attention','R','R_A','A'),('moe','R_A','R','M')]:
                end=layer+1 if branch=='moe' else layer
                stats=energy_terms(a[inputs][layer],b[inputs][layer],a[delta][layer],b[delta][layer],
                                   a[outputs][end],b[outputs][end])
                rows.append({**{k:meta[k] for k in ('sample_key','calibration_sample_id','source','prefix_length_j')},
                             'layer':layer,'branch':branch,**stats})
    write_jsonl(out/'energy_terms.jsonl',rows)
    summaries={}
    for layer,branch in [(31,'moe'),(41,'attention')]:
        selected=[r for r in rows if r['layer']==layer and r['branch']==branch]
        summaries[f'L{layer}_{branch}']={k:sample_statistics(selected,k,sample_ids=protocol['val_ids'])
            for k in ('local_energy','injection_energy','upstream_cross','net_energy_change')}
    atomic_write_json(out/'summary.json',{'status':'COMPLETE','states':len(states),'layers':48,
        'branches':2,'sample_count':len(grouped),'protocol_sha256':sha256(phase_root(root)/'protocol.json'),
        'capture_sha256':provenance,'selected':summaries,
        'interpretation':'descriptive energy accounting; neither causal attribution nor additive KL shares',
        'normalization':'unnormalized squared L2; each sample averages its four states',
        'downstream_interventions':'separate pending experiment; no unmatched-cohort join'})
    print({'status':'COMPLETE','rows':len(rows),'samples':len(grouped)},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    run(parser.parse_args().root)
