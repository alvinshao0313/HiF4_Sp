"""Prespecified sample-level, source-stratified paired confirmation statistics."""
from __future__ import annotations
import math
import torch


def paired_confirmation(baseline:list[dict],candidate:list[dict],manifest:dict,*,metric:str='final_logit_kl',family_size:int|None=None)->dict:
    if family_size is None:
        if manifest.get('comparison_manifest_status')!='FROZEN' or not manifest.get('comparisons'):
            raise RuntimeError('confirmation requires a frozen, nonempty comparison family')
        family_size=len(manifest['comparisons'])
    expected={s['sample_id']:s['source'] for s in manifest['samples']}
    js=sorted(manifest['prefix_lengths'])
    def aggregate(rows):
        by={}
        for row in rows:
            sid=row['calibration_sample_id']
            if sid not in expected or row['source']!=expected[sid]:raise RuntimeError('unexpected confirmation sample/source')
            value=float(row[metric])
            if not math.isfinite(value):raise RuntimeError('nonfinite confirmation metric')
            by.setdefault(sid,[]).append((int(row['prefix_length_j']),value))
        if set(by)!=set(expected):raise RuntimeError('incomplete confirmation sample coverage')
        if any(sorted(j for j,_ in group)!=js for group in by.values()):raise RuntimeError('missing or duplicate confirmation states')
        return {sid:sum(v for _,v in group)/len(js) for sid,group in by.items()}
    b,c=aggregate(baseline),aggregate(candidate)
    sources=sorted(set(expected.values()))
    if family_size<1:raise ValueError('family size must be positive')
    replicates=int(manifest['bootstrap']['replicates']);seed=int(manifest['bootstrap']['seed'])
    g=torch.Generator().manual_seed(seed)
    boot=torch.zeros(replicates,dtype=torch.float64)
    by_source={};gains={sid:b[sid]-c[sid] for sid in sorted(expected)}
    for source in sources:
        x=torch.tensor([gains[s] for s in sorted(expected) if expected[s]==source],dtype=torch.float64)
        if x.numel()<2:raise RuntimeError('source stratum needs at least two samples')
        by_source[source]=float(x.mean())
        idx=torch.randint(len(x),(replicates,len(x)),generator=g)
        boot += (len(x)/len(expected))*x[idx].mean(1)
    alpha=.05/family_size
    lower,upper=torch.quantile(boot,torch.tensor([alpha/2,1-alpha/2],dtype=torch.float64)).tolist()
    return {'metric':metric,'n_samples':len(expected),'states_per_sample':len(js),'mean_gain':sum(gains.values())/len(gains),
            'ci_family_lo':lower,'ci_family_hi':upper,'family_size':family_size,'family_alpha':.05,
            'source_mean_gain':by_source,'positive_samples':sum(v>0 for v in gains.values()),
            'supported':lower>0 and all(v>0 for v in by_source.values()),
            'sample_gains':gains,'bootstrap_replicates':replicates,'seed':seed,
            'scope':'one training seed, paired over samples; training seeds are not new sample units'}
