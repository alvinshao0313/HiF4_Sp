"""Immutable v3 protocol, confirmation cohort, and fail-closed training gate."""
from __future__ import annotations

import hashlib
import random
from pathlib import Path

import torch

from .attention_semantics import TRAINING_PATH_VERSION
from .candidate_runtime import sha256
from .config import S1K_SHARED_CALIB, WIKITEXT2_SHARED_CALIB
from .router_teacher_cache import load_objective_calibration_samples
from .run_state import read_json, read_jsonl, atomic_write_json

PHASE = 'corrected_objectives_v3'
PILOT = {31:('O0','O1_M','O2_M','O3_full'),41:('O0','O1_A','O2_A','O4')}


def phase_root(root: Path) -> Path:
    return Path(root)/'60_objective'/PHASE


def token_hash(ids) -> str:
    return hashlib.sha256(torch.as_tensor(ids,dtype=torch.int64).contiguous().numpy().tobytes()).hexdigest()


def write_frozen(path: Path, payload: dict) -> None:
    if path.exists():
        if read_json(path)!=payload:
            raise RuntimeError(f'immutable protocol mismatch: {path}')
    else:
        atomic_write_json(path,payload)


def prepare_protocol(root: Path) -> dict:
    root=Path(root).resolve();out=phase_root(root)
    split=read_json(root/'00_protocol/objective_split_manifest.json')
    if split['status']!='FROZEN': raise RuntimeError('objective split is not frozen')
    ids=split['train_ids']+split['val_ids']
    samples=load_objective_calibration_samples(ids)
    hashes={s.sample_id:token_hash(s.input_ids) for s in samples}
    excluded_ids=set(ids)
    excluded_hashes=set(hashes.values())
    excluded_prefixes={token_hash(s.input_ids[:128]) for s in samples}
    for meta in read_jsonl(root/'00_protocol/internal_error_state_cohort.jsonl'):
        excluded_ids.add(meta['calibration_sample_id'])
    excluded_samples=load_objective_calibration_samples(sorted(excluded_ids))
    excluded_hashes.update(token_hash(s.input_ids) for s in excluded_samples)
    excluded_prefixes.update(token_hash(s.input_ids[:128]) for s in excluded_samples)
    confirmation=[]
    rng=random.Random(20260916)
    for source,pool in [('wikitext2',WIKITEXT2_SHARED_CALIB),('s1k_original',S1K_SHARED_CALIB)]:
        by_id={}
        for name in ('train','val'):
            for s in torch.load(pool/'calibration'/f'{name}.pt',map_location='cpu',weights_only=False):
                if s.sample_id in by_id and token_hash(s.input_ids)!=token_hash(by_id[s.sample_id].input_ids):
                    raise RuntimeError('sample ID has inconsistent token content')
                by_id[s.sample_id]=s
        ordered=sorted(by_id);rng.shuffle(ordered)
        selected=[]
        for sid in ordered:
            s=by_id[sid];full=token_hash(s.input_ids);prefix=token_hash(s.input_ids[:128])
            if sid in excluded_ids or len(s.input_ids)<129 or full in excluded_hashes or prefix in excluded_prefixes: continue
            selected.append({'sample_id':sid,'source':source,'token_sha256':full,'prefix128_sha256':prefix,'length':len(s.input_ids)})
            excluded_hashes.add(full);excluded_prefixes.add(prefix)
            if len(selected)==32:break
        if len(selected)!=32: raise RuntimeError(f'insufficient untouched confirmation samples for {source}')
        confirmation.extend(selected)
    write_frozen(out/'confirmation_manifest.json',{'status':'FROZEN','seed':20260916,'samples':confirmation,
        'prefix_lengths':[8,32,64,128], 'primary_pairs':[['L31_O3_full','L31_O2_M'],['L41_O4','L41_O2_A']],
        'endpoint':'actual-vLLM exact KL; residual NMSE is a separate mechanism endpoint',
        'bootstrap':{'stratify':'source','unit':'sample mean of four states','replicates':50000,'seed':20260916},
        'multiplicity':'Bonferroni simultaneous two-sided intervals across all prespecified KL contrasts; family alpha=0.05',
        'additional_training_seeds':[20260910,20260911],
        'excluded_ids':sorted(excluded_ids),'no_outcome_selection':True})
    recipes=[]
    for path in sorted((root/'60_objective/objective_candidates').glob('L*/*/recipe.json')):
        original=read_json(path); layer=int(original['layer']);loss=original['loss']
        recipe={**original,'training_phase':PHASE,'training_path_version':TRAINING_PATH_VERSION,
                'steps':320,'output_dir':str(out/'candidates'/f'L{layer}'/loss),
                'sample_token_sha256':hashes,'original_recipe_sha256':sha256(path),
                'actual_teacher_root':str(out/'actual_teacher')}
        if loss in ('O3_full','O3_topk'): recipe['router_teacher_cache_dir']=str(out/'actual_teacher/router')
        recipe_path=out/'recipes'/f'L{layer}_{loss}.json'
        write_frozen(recipe_path,recipe)
        recipes.append({'layer':layer,'loss':loss,'recipe':str(recipe_path),'sha256':sha256(recipe_path),
                        'batch':'pilot' if loss in PILOT.get(layer,()) else 'remaining'})
    if len(recipes)!=23 or sum(r['batch']=='pilot' for r in recipes)!=8:raise RuntimeError('incorrect 8+15 candidate matrix')
    manifest={'phase':PHASE,'training_path_version':TRAINING_PATH_VERSION,'run_root':str(root),
              'gpus':[2,3],'recipes':recipes,'split_sha256':sha256(root/'00_protocol/objective_split_manifest.json'),
              'confirmation_sha256':sha256(out/'confirmation_manifest.json'),'sample_token_sha256':hashes,
              'longest_sample':max(({'sample_id':s.sample_id,'length':len(s.input_ids)} for s in samples),key=lambda x:x['length']),
              'no_automatic_s5':True}
    write_frozen(out/'protocol.json',manifest)
    return manifest


def require_training_ready(root: Path, recipe: dict) -> None:
    if recipe.get('training_phase') == 'mechanism_driven_v4':
        from .mechanism_phase import require_training_ready as require_mechanism_ready
        return require_mechanism_ready(root, recipe)
    if recipe.get('training_phase')!=PHASE or recipe.get('training_path_version')!=TRAINING_PATH_VERSION:
        raise RuntimeError('new training requires an explicit current phase and path version')
    out=phase_root(root).resolve()
    dest=Path(recipe['output_dir']).resolve()
    if not dest.is_relative_to(out/'candidates'):raise RuntimeError('candidate must remain inside corrected phase')
    manifest=read_json(out/'protocol.json')
    entry=next((r for r in manifest['recipes'] if r['layer']==int(recipe['layer']) and r['loss']==recipe['loss']),None)
    if entry is None or read_json(Path(entry['recipe']))!=recipe or sha256(Path(entry['recipe']))!=entry['sha256']:
        raise RuntimeError('recipe differs from frozen protocol')
    gate=read_json(out/'alignment_gate.json')
    if gate.get('status')!='PASS' or gate.get('protocol_sha256')!=sha256(out/'protocol.json'):
        raise RuntimeError('training blocked: actual-path alignment has not passed')
    teacher=read_json(out/'actual_teacher/manifest.json')
    if teacher.get('status')!='COMPLETE' or teacher.get('execution_path')!='actual_vllm_tp2_forced_history':
        raise RuntimeError('training requires complete actual-path teacher states')
    if teacher.get('sample_token_sha256')!=manifest['sample_token_sha256']:
        raise RuntimeError('teacher token provenance differs')


def completion_record(directory: Path, recipe: dict) -> dict:
    names=('checkpoint.pt','train_metrics.jsonl','val_metrics.json','cost.json','recipe.json')
    if recipe.get('training_path_version')==4:
        names += ('training_provenance.json',)
    return {'status':'COMPLETE','training_path_version':recipe.get('training_path_version', TRAINING_PATH_VERSION),'recipe':recipe,
            'sha256':{name:sha256(directory/name) for name in names}}


def validated_completion(directory: Path, recipe: dict) -> bool:
    path=directory/'complete.json'
    if not path.exists():return False
    if read_json(path)!=completion_record(directory,recipe):raise RuntimeError('completed recipe content/provenance changed')
    return True
