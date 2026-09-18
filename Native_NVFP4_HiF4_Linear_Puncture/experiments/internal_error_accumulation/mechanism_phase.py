"""Versioned mechanism study; numerical parity is not a training gate."""
from __future__ import annotations
from pathlib import Path
from .candidate_runtime import sha256
from .corrected_phase import write_frozen
from .run_state import read_json, atomic_write_json

PHASE = 'mechanism_driven_v4'
VERSION = 4


def phase_root(root):
    return Path(root)/'60_objective'/PHASE


def prepare_protocol(root):
    root = Path(root).resolve()
    out = phase_root(root)
    old = root/'60_objective/corrected_objectives_v3'
    original = read_json(old/'protocol.json')
    split = read_json(root/'00_protocol/objective_split_manifest.json')
    recipes = []
    for entry in original['recipes']:
        r = read_json(Path(entry['recipe']))
        cid = f'L{r["layer"]}_{r["loss"]}'
        r.update(training_phase=PHASE, training_path_version=VERSION,
                 candidate_id=cid, upstream_error_weight=1.0,
                 frozen_suffix_layers=0, history_kv_weight=0.0,
                 actual_teacher_root=str(out/'actual_teacher'),
                 output_dir=str(out/'candidates'/cid))
        if r['loss'].startswith('O3'):
            r['router_teacher_cache_dir'] = str(out/'actual_teacher/router')
        recipes.append((r, entry['batch']))
        if (int(r['layer']), r['loss']) in [(31, 'O2_M'), (41, 'O2_A')]:
            for alpha, tag in [(0., 'alpha0'), (.5, 'alpha05')]:
                rid = cid+'_'+tag
                recipes.append(({**r, 'candidate_id': rid, 'upstream_error_weight': alpha,
                                 'output_dir': str(out/'candidates'/rid)}, 'coefficient_control'))
    entries = []
    for recipe, batch in recipes:
        path = out/'recipes'/f'{recipe["candidate_id"]}.json'
        write_frozen(path, recipe)
        entries.append({'candidate_id': recipe['candidate_id'], 'batch': batch,
                        'path': str(path), 'sha256': sha256(path)})
    if len(entries) != 27:
        raise RuntimeError('expected 23 original candidates + four coefficient controls')
    confirmation = read_json(old/'confirmation_manifest.json')
    # Preserve IDs without prematurely freezing a comparison family that
    # excludes conditional prototypes. No confirmation results are read here.
    write_frozen(out/'confirmation_cohort.json', {
        'samples': confirmation['samples'], 'prefix_lengths': confirmation['prefix_lengths'],
        'source_manifest_sha256': sha256(old/'confirmation_manifest.json'),
        'bootstrap': confirmation['bootstrap'],
        'additional_training_seeds': confirmation['additional_training_seeds'],
        'comparison_manifest_status': 'NOT_YET_FROZEN', 'no_outcome_selection': True})
    manifest = {
        'phase': PHASE, 'training_path_version': VERSION, 'recipes': entries,
        'sample_token_sha256': original['sample_token_sha256'],
        'train_ids': split['train_ids'], 'val_ids': split['val_ids'],
        'longest_sample': original['longest_sample'], 'gpus': [2, 3],
        'parent_protocol_sha256': sha256(old/'protocol.json'),
        'known_premise': 'local MSE improvement does not reliably predict final KL improvement',
        'scope': 'fixed-token-history NVFP4 to HiF4 error propagation across depth',
        'numerical_parity_is_training_gate': False,
        'production_interventions_require_noop': True,
        'teacher_execution_path': 'actual_vllm_tp2_teacher_forced_prefill',
        'teacher_path_note': 'all supplied tokens are causal prompt positions; batch/decode numerical differences are measured, not semantic substitutions',
        'direction_seeds': [20260916, 20260917, 20260918, 20260919],
        'direction_strengths': [0., .25, .5, 1.], 'no_automatic_s5': True,
        'conditional_prototypes': {'window4': {'suffix_layers': 4}, 'history': {'weight': .1}},
    }
    write_frozen(out/'protocol.json', manifest)
    return manifest


def require_training_ready(root, recipe):
    out = phase_root(root)
    if recipe.get('training_phase') != PHASE or recipe.get('training_path_version') != VERSION:
        raise RuntimeError('invalid mechanism training protocol')
    protocol = read_json(out/'protocol.json')
    entry = next((r for r in protocol['recipes'] if r['candidate_id'] == recipe['candidate_id']), None)
    if entry is None or sha256(Path(entry['path'])) != entry['sha256'] or read_json(Path(entry['path'])) != recipe:
        raise RuntimeError('recipe is not the frozen mechanism recipe')
    if not Path(recipe['output_dir']).resolve().is_relative_to((out/'candidates').resolve()):
        raise RuntimeError('candidate output escapes the current phase')
    gate = read_json(out/'objective_correctness_gate.json')
    if gate.get('status') != 'PASS' or gate.get('protocol_sha256') != sha256(out/'protocol.json'):
        raise RuntimeError('objective correctness has not passed')
    checks=gate.get('checks',{})
    if not gate.get('required') or set(checks)!=set(gate['required']):
        raise RuntimeError('objective gate evidence coverage is incomplete')
    for name,check in checks.items():
        if check.get('status')!='PASS' or not check.get('evidence_sha256'):
            raise RuntimeError(f'objective gate evidence missing: {name}')
        for path,digest in check['evidence_sha256'].items():
            if sha256(Path(path))!=digest:
                raise RuntimeError(f'objective gate evidence changed: {name}')
    if not gate.get('source_sha256'):
        raise RuntimeError('objective gate must bind the validated source version')
    for path,digest in gate['source_sha256'].items():
        if sha256(Path(path))!=digest:
            raise RuntimeError('training source changed after objective validation')
    teacher = read_json(out/'actual_teacher/manifest.json')
    if teacher.get('status') != 'COMPLETE' or teacher.get('sample_token_sha256') != protocol['sample_token_sha256']:
        raise RuntimeError('actual teacher is incomplete or token provenance differs')
    if teacher.get('execution_path') != protocol['teacher_execution_path']:
        raise RuntimeError('teacher execution path differs from protocol')
    if teacher.get('training_path_version') != VERSION:
        raise RuntimeError('teacher path version differs from the training phase')


def initial_gates(root):
    out = phase_root(root)
    # These are explicitly incomplete requirements, not an exact-match gate.
    path = out/'objective_correctness_gate.json'
    if not path.exists():
        atomic_write_json(path, {'status': 'PENDING', 'protocol_sha256': sha256(out/'protocol.json'),
            'required': ['causal_and_objective_boundaries', 'moe_training_backward',
                         'longest_training_forward_backward', 'nonzero_diag_export_structure',
                         'actual_teacher_complete'],
            'numerical_parity_required': False, 'training_started': False})


def training_provenance(root, recipe):
    """Bind the executed training source separately from teacher acquisition."""
    out=phase_root(root)
    code=Path(__file__).parent
    files=list((code.parent/'e2e_diag_reconstruction/core').glob('*.py'))
    files += [code/name for name in ('selected_layer_objective_trainer.py','mechanism_objectives.py',
              'actual_teacher.py','attention_semantics.py','router_objective.py','router_teacher_cache.py',
              'mechanism_phase.py')]
    return {'protocol_sha256':sha256(out/'protocol.json'),
            'teacher_manifest_sha256':sha256(Path(recipe['actual_teacher_root'])/'manifest.json'),
            'objective_gate_sha256':sha256(out/'objective_correctness_gate.json'),
            'sample_token_sha256':recipe['sample_token_sha256'],
            'source_sha256':{str(path.resolve()):sha256(path) for path in sorted(files)},
            'training_path_version':VERSION,'candidate_id':recipe['candidate_id']}
