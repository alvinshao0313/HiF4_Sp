"""S5: train progressive current-state Top-K candidates, then actual-path validation."""
from __future__ import annotations

from pathlib import Path

from .candidate_runtime import materialize_candidate, sha256
from .run_state import atomic_write_json, read_json
from .variant_validation import run_variant_structural_validation


def validate_topk_selection(gate: dict, scope: dict) -> tuple[list[int], list[int], dict[str, str]]:
    if gate.get('status') != 'ALLOW_TOPK':
        raise RuntimeError(f"S5 requires objective_expansion_gate.json status=ALLOW_TOPK, got {gate.get('status')}")
    ranked = [int(x) for x in gate['topk_layers']]
    ks = [int(x) for x in gate['k_values']]
    losses = gate['objective_by_layer']
    if not ranked or len(set(ranked)) != len(ranked):
        raise ValueError('Top-K requires unique ranked layers')
    if not ks or len(set(ks)) != len(ks) or any(k not in (4, 8) or k > len(ranked) for k in ks):
        raise ValueError('K must be 4 or 8 and cannot exceed approved layer count')
    if not set(ranked) <= set(scope['selected_objective_layers']):
        raise ValueError('Top-K layers outside frozen objective scope')
    if set(losses) != {str(x) for x in ranked}:
        raise ValueError('must specify an objective for every ranked layer')
    for layer in ranked:
        loss = losses[str(layer)]
        expected_o2 = 'O2_A' if scope['causal_source_by_layer'][str(layer)] == 'attention' else 'O2_M'
        if loss != expected_o2 and not (loss in ('O3_full', 'O3_topk') and layer in scope['router_objective_layers']):
            raise ValueError(f'ineligible Top-K objective at L{layer}: {loss}')
    return ranked, ks, losses


def _sequential_current_state_recapture(*, run_root: Path, topk_layers: list[int], k: int) -> dict:
    if k > len(topk_layers) or k <= 0 or len(set(topk_layers)) != len(topk_layers):
        raise ValueError('invalid Top-K selection')
    # Select by approved importance ranking FIRST, then train in model order.
    ordered = sorted(topk_layers[:k])
    return {'current_state_recapture': True, 'k': k, 'ordered_layers': ordered,
            'steps': [{'step': i, 'adopt_layer': layer, 'recapture_upstream_for_layers': ordered[i+1:],
                       'forbid_frozen_baseline_residuals': True} for i, layer in enumerate(ordered)]}


def require_structural_completion(run_root: Path, structural: dict, expected: set[str]) -> None:
    root = run_root / '70_variant_validation'
    results = root / 'variant_validation_results.json'
    report = root / 'E1_E4_STRUCTURAL_VALIDATION.md'
    if not results.is_file() or not report.is_file():
        raise RuntimeError('S5 incomplete: missing actual structural results/report')
    if read_json(results) != structural or any(word in report.read_text() for word in ('Pending', 'PLAN_ONLY')):
        raise RuntimeError('S5 incomplete: structural results are stale or plan-only')
    if set(structural['variants']) != expected:
        raise RuntimeError('S5 incomplete: missing baseline/candidate variant')
    for name, metrics in structural['variants'].items():
        marker = root / 'captures' / name / 'validation_complete.json'
        if not marker.is_file():
            raise RuntimeError(f'S5 incomplete: no actual capture for {name}')
        capture = read_json(marker)
        if (metrics.get('status') != 'COMPLETE' or metrics.get('n_states') != 64
                or capture.get('status') != 'COMPLETE' or capture.get('n_states') != 64):
            raise RuntimeError(f'S5 incomplete: {name} missing 64-state actual validation')


def run_topk_and_variant_validate(*, run_root: Path, model_path: str, phasea_root: Path) -> dict:
    import subprocess
    import sys
    from .selected_layer_objective_trainer import recipe_artifacts_complete

    run_root = Path(run_root)
    gate_path = run_root / '60_objective/objective_expansion_gate.json'
    gate = read_json(gate_path) if gate_path.is_file() else {'status': 'MISSING'}
    if gate.get('status') != 'ALLOW_TOPK':
        raise RuntimeError(f"S5 requires objective_expansion_gate.json status=ALLOW_TOPK, got {gate.get('status')}")
    scope_path = run_root / '60_objective/objective_scope.json'
    if not scope_path.is_file():
        raise RuntimeError('S5 incomplete: missing frozen objective scope')
    scope = read_json(scope_path)
    if scope.get('status') != 'FROZEN':
        raise RuntimeError('S5 incomplete: objective scope not frozen')
    ranked, ks, losses = validate_topk_selection(gate, scope)
    candidates = []
    for k in ks:
        root = run_root / '60_objective/topk_candidates' / f'K{k}'
        protocol = _sequential_current_state_recapture(run_root=run_root, topk_layers=ranked, k=k)
        root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(root / 'recapture_protocol.json', protocol)
        upstream = {}
        for layer in protocol['ordered_layers']:
            loss = losses[str(layer)]
            original = run_root / f'60_objective/objective_candidates/L{layer}/{loss}/recipe.json'
            recipe = {**read_json(original), 'output_dir': str(root / f'L{layer}'),
                      'upstream_checkpoints': dict(upstream)}
            recipe_path = root / f'recipe_L{layer}.json'
            if recipe_path.exists() and read_json(recipe_path) != recipe:
                raise RuntimeError('progressive recipe changed on resume')
            atomic_write_json(recipe_path, recipe)
            # A separate process releases all training CUDA memory before TP2 validation.
            with (run_root / f'logs/topk_K{k}_L{layer}.log').open('a') as log:
                subprocess.run([sys.executable, '-u', '-m', __name__, '--recipe', str(recipe_path),
                                '--run-root', str(run_root), '--model-path', model_path,
                                '--phasea-root', str(phasea_root)], stdout=log, stderr=subprocess.STDOUT, check=True)
            out = Path(recipe['output_dir'])
            if not recipe_artifacts_complete(out):
                raise RuntimeError('Top-K training missing actual artifacts')
            recapture = read_json(out / 'current_state_propagation.json')
            expected_samples = set(recipe['train_ids']) | set(recipe['val_ids'])
            if (recapture['status'] != 'COMPLETE' or recapture['upstream_checkpoints'] != upstream
                    or set(recapture['input_sha256_by_sample']) != expected_samples):
                raise RuntimeError('Top-K current-state propagation provenance mismatch')
            checkpoint = out / 'checkpoint.pt'
            upstream[str(layer)] = {'path': str(checkpoint), 'sha256': sha256(checkpoint)}
        model_dir = materialize_candidate(checkpoints={int(l): Path(v['path']) for l,v in upstream.items()},
                                           output_dir=root / 'model', model_path=model_path, phasea_root=phasea_root)
        candidate = {'name': f'TOPK_K{k}', 'model_dir': str(model_dir), 'checkpoints': upstream}
        candidates.append(candidate)
        atomic_write_json(root / 'candidate_complete.json', {**candidate, 'status': 'COMPLETE'})
    structural = run_variant_structural_validation(run_root=run_root, model_path=model_path,
                    phasea_root=phasea_root, variants=['E1','E2','E3','E4'], best_candidate_variants=candidates)
    expected = {'E1','E2','E3','E4'} | {c['name'] for c in candidates}
    require_structural_completion(run_root, structural, expected)
    manifest = {'status': 'COMPLETE', 'current_state_recapture': True, 'k_values': ks,
                'topk_layers': ranked, 'candidates': candidates}
    atomic_write_json(run_root / '60_objective/topk_current_state_manifest.json', manifest)
    return {'topk': manifest, 'gate': gate, 'structural': structural}


if __name__ == '__main__':
    import argparse
    from .selected_layer_objective_trainer import train_selected_layer_recipe
    p = argparse.ArgumentParser()
    for name in ('recipe','run-root','model-path','phasea-root'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    train_selected_layer_recipe(recipe=read_json(Path(a.recipe)), run_root=Path(a.run_root),
                                model_path=a.model_path, phasea_root=Path(a.phasea_root))
