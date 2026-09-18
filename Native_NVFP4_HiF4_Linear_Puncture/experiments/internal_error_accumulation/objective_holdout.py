"""Independent selected-layer validation on real TP2 vLLM forced histories.

Each engine lives in a separate subprocess. Training losses are never used to
rank objectives. This stage stops for review and cannot authorize Top-K itself.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from .candidate_runtime import materialize_candidate, sha256
from .config import PREFIX_LENGTHS_J, REPLICA_BOUNDARIES
from .run_state import atomic_write_json, read_json, read_jsonl, write_jsonl
from .math_utils import exact_kl, sample_bootstrap_ci


def repair_legacy_validation(run_root: Path) -> None:
    """Versioned, exact correction; original records remain beside corrected files."""
    for path in sorted((run_root / '60_objective/objective_candidates').glob('L*/O*/val_metrics.json')):
        metrics = read_json(path)
        if metrics.get('metrics_schema_version') == 2:
            continue
        archive = path.with_name('val_metrics.schema1.json')
        if archive.exists() and read_json(archive) != metrics:
            raise RuntimeError(f'legacy validation archive mismatch: {path}')
        atomic_write_json(archive, metrics)
        if not metrics['loss_name'].startswith('O3_'):
            metrics['loss'] /= 2.0
        metrics['router_causal_contribution'] = None
        metrics['metrics_schema_version'] = 2
        metrics['evaluation_path'] = 'training_runtime_objective_only'
        metrics['correction'] = 'removed duplicate loss accumulation and stale causal lookup; original in val_metrics.schema1.json'
        atomic_write_json(path, metrics)


def freeze_holdout(run_root: Path) -> Path:
    from .selected_layer_objective_trainer import load_objective_calibration_samples
    from .build_state_cohort import expand_states, _sha256_ids
    split = read_json(run_root / '00_protocol/objective_split_manifest.json')
    if split['status'] != 'FROZEN':
        raise RuntimeError('objective split must be frozen')
    train, val, excluded = (set(split[k]) for k in ('train_ids', 'val_ids', 'excluded_causal_sample_ids'))
    if len(val) != 16 or train & val or (train | val) & excluded:
        raise RuntimeError('invalid independent objective split')
    samples = load_objective_calibration_samples(split['val_ids'])
    selection = {'discovery': [], 'holdout': [], 'prefix_lengths_j': list(PREFIX_LENGTHS_J)}
    for s in samples:
        ids = s.input_ids.tolist()
        if s.sample_id.startswith('s1k_original_'):
            source = 's1k_original'
        elif s.sample_id.startswith('wikitext2_'):
            source = 'wikitext2'
        else:
            raise ValueError(f'unknown calibration source: {s.sample_id}')
        selection['holdout'].append({'calibration_sample_id': s.sample_id, 'source': source,
                                    'source_index': s.source_index, 'input_ids': ids,
                                    'length': len(ids), 'input_ids_sha256': _sha256_ids(ids)})
    if sum(s['source'] == 'wikitext2' for s in selection['holdout']) != 8:
        raise RuntimeError('holdout must be 8 WikiText2 + 8 S1K')
    rows = expand_states(selection)
    out = run_root / '60_objective/actual_holdout'
    out.mkdir(parents=True, exist_ok=True)
    cohort = out / 'cohort.jsonl'
    if cohort.exists() and read_jsonl(cohort) != rows:
        raise RuntimeError('holdout cohort changed')
    write_jsonl(cohort, rows)
    protocol = {'status': 'FROZEN', 'split_sha256': sha256(run_root / '00_protocol/objective_split_manifest.json'),
                'cohort_sha256': sha256(cohort), 'n_samples': 16, 'n_states': 64,
                'prefix_lengths_j': list(PREFIX_LENGTHS_J), 'path': 'actual_vllm_tp2',
                'comparison': 'candidate vs same-layer O0/O1 and E1; 4 states averaged per sample',
                'stability': 'paired sample bootstrap 95% lower bound > 0 for KL and cumulative NMSE; positive mean gain in both sources',
                'expansion': 'recommend only if at least 4 layers qualify; no automatic ALLOW_TOPK',
                'router': 'candidate-specific E0 router freeze, exact candidate noop identity',
                'gpu_ids': '2,3'}
    protocol_path = out / 'protocol.json'
    if protocol_path.exists() and read_json(protocol_path) != protocol:
        raise RuntimeError('holdout protocol changed')
    atomic_write_json(protocol_path, protocol)
    return cohort


def _nmse(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    denom = b.square().sum()
    if denom <= 0:
        raise RuntimeError('zero reference norm')
    return float((a - b).square().sum() / denom)


def validate_capture(root: Path, variant: str, states: list[dict]) -> None:
    from .capture_states import load_rank_records
    from .residual_ledger import index_capture_records, check_tp_replica, extract_layer_tensors, runtime_residual_closure_for_variant
    for meta in states:
        records = sum((load_rank_records(root / 'hooks', variant, meta['sample_key'], rank) for rank in (0, 1)), [])
        index = index_capture_records(records)
        result = check_tp_replica(index, sample_key=meta['sample_key'], decode_index=meta['decode_index'],
                                  boundaries=list(REPLICA_BOUNDARIES), layers=list(range(48)) + [None])
        if not result['pass']:
            raise RuntimeError(f'TP replica identity failed: {result}')
        tensors = extract_layer_tensors(index, sample_key=meta['sample_key'], decode_index=meta['decode_index'])
        runtime_residual_closure_for_variant(tensors, repeatability_max_abs=0.0, repeatability_l2=0.0)


def capture_worker(job: dict) -> None:
    from .capture_states import capture_variant_states, load_rank_records, load_raw_logits
    from .residual_ledger import index_capture_records, extract_layer_tensors
    from .structural_accumulation import analyze_state_pair
    from .layer_protection import _generate_with_intervention
    from .one_step_intervention import InstallCausalInterventionOp
    from .router_objective import (router_proxy_metrics_from_logits, build_production_topk_teacher,
                                    topk_id_match_ratio, topk_overlap_mean, outside_teacher_topk_mass)
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.build_llm import build_real_vllm
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.worker_hooks import InstallHooksOp, remove_hooks

    torch.set_num_threads(8)
    out, cohort = Path(job['output_root']), Path(job['cohort'])
    states = read_jsonl(cohort)
    variant = job['variant']
    model_dir = Path(job['materialized_model_path']) if job.get('materialized_model_path') else None
    llm, runtime = build_real_vllm(variant, model_path=job['model_path'], phasea_root=Path(job['phasea_root']),
                                  materialized_model_path=model_dir, gpu_memory_utilization=0.90)
    capture_variant_states(variant=variant, cohort_path=cohort, output_root=out,
                           capture_level='core', llm_and_runtime=(llm, runtime))
    validate_capture(out, variant, states)
    rows = []
    llm.apply_model(InstallHooksOp(variant, str(out / 'intervention_hooks'), 'core'))
    try:
        for meta in states:
            key, di = meta['sample_key'], meta['decode_index']
            logits = load_raw_logits(out / 'raw_logits', variant, key, di)
            repeat = _generate_with_intervention(llm, meta, variant, out / 'repeat_logits', None)
            if not torch.equal(logits, repeat):
                raise RuntimeError(f'{key}: repeatability failed (exact measured envelope required)')
            if variant == 'E0':
                continue
            e0 = Path(job['reference_root'])
            rec0 = load_rank_records(e0 / 'hooks', 'E0', key, 0)
            rec1 = load_rank_records(out / 'hooks', variant, key, 0)
            l0 = load_raw_logits(e0 / 'raw_logits', 'E0', key, di)
            structural = analyze_state_pair(sample_meta=meta, e0_records=rec0, e1_records=rec1,
                                             e0_logits=l0, e1_logits=logits)
            if not structural['ledger_pass']:
                raise RuntimeError('candidate residual closure failed')
            idx0, idx1 = index_capture_records(rec0), index_capture_records(rec1)
            t0 = extract_layer_tensors(idx0, sample_key=key, decode_index=di)
            t1 = extract_layer_tensors(idx1, sample_key=key, decode_index=di)
            for layer in job['layers']:
                source = job['causal_source_by_layer'][str(layer)]
                r0 = t0['R_A'][layer] if source == 'attention' else t0['R'][layer + 1]
                r1 = t1['R_A'][layer] if source == 'attention' else t1['R'][layer + 1]
                router0 = idx0[(key, di, layer, 'router_logits', 'logits', 0)]
                router1 = idx1[(key, di, layer, 'router_logits', 'logits', 0)]
                args = {'sample_key': key, 'layer': layer, 'abs_position': meta['abs_position']}
                noop = _generate_with_intervention(llm, meta, variant, out / 'router_noop_logits',
                    InstallCausalInterventionOp('router_freeze', {**args, 'router_logits': router1}))
                if not torch.equal(noop, logits):
                    raise RuntimeError(f'{key} L{layer}: router noop identity failed')
                repaired = _generate_with_intervention(llm, meta, variant, out / 'router_repair_logits',
                    InstallCausalInterventionOp('router_freeze', {**args, 'router_logits': router0}))
                teacher_ids, _ = build_production_topk_teacher(router0.reshape(1, -1).float(), top_k=8, norm_topk_prob=True)
                row = {k: meta[k] for k in ('sample_key', 'calibration_sample_id', 'source', 'prefix_length_j')}
                row.update({'layer': layer, 'candidate': job['label'],
                            'final_logit_kl': structural['kl_exact'],
                            'cumulative_residual_nmse': _nmse(r1, r0),
                            'final_residual_nmse': _nmse(t1['R'][48], t0['R'][48]),
                            'final_hidden_nmse': _nmse(t1['final_hidden'], t0['final_hidden']),
                            'positive_G': sum(max(0., r['G_A']) + max(0., r['G_M']) for r in structural['rows']),
                            'router_causal_contribution': structural['kl_exact'] - exact_kl(l0, repaired),
                            'topk_id_match_ratio': float(topk_id_match_ratio(teacher_ids, router1.reshape(1, -1))),
                            'topk_overlap_mean': float(topk_overlap_mean(teacher_ids, router1.reshape(1, -1))),
                            'outside_teacher_topk_mass': float(outside_teacher_topk_mass(teacher_ids, router1.reshape(1, -1))),
                            **router_proxy_metrics_from_logits(router0, router1, top_k=8),
                            'identity_pass': True})
                rows.append(row)
            write_jsonl(out / 'state_metrics.jsonl', rows)
    finally:
        llm.apply_model(remove_hooks)
    atomic_write_json(out / 'validation_complete.json', {'status': 'COMPLETE', 'job': job,
                      'n_states': len(states), 'n_rows': len(rows), 'repeatability_max_abs': 0.0})


def run_capture_job(job: dict, *, log_path: Path) -> None:
    out = Path(job['output_root'])
    complete = out / 'validation_complete.json'
    if complete.exists():
        saved = read_json(complete)
        if saved['job'] != job or saved['status'] != 'COMPLETE':
            raise RuntimeError(f'capture provenance mismatch: {out}')
        if job['variant'] != 'E0':
            rows = read_jsonl(out / 'state_metrics.jsonl')
            if len(rows) != saved['n_rows'] or any(not r['identity_pass'] for r in rows):
                raise RuntimeError('incomplete metrics on resume')
        return
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out / 'job.json', job)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a') as log:
        subprocess.run([sys.executable, '-u', '-m', __name__, '--job', str(out / 'job.json')],
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def paired_gain(baseline: list[dict], candidate: list[dict], metric: str, *, higher_is_better: bool = False) -> dict:
    def samples(rows):
        grouped = {}
        for row in rows:
            sid = row['calibration_sample_id']
            grouped.setdefault(sid, []).append(row)
        values = {}
        for sid, group in grouped.items():
            if sorted(r['prefix_length_j'] for r in group) != list(PREFIX_LENGTHS_J):
                raise RuntimeError('missing/duplicate repeated state')
            values[sid] = (sum(r[metric] for r in group) / len(group), group[0]['source'])
        return values
    a, b = samples(baseline), samples(candidate)
    if set(a) != set(b) or len(a) < 2:
        raise RuntimeError('paired comparison requires matching samples and at least two independent units')
    direction = -1 if higher_is_better else 1
    gains = [direction * (a[s][0] - b[s][0]) for s in sorted(a)]
    result = sample_bootstrap_ci(gains, seed=20260909, n_boot=10000)
    result['source_mean_gain'] = {source: sum(direction * (a[s][0] - b[s][0]) for s in a if a[s][1] == source) /
                                sum(a[s][1] == source for s in a) for source in ('wikitext2', 's1k_original')}
    result['stable'] = result['ci95_lo'] > 0 and all(v > 0 for v in result['source_mean_gain'].values())
    return result


def review_results(run_root: Path) -> dict:
    root = run_root / '60_objective/actual_holdout'
    scope = read_json(run_root / '60_objective/objective_scope.json')
    base = read_jsonl(root / 'E1/state_metrics.jsonl')
    comparisons, eligible, summaries, router_comparisons = {}, [], {}, {}
    def summarize(rows):
        metrics = ('final_logit_kl', 'cumulative_residual_nmse', 'final_residual_nmse',
                   'final_hidden_nmse', 'positive_G', 'router_causal_contribution',
                   'router_full_kl', 'router_topk_total', 'topk_id_match_ratio',
                   'topk_overlap_mean', 'outside_teacher_topk_mass')
        result = {}
        for metric in metrics:
            by_sample = {}
            sources = {}
            for row in rows:
                sid = row['calibration_sample_id']
                by_sample.setdefault(sid, []).append(row[metric])
                sources[sid] = row['source']
            if len(by_sample) != 16 or any(len(v) != 4 for v in by_sample.values()):
                raise RuntimeError('incomplete holdout metrics')
            means = {s: sum(v) / 4 for s,v in by_sample.items()}
            stat = sample_bootstrap_ci(list(means.values()), seed=20260909, n_boot=10000)
            stat['source_mean'] = {src: sum(means[s] for s in means if sources[s] == src) / 8
                                   for src in ('wikitext2', 's1k_original')}
            result[metric] = stat
        return result
    for layer in scope['selected_objective_layers']:
        directory = run_root / '60_objective/objective_candidates' / f'L{layer}'
        by_loss = {p.parent.name: read_jsonl(root / f'L{layer}_{p.parent.name}/state_metrics.jsonl')
                   for p in directory.glob('*/checkpoint.pt')}
        o1 = 'O1_A' if scope['causal_source_by_layer'][str(layer)] == 'attention' else 'O1_M'
        o2 = o1.replace('O1', 'O2')
        refs = {'E1': [r for r in base if r['layer'] == layer], 'O0': by_loss['O0'], o1: by_loss[o1]}
        for loss, rows in {'E1': refs['E1'], **by_loss}.items():
            summaries[f'L{layer}:{loss}'] = summarize(rows)
        for loss in [o2] + [x for x in ('O3_full', 'O3_topk') if x in by_loss]:
            refs_for_loss = dict(refs)
            if loss.startswith('O3'):
                refs_for_loss[o2] = by_loss[o2]
            entry = {ref: {metric: paired_gain(rows, by_loss[loss], metric)
                          for metric in ('final_logit_kl', 'cumulative_residual_nmse')}
                     for ref, rows in refs_for_loss.items()}
            comparisons[f'L{layer}:{loss}'] = entry
        if all(v['stable'] for metrics in comparisons[f'L{layer}:{o2}'].values() for v in metrics.values()):
            eligible.append(layer)
        if 'O3_topk' in by_loss:
            router_comparisons[str(layer)] = {metric: paired_gain(by_loss['O3_full'], by_loss['O3_topk'], metric,
                higher_is_better=metric in ('topk_id_match_ratio', 'topk_overlap_mean'))
                for metric in ('final_logit_kl', 'outside_teacher_topk_mass', 'topk_id_match_ratio', 'topk_overlap_mean')}
    result = {'status': 'WAITING_REVIEW', 'numerical_recommendation': 'TOPK_SUPPORTED' if len(eligible) >= 4 else 'STOP_EXPANSION',
              'eligible_o2_layers': eligible, 'comparisons': comparisons,
              'o3_topk_vs_full': router_comparisons,
              'evidence': str(root), 'reason': 'S5 requires explicit approval and frozen ranked layers/objectives/K; this analysis grants none'}
    atomic_write_json(run_root / '60_objective/objective_expansion_gate.json', result)
    atomic_write_json(root / 'all_objective_metrics_summary.json', summaries)
    lines = ['# S4 actual-path objective expansion review', '',
             f"Recommendation: {result['numerical_recommendation']}", f'Eligible O2 layers: {eligible}', '',
             '16 independent samples, 4 fixed-history states each; paired bootstrap and both-source direction checks.', '',
             '| Candidate | Comparator | KL gain [95% CI] | Cumulative NMSE gain [95% CI] | Stable |',
             '|---|---|---|---|---|']
    for label, entries in comparisons.items():
        for ref, metrics in entries.items():
            kl, nmse = metrics['final_logit_kl'], metrics['cumulative_residual_nmse']
            lines.append(f"| {label} | {ref} | {kl['mean']:.6g} [{kl['ci95_lo']:.6g}, {kl['ci95_hi']:.6g}] | {nmse['mean']:.6g} [{nmse['ci95_lo']:.6g}, {nmse['ci95_hi']:.6g}] | {kl['stable'] and nmse['stable']} |")
    (root / 'GATE_REVIEW.md').write_text('\n'.join(lines) + '\n')
    report = run_root / '60_objective/OBJECTIVE_TRAINING_ABLATION_REPORT.md'
    report.write_text('\n'.join(lines) + '\n\nAll objective metrics, per-source statistics and O4 controls: '
                      '`actual_holdout/all_objective_metrics_summary.json`.\n'
                      'State-level actual-path records: `actual_holdout/<candidate>/state_metrics.jsonl`.\n')
    return result


def run_objective_holdout(*, run_root: Path, model_path: str, phasea_root: Path) -> dict:
    torch.set_num_threads(8)
    repair_legacy_validation(run_root)
    cohort = freeze_holdout(run_root)
    scope = read_json(run_root / '60_objective/objective_scope.json')
    root = cohort.parent
    recipe_root = run_root / '60_objective/objective_candidates'
    declared = read_json(recipe_root / 'training_recipes.json')['recipes']
    checkpoints = sorted(recipe_root.glob('L*/O*/checkpoint.pt'))
    expected_tags = {(int(r['layer']), r['loss']) for r in declared}
    actual_tags = {(int(p.parent.parent.name[1:]), p.parent.name) for p in checkpoints}
    if len(expected_tags) != len(declared) or actual_tags != expected_tags:
        raise RuntimeError('actual checkpoint set differs from frozen recipes')
    if {l for l, _ in actual_tags} != set(scope['selected_objective_layers']):
        raise RuntimeError('checkpoint layers differ from frozen scope')
    for layer, loss in actual_tags:
        if loss.startswith('O3_') and layer not in scope['router_objective_layers']:
            raise RuntimeError('O3 checkpoint outside frozen recipe scope')
    frozen_inputs = {str(p): sha256(p) for p in checkpoints}
    inputs_path = root / 'candidate_inputs.json'
    if inputs_path.exists() and read_json(inputs_path) != frozen_inputs:
        raise RuntimeError('S4 checkpoints changed during holdout validation')
    atomic_write_json(inputs_path, frozen_inputs)
    common = {'cohort': str(cohort), 'cohort_sha256': sha256(cohort), 'model_path': model_path,
              'phasea_root': str(phasea_root), 'reference_root': str(root / 'E0'),
              'causal_source_by_layer': scope['causal_source_by_layer']}
    for variant in ('E0', 'E1'):
        job = {**common, 'variant': variant, 'label': variant, 'output_root': str(root / variant),
               'layers': [] if variant == 'E0' else scope['selected_objective_layers']}
        print(f'[holdout] {variant}', flush=True)
        run_capture_job(job, log_path=run_root / f'logs/holdout_{variant}.log')
    for checkpoint in checkpoints:
        layer = int(checkpoint.parent.parent.name[1:])
        loss = checkpoint.parent.name
        label = f'L{layer}_{loss}'
        print(f'[holdout] {label}', flush=True)
        model_dir = materialize_candidate(checkpoints={layer: checkpoint}, output_dir=root / 'models' / label,
                                          model_path=model_path, phasea_root=phasea_root)
        job = {**common, 'variant': 'E1', 'label': label, 'output_root': str(root / label), 'layers': [layer],
               'materialized_model_path': str(model_dir), 'checkpoint_sha256': sha256(checkpoint)}
        run_capture_job(job, log_path=run_root / f'logs/holdout_{label}.log')
    return review_results(run_root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    capture_worker(read_json(Path(args.job)))
