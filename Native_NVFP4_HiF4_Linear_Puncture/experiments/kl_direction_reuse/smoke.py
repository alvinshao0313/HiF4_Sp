"""Evidence for the agreed L24 smoke run; never a scientific performance report."""
import json
import math
from pathlib import Path

import torch

from .artifacts import checked_load, parameters_hash, read_json, sha256, write_json
from .capture import CaptureStore
from .data import Dataset
from .directions import Directions
from .train import _TrainingView, require_verification, training_provenance

SMOKE_RECIPES = ('L24_mse', 'L24_cached_kl_k1', 'L24_direct_kl')


def controller_hashes():
    scripts = Path(__file__).parent / 'scripts'
    return {name: sha256(scripts / name) for name in
            ('long_pipeline.py', 'smoke_pipeline.py', 'launch_long_pipeline.sh')}


def settings(ds, train_gpus, eval_gpus):
    # Preserve membership of the actual maximum-length training batch.
    batch = max(ds.protocol['batches'], key=lambda b: max(len(ds.samples[s]['input_ids']) for s in b))
    ordered = sorted(ds.ids('test'), key=lambda s: (len(ds.samples[s]['input_ids']), s))
    return dict(budget_seconds=600., train_sample_ids=batch,
                eval_sample_ids=[ordered[0], ordered[-1]], recipes=list(SMOKE_RECIPES),
                train_gpus=list(train_gpus), eval_gpus=list(eval_gpus),
                checkpoint_thresholds=[150, 300, 450])


def resource_check(rows):
    if len(rows) < 2:
        raise RuntimeError('smoke needs repeated complete updates for resource observations')
    # Raw RSS includes libc pages retained for reuse.  The gate therefore uses
    # live libc heap bytes, while recording RSS as a diagnostic observation.
    if not all('heap_live' in row for row in rows):
        raise RuntimeError('smoke resource observations lack live heap accounting')
    bounds = dict(cuda_allocated=64 << 20, heap_live=64 << 20)
    result = {}
    for key, allowance in bounds.items():
        values = [r[key] for r in rows]
        if any(v < 0 or not math.isfinite(v) for v in values):
            raise RuntimeError(f'invalid resource measurement: {key}')
        growth = max(values[1:]) - values[0]
        result[key] = dict(first=values[0], last=values[-1], peak=max(values),
                           growth_bytes=growth, allowance_bytes=allowance)
        if growth > allowance:
            raise RuntimeError(f'unresolved resource growth in smoke: {key} {growth} bytes')
    rss = [r['rss_bytes'] for r in rows]
    if any(v < 0 or not math.isfinite(v) for v in rss):
        raise RuntimeError('invalid resource measurement: rss_bytes')
    result['rss_bytes'] = dict(first=rss[0], last=rss[-1], peak=max(rss),
                               growth_bytes=max(rss[1:]) - rss[0],
                               interpretation='allocator high-water diagnostic; not a leak gate')
    return result


def summarize_smoke(root):
    ds = Dataset(root)
    config = read_json(ds.root / 'smoke_config.json')
    provenance = training_provenance(ds.root)
    for layer in (8, 24, 40):
        require_verification(ds.root, layer, provenance)
    evidence, cases = {}, []
    def bind(path):
        path = Path(path).resolve()
        evidence[str(path)] = sha256(path)
    bind(ds.root / 'smoke_config.json')
    for layer in (8, 24, 40):
        for name in ('prepare.json', 'report.json', 'actual_probe/manifest.json'):
            bind(ds.root / 'verification' / f'L{layer:02d}' / name)
    for recipe in SMOKE_RECIPES:
        run = ds.root / 'runs' / recipe
        m = read_json(run / 'manifest.json')
        if (m['status'] != 'COMPLETE' or m['steps'] < 2 or m['provenance'] != provenance or
                m['budget_seconds'] != 600. or m['train_sample_ids'] != config['train_sample_ids']):
            raise RuntimeError(f'incomplete or misconfigured smoke training: {recipe}')
        cp = checked_load(m['final_checkpoint'])
        resume = checked_load(read_json(run / 'state.json')['checkpoint'])
        digest = parameters_hash(cp['parameters'])
        if (digest != m['parameters_sha256'] or digest != parameters_hash(resume['parameters']) or
                digest == m['initial_parameters_sha256']):
            raise RuntimeError('checkpoint roundtrip or nonzero-update check failed')
        for threshold in config['checkpoint_thresholds']:
            path = run / 'checkpoints' / f'time_{threshold}.pt'
            from .artifacts import load
            saved = load(path)
            if saved['evaluation_seconds'] != threshold or saved['elapsed'] > threshold:
                raise RuntimeError('time checkpoint captured a future update')
            if any(not torch.isfinite(t).all() for t in saved['parameters'].values()):
                raise RuntimeError('nonfinite saved parameters')
            bind(path)
        rows = [json.loads(line) for line in (run / 'training.jsonl').read_text().splitlines()]
        if len(rows) != m['steps'] or any(not all(math.isfinite(r[k]) for k in
                ('main', 'router', 'gradient_norm')) for r in rows):
            raise RuntimeError('missing updates or nonfinite training measurements')
        resources = resource_check(rows)
        refreshes = []
        if recipe == 'L24_cached_kl_k1':
            view = _TrainingView(ds, config['train_sample_ids'])
            for manifest in sorted((run / 'directions').glob('*/manifest.json')):
                if read_json(manifest)['status'] != 'COMPLETE':
                    continue  # A budget-expired refresh is not used by committed updates.
                cache = Directions(manifest.parent, provenance=provenance, ds=view, layer=24)
                denominator = sum(len(ds.samples[s]['input_ids']) for s in config['train_sample_ids'])
                for sid in config['train_sample_ids']:
                    cache.sample(sid, denominator)
                bind(manifest)
                refreshes.append(manifest.parent.name)
            if len(refreshes) < 2:
                raise RuntimeError('k1 smoke did not exercise the next-epoch direction refresh')
        export = read_json(run / 'exports/final/export.json')
        capture = CaptureStore(run / 'evaluation/final', ds)
        shard = 'model-layer-00024-of-00048.safetensors'
        baseline = read_json(ds.root / 'baseline/export.json')
        if (export['parameters_sha256'] != digest or export['checkpoint_sha256'] != sha256(run / 'final.pt') or
                export['files'][shard] == baseline['files'][shard] or
                capture.manifest['model_files'] != export['files'] or
                set(capture.manifest['samples']) != set(config['eval_sample_ids'])):
            raise RuntimeError('TP2 smoke evaluated a mismatched or unchanged candidate')
        for sid, row in capture.manifest['samples'].items():
            n, metrics = len(ds.samples[sid]['input_ids']), row['metrics']
            if (metrics['kl_tokens'] != n or metrics['nll_tokens'] != n-1 or
                    not all(math.isfinite(v) for v in metrics.values())):
                raise RuntimeError('incomplete/nonfinite TP2 prediction metrics')
        for name in ('manifest.json', 'final.pt', 'resume.pt', 'state.json', 'training.jsonl',
                     'exports/final/export.json', 'evaluation/final/manifest.json'):
            bind(run / name)
        cases.append(dict(recipe=recipe, steps=m['steps'], resources=resources,
                          complete_refreshes=refreshes, parameters_sha256=digest))
    result = dict(status='PASS', root=str(ds.root), provenance=provenance,
                  controllers=controller_hashes(), config=config, cases=cases, evidence=evidence,
                  limitation='L24 repeated updates and longest complete train batch; no convergence claim.')
    write_json(result, ds.root / 'smoke_report.json')
    return result


def require_smoke(path, root, train_gpus, eval_gpus):
    report = read_json(path)
    if (report['status'] != 'PASS' or report['provenance'] != training_provenance(root) or
            report['controllers'] != controller_hashes() or
            report['config']['train_gpus'] != list(train_gpus) or
            report['config']['eval_gpus'] != list(eval_gpus) or
            [c['recipe'] for c in report['cases']] != list(SMOKE_RECIPES)):
        raise RuntimeError('current code and allocation have no matching complete smoke PASS')
    for artifact, digest in report['evidence'].items():
        if sha256(artifact) != digest:
            raise RuntimeError(f'smoke evidence changed: {artifact}')
    if (Path(root) / 'smoke_config.json').exists() or Path(root).resolve() == Path(report['root']).resolve():
        raise RuntimeError('formal training requires a separate root without smoke settings')
    return report
