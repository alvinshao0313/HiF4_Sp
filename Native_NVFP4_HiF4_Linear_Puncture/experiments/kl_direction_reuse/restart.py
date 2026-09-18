"""New run directories reference certified immutable inputs, never old trained states."""
import json
from pathlib import Path
import shutil

from .artifacts import capture_fingerprint, read_json, sha256, write_json
from .capture import CaptureStore
from .data import Dataset
from .materialize import check_baseline, model_identity
from .train import require_verification, training_provenance


def reuse_inputs(source, destination, *, verification=False):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"new output directory required: {destination}")
    ds = Dataset(source)
    base = check_baseline(source)
    if base['source_files'] != model_identity(ds.snapshot):
        raise RuntimeError('preserved native model changed')
    proof = read_json(source / 'capture_reuse.json')
    if proof['status'] != 'CERTIFIED' or proof['producer_sha256'] != capture_fingerprint():
        raise RuntimeError('capture producer changed; cannot reuse preserved inputs')
    bound = {str((source / 'protocol.json').resolve()): sha256(source / 'protocol.json'),
             str((source / 'baseline/export.json').resolve()): sha256(source / 'baseline/export.json')}
    for variant, identity in [('native', base['source_files']), ('baseline', base['files'])]:
        store = CaptureStore(source / 'captures' / variant, ds)
        if set(store.manifest['samples']) != set(ds.samples) or store.manifest['model_files'] != identity:
            raise RuntimeError(f'incomplete or mismatched {variant} capture')
        path = (source / 'captures' / variant / 'manifest.json').resolve()
        bound[str(path)] = sha256(path)
    rows = [json.loads(line) for line in (source / 'stages.jsonl').read_text().splitlines()]
    selected = []
    for command, args in [('prepare', []), ('baseline', []),
                          ('capture', ['--variant', 'native']), ('capture', ['--variant', 'baseline'])]:
        matches = [r for r in rows if r['command'] == command and r['args'] == args and
                   r['status'] == 'COMPLETE' and r['returncode'] == 0]
        if len(matches) != 1:
            raise RuntimeError(f'expected one successful input production record: {command} {args}')
        row = matches[0]
        selected.append({**row, 'reused': True, 'origin_root': str(source), 'wall_seconds': 0.,
                         'original_wall_seconds': row.get('original_wall_seconds', row['wall_seconds'])})
    if verification:
        provenance = training_provenance(source)
        for layer in (8, 24, 40):
            require_verification(source, layer, provenance)
            for name in ('prepare.json', 'report.json', 'actual_probe/manifest.json'):
                path = (source / 'verification' / f'L{layer:02d}' / name).resolve()
                bound[str(path)] = sha256(path)
    destination.mkdir(parents=True)
    shutil.copy2(source / 'protocol.json', destination / 'protocol.json')
    shutil.copy2(source / 'capture_reuse.json', destination / 'capture_reuse.json')
    for name in ('data', 'baseline', 'captures'):
        (destination / name).symlink_to((source / name).resolve(), target_is_directory=True)
    if verification:
        (destination / 'verification').symlink_to((source / 'verification').resolve(), target_is_directory=True)
    write_json(dict(status='CERTIFIED', source_root=str(source), immutable_inputs=bound,
                    verification_root=str(source) if verification else None,
                    producer_sha256=proof['producer_sha256']), destination / 'input_reuse.json')
    (destination / 'stages.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in selected))
    return destination
