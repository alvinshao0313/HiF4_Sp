"""Explicitly certify unchanged capture producers after a training-only fix."""
from pathlib import Path
from .artifacts import capture_fingerprint, read_json, sha256, source_fingerprint, write_json


def certify_captures(root, preserved):
    root, preserved = Path(root).resolve(), Path(preserved).resolve()
    original = read_json(preserved.parent / 'source_before.json')
    prior = read_json(root / 'capture_reuse.json') if (root / 'capture_reuse.json').exists() else None
    # The first certificate was issued before the differentiable runtime was
    # repaired.  Its full source digest therefore must not be compared with
    # the repaired training files.  Reuse is still fail-closed on the actual
    # capture producer AST and immutable external dependency fingerprints.
    if prior is not None and prior.get('status') == 'CERTIFIED':
        if prior.get('original_source') != original['source']:
            raise RuntimeError('capture reuse certificate has inconsistent source provenance')
    elif source_fingerprint(preserved) != original['source']:
        raise RuntimeError('external code or packages changed since source preservation')
    producer = capture_fingerprint(preserved)
    if producer != capture_fingerprint():
        raise RuntimeError('capture producer changed; existing captures cannot be reused')
    manifests = {}
    for variant, digest in original['captures'].items():
        path = root / 'captures' / variant / 'manifest.json'
        value = read_json(path)
        if sha256(path) != digest or value['source_sha256'] != original['source'] or value['status'] != 'COMPLETE':
            raise RuntimeError(f'preserved capture changed: {variant}')
        manifests[str(path)] = digest
    proof = dict(status='CERTIFIED', producer_sha256=producer, manifests=manifests,
                 original_source=original['source'], preserved_source=str(preserved),
                 repaired_source=source_fingerprint(),
                 reason='Capture functions, their helpers, external code and package versions are unchanged.')
    write_json(proof, root / 'capture_reuse.json')
    return proof
