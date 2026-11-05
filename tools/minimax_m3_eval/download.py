"""Resume pinned public HF files with curl's bounded transport retries."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess


def save(path, value):
    temporary = path.with_suffix('.writing')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def download(root):
    root = Path(root)
    manifest = json.loads((root / 'download_manifest.json').read_text())
    snapshot = Path(manifest['snapshot'])
    blobs = snapshot.parents[1] / 'blobs'
    snapshot.mkdir(parents=True, exist_ok=True)
    blobs.mkdir(exist_ok=True)

    def fetch(entry):
        dest = snapshot / entry['name']
        if dest.is_file() and dest.stat().st_size == entry['size']:
            return
        digest = entry['sha256'] or entry['git_blob_id']
        blob = blobs / digest
        partial = blobs / (digest + '.incomplete')
        if not blob.is_file():
            url = f"https://huggingface.co/{manifest['repo_id']}/resolve/{manifest['revision']}/{entry['name']}"
            subprocess.run(['curl', '--fail', '--location', '--continue-at', '-',
                '--retry', '5', '--retry-all-errors', '--retry-delay', '5',
                '--connect-timeout', '30', '--speed-limit', '1024', '--speed-time', '180',
                '--output', str(partial), url], check=True)
            if partial.stat().st_size != entry['size']:
                raise ValueError(f"Incorrect download size: {entry['name']}")
            with partial.open('rb') as handle:
                if entry['sha256']:
                    actual = hashlib.file_digest(handle, 'sha256').hexdigest()
                else:
                    hasher = hashlib.sha1(f"blob {entry['size']}\0".encode())
                    for chunk in iter(lambda: handle.read(8 << 20), b''):
                        hasher.update(chunk)
                    actual = hasher.hexdigest()
            if actual != digest:
                raise ValueError(f"Downloaded digest mismatch: {entry['name']}")
            partial.replace(blob)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.is_symlink():
            dest.symlink_to(os.path.relpath(blob, dest.parent))
        if dest.stat().st_size != entry['size']:
            raise ValueError(f"Cached file size mismatch: {entry['name']}")
        print('complete', entry['name'], entry['size'], flush=True)

    state = dict(status='running', pid=os.getpid(), revision=manifest['revision'])
    save(root / 'download_status.json', state)
    try:
        # Fetch tokenizer/config files before the large weight transfers.
        small = [f for f in manifest['files'] if not f['name'].endswith('.safetensors')]
        weights = [f for f in manifest['files'] if f['name'].endswith('.safetensors')]
        for entry in small:
            fetch(entry)
        save(root / 'metadata_ready.json', {'complete': True, 'snapshot': str(snapshot)})
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(fetch, weights))
        missing = [f['name'] for f in manifest['files'] if not (snapshot / f['name']).is_file()
                   or (snapshot / f['name']).stat().st_size != f['size']]
        if missing:
            raise FileNotFoundError(missing)
        state.update(status='complete', exit_code=0, files=len(manifest['files']))
    except Exception as error:
        state.update(status='failed', exit_code=1, error=repr(error))
        raise
    finally:
        state['updated_at'] = datetime.now(timezone.utc).isoformat()
        save(root / 'download_status.json', state)
