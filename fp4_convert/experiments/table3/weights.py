"""Reuse matching complete exports and verify actual indexed tensors on reload."""
from pathlib import Path

from .jobs import call, now, write
from .plan import digest, read


def inspect(config):
    from safetensors import safe_open
    from fp4_convert.experiments.paper_baselines.checkpoint import Checkpoint

    path = Path(config['checkpoint'])
    manifest = read(path/'conversion.json')
    source = Checkpoint(config['source'])
    tag = read(path/'config.json')['quantization_config']
    if (any(manifest[k] != config[k] for k in ('source','target','method'))
            or manifest['source_id'] != source.source_id() or set(manifest['matrices']) != source.native
            or set(tag['nvfp4']) != source.native or tag['method'] != config['method']
            or tag['target'] != config['target'] or tag['quant_method'] != 'paper_fp4'
            or not manifest['activation_rule'] or not manifest['format_rule']):
        raise ValueError(f'Export identity or coverage differs: {path}')
    index = read(path/'model.safetensors.index.json')['weight_map']
    grouped = config['method'] in ('g256','g512')
    outer = {key.removesuffix('.weight_scale_fp32') for key in index if key.endswith('.weight_scale_fp32')}
    if outer != (source.native if grouped else set()):
        raise ValueError(f'Incorrect outer FP32 tensor coverage: {path}')
    shards = {}
    for shard in sorted(set(index.values())):
        file = path/shard
        with safe_open(file, framework='pt', device='cpu') as reader:
            if set(reader.keys()) != {k for k,s in index.items() if s==shard}:
                raise ValueError(f'Index/header mismatch: {shard}')
            for key in reader.keys():
                name, _, suffix = key.rpartition('.')
                if name not in source.native or suffix not in ('weight','weight_scale_fp32'):
                    continue
                shape = manifest['matrices'][name]['shape']
                dtype = 'BF16'
                if suffix == 'weight_scale_fp32':
                    g = int(config['method'][1:])
                    dtype, shape = 'F32', [shape[0], (shape[1]+g-1)//g]
                value = reader.get_slice(key)
                if value.get_dtype() != dtype or value.get_shape() != shape:
                    raise ValueError(f'Invalid reload tensor: {key}')
        stat = file.stat()
        shards[shard] = dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)
    if not all(name+'.weight' in index for name in source.native):
        raise ValueError(f'Missing weight in export {path}')
    return dict(config=config['id'], checkpoint=str(path), conversion_sha256=digest(path/'conversion.json'),
                config_sha256=digest(path/'config.json'), index_sha256=digest(path/'model.safetensors.index.json'),
                source_id=manifest['source_id'], projections=len(source.native), shards=shards)


def ensure(root, config, gpu, locks):
    key = config['id']
    dest = Path(config['checkpoint'])
    receipt = root/'exports'/f'{key}.json'
    previous = read(receipt) if receipt.exists() else {}
    if not dest.exists():
        candidate = read(root/'inventory.json')['reusable_export_candidates'].get(key)
        if candidate:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.symlink_to(Path(candidate).resolve(), target_is_directory=True)
        else:
            call(root, config, 'export', lambda _: [
                'fp4_convert.experiments.paper_baselines.export', config['source'], dest,
                '--target', config['target'], '--method', config['method'], '--device','cuda'],
                 gpu, locks, {'checkpoint':str(dest)})
    result = inspect(config)
    if previous:
        if any(previous.get(k) != v for k,v in result.items()):
            raise ValueError(f'Accepted checkpoint changed: {receipt}')
    else:
        result.update(verified=now(), plan_sha256=digest(root/'plan.json'))
        write(receipt,result)
    return digest(receipt)
