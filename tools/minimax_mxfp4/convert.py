"""Stream checkpoint shards through NVFP4 -> BF16 -> packed MXFP4."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time

from safetensors import safe_open
from safetensors.torch import save_file
import torch

from tools.minimax_m3_eval.download import save
from .format import SCALE_RULE, dequantize, nvfp4_to_bf16, quantize

EXPERT = re.compile(r'(.+\.block_sparse_moe\.experts\.\d+\.w[123])\.weight_packed$')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def tensor_digest(tensor):
    return hashlib.sha256(tensor.contiguous().reshape(-1).view(torch.uint8).numpy()).hexdigest()


def convert(source, output, threads=8):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Conversion output must be new: {output}')
    torch.set_num_threads(threads)
    config = json.loads((source / 'config.json').read_text())
    quant = config['quantization_config']
    if (quant.get('moe_quant_algo') != 'NVFP4' or quant.get('group_size') != 16
            or quant.get('moe_quant_format') != 'nvfp4-pack-quantized'):
        raise ValueError('Expected the original MiniMax packed NVFP4 checkpoint')
    index = json.loads((source / 'model.safetensors.index.json').read_text())
    weight_map = index['weight_map']
    selected = sorted(name for name in weight_map if EXPERT.fullmatch(name))
    text = config['text_config']
    expected = sum(text['moe_layer_freq']) * text['num_local_experts'] * 3
    if len(selected) != expected:
        raise ValueError(f'Expected {expected} expert matrices, found {len(selected)}')
    for name in selected:
        stem = name.removesuffix('.weight_packed')
        for suffix in ('weight_scale', 'weight_global_scale', 'input_global_scale'):
            if stem + '.' + suffix not in weight_map or weight_map[stem + '.' + suffix] != weight_map[name]:
                raise ValueError(f'NVFP4 components must share a shard: {name}')
    output.mkdir(parents=True)
    state = {'status': 'running', 'pid': os.getpid(), 'source': str(source), 'output': str(output),
             'threads': threads, 'started_at': datetime.now(timezone.utc).isoformat(),
             'scale_rule': SCALE_RULE, 'weight_bits': 4, 'scale_bits': 8, 'group_size': 32,
             'activation_format': 'MXFP4', 'expected_matrices': expected, 'completed_matrices': 0,
             'shards': [], 'unchanged_tensors': 0, 'squared_error': 0., 'squared_reference': 0.}
    save(output / 'conversion_status.json', state)
    new_map = {}; payload_bytes = 0
    try:
        # Metadata and the unused DSpark weights can be shared safely: this
        # converter never writes to a linked source file.
        shards = sorted(set(weight_map.values()))
        for path in source.rglob('*'):
            if not path.is_file():
                continue
            relative = path.relative_to(source)
            if str(relative) in shards or str(relative) in ('config.json', 'model.safetensors.index.json'):
                continue
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(path.resolve(), destination)
        quant.update(moe_quant_algo='MXFP4', moe_quant_format='mxfp4-pack-quantized',
                     group_size=32, moe_activation_quant_algo='MXFP4', mxfp4_scale_rule=SCALE_RULE)
        save(output / 'config.json', config)
        for shard in shards:
            started = time.monotonic()
            destination = output / shard
            tensors = {}; changes = []; unchanged = {}
            with safe_open(source / shard, framework='pt', device='cpu') as handle:
                names = handle.keys()
                for name in names:
                    if EXPERT.fullmatch(name):
                        stem = name.removesuffix('.weight_packed')
                        bf16 = nvfp4_to_bf16(handle.get_tensor(name),
                            handle.get_tensor(stem + '.weight_scale'),
                            handle.get_tensor(stem + '.weight_global_scale'))
                        packed, scales = quantize(bf16)
                        reconstructed = dequantize(packed, scales)
                        error = (reconstructed.float() - bf16.float()).square().sum().item()
                        reference = bf16.float().square().sum().item()
                        changes.append({'name': stem, 'shape': list(bf16.shape),
                            'squared_error': error, 'squared_reference': reference})
                        tensors[name] = packed
                        tensors[stem + '.weight_scale'] = scales
                        del bf16, reconstructed
                    elif '.block_sparse_moe.experts.' in name:
                        continue
                    else:
                        tensors[name] = handle.get_tensor(name)
                        unchanged[name] = tensor_digest(tensors[name])
                temporary = destination.with_suffix('.writing')
                save_file(tensors, temporary, metadata={'format': 'pt'})
                temporary.replace(destination)
            # Reload actual serialized data, preserving the packed-byte contract
            # and the exact unmodified non-expert tensors.
            with safe_open(destination, framework='pt', device='cpu') as written:
                for name, value in tensors.items():
                    saved = written.get_tensor(name)
                    if saved.dtype != value.dtype or saved.shape != value.shape or tensor_digest(saved) != tensor_digest(value):
                        raise ValueError(f'Serialized tensor differs: {name}')
                    payload_bytes += saved.numel() * saved.element_size()
                    new_map[name] = shard
            state['completed_matrices'] += len(changes)
            state['unchanged_tensors'] += len(unchanged)
            state['squared_error'] += sum(x['squared_error'] for x in changes)
            state['squared_reference'] += sum(x['squared_reference'] for x in changes)
            record = {'name': shard, 'bytes': destination.stat().st_size, 'sha256': digest(destination),
                      'seconds': time.monotonic() - started, 'matrices': changes,
                      'unchanged_tensor_sha256': unchanged}
            state['shards'].append({k:v for k,v in record.items() if k not in ('matrices','unchanged_tensor_sha256')})
            save(output / (shard + '.audit.json'), record)
            save(output / 'conversion_status.json', state)
            print(json.dumps({'shard': shard, 'matrices_done': state['completed_matrices'],
                'expected_matrices': expected, 'seconds': record['seconds']}), flush=True)
            del tensors
        if state['completed_matrices'] != expected:
            raise ValueError('Conversion coverage is incomplete')
        save(output / 'model.safetensors.index.json', {'metadata': {'total_size': payload_bytes}, 'weight_map': new_map})
        files = []
        for path in sorted(output.rglob('*')):
            if not path.is_file() or path.name.endswith('.audit.json') or path.name == 'conversion_status.json':
                continue
            files.append({'name': str(path.relative_to(output)), 'size': path.stat().st_size, 'sha256': digest(path)})
        save(output / 'conversion_manifest.json', {'snapshot': str(output), 'repo_id': 'local/MiniMax-M3.1-Flash-MXFP4',
            'revision': digest(output / 'config.json'), 'files': files})
        state.update(status='complete', exit_code=0, payload_bytes=payload_bytes,
            nmse=state['squared_error'] / state['squared_reference'],
            manifest_sha256=digest(output / 'conversion_manifest.json'))
    except Exception as error:
        state.update(status='failed', exit_code=1, error=repr(error))
        raise
    finally:
        state['updated_at'] = datetime.now(timezone.utc).isoformat()
        save(output / 'conversion_status.json', state)
