"""Actual TP2 teacher-forced prefill capture, streamed by sample and layer."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import torch

from .candidate_runtime import sha256
from .config import DEFAULT_MODEL_PATH, DEFAULT_PHASEA_ROOT
from .mechanism_phase import phase_root, VERSION
from .router_teacher_cache import load_objective_calibration_samples
from .run_state import read_json, atomic_write_json
from ..long_trajectory_stability.real_vllm_hooks.build_llm import build_real_vllm

LAYERS = (19, 31, 35, 39, 41, 43, 45, 47)
ATTR = '_iea_actual_teacher_capture'


def tensor_hash(x):
    return hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def save_tensor(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f'teacher artifact already exists: {path}')
    temporary = path.with_suffix(path.suffix+'.partial')
    torch.save(payload, temporary)
    temporary.replace(path)
    return {'path': str(path), 'sha256': sha256(path)}


class InstallCapture:
    def __init__(self, root, variant):
        self.root, self.variant = str(root), variant

    def __call__(self, model):
        from vllm.distributed import get_tensor_model_parallel_rank
        rank = get_tensor_model_parallel_rank()
        if hasattr(model, ATTR):
            raise RuntimeError('teacher capture already installed')
        state = {'rank': rank, 'sample_id': None, 'positions': None, 'fields': {}, 'handles': []}
        setattr(model, ATTR, state)
        compute_logits = model.compute_logits
        def observe_logits(*args, **kwargs):
            logits = compute_logits(*args, **kwargs)
            if state['sample_id'] is not None and logits is not None:
                state['generated_logits'] = logits[-1:].detach().cpu().clone()
            return logits
        model.compute_logits = observe_logits
        def capture(name, tensor):
            if state['sample_id'] is None:
                return
            pos = state['positions']
            if pos is None or tensor.shape[0] != len(pos):
                raise RuntimeError(f'teacher token/row mismatch at {name}')
            state['fields'].setdefault(name, []).append((pos.clone(), tensor.detach().cpu().clone()))
        def model_pre(module, args, kwargs):
            if state['sample_id'] is None:
                return
            positions = kwargs.get('positions', args[1] if len(args)>1 else None)
            if not isinstance(positions, torch.Tensor):
                raise RuntimeError('missing actual positions')
            state['positions'] = positions.detach().cpu().reshape(-1)
        state['handles'].append(model.register_forward_pre_hook(model_pre, with_kwargs=True))
        for layer in LAYERS:
            block = model.model.layers[layer]
            def input_hook(module, args, output, l=layer):
                capture(f'{l}/input', output[1])
            def post_hook(module, args, output, l=layer):
                capture(f'{l}/moe_input', output[0])
                capture(f'{l}/post_attn', output[1])
            def attn_hook(module, args, output, l=layer):
                capture(f'{l}/attn', output[0])
            def moe_hook(module, args, output, l=layer):
                capture(f'{l}/moe', output)
            def output_hook(module, args, output, l=layer):
                capture(f'{l}/output', output[1])
            state['handles'] += [block.input_layernorm.register_forward_hook(input_hook),
                block.post_attention_layernorm.register_forward_hook(post_hook),
                block.self_attn.o_proj.register_forward_hook(attn_hook),
                block.mlp.register_forward_hook(moe_hook)]
            successor = model.model.layers[layer+1].input_layernorm if layer<47 else model.model.norm
            state['handles'].append(successor.register_forward_hook(output_hook))
            if layer in (19, 31):
                def router_hook(module, args, output, l=layer):
                    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
                    logits = output[0] if isinstance(output, tuple) else output
                    weights, ids, _ = fused_topk(args[0], logits, 8, True)
                    capture(f'{l}/router_logits', logits)
                    capture(f'{l}/topk_ids', ids)
                    capture(f'{l}/topk_weights', weights)
                state['handles'].append(block.mlp.gate.register_forward_hook(router_hook))
            if layer in (41, 43):
                def kv_hook(module, args, l=layer):
                    capture(f'{l}/cache_k', args[1])
                    capture(f'{l}/cache_v', args[2])
                state['handles'].append(block.self_attn.attn.register_forward_pre_hook(kv_hook))
        def norm_hook(module, args, output):
            capture('final_hidden', output[0])
        state['handles'].append(model.model.norm.register_forward_hook(norm_hook))
        return {'rank': rank, 'status': 'INSTALLED'}


class BeginCapture:
    def __init__(self, sid, length):
        self.sid, self.length = sid, int(length)

    def __call__(self, model):
        s = getattr(model, ATTR)
        if s['sample_id'] is not None:
            raise RuntimeError('previous teacher sample not released')
        s.update(sample_id=self.sid, length=self.length, positions=None, fields={}, generated_logits=None)
        return {'rank': s['rank'], 'sample_id': self.sid}


class FinishCapture:
    def __init__(self, root, variant):
        self.root, self.variant = str(root), variant

    def __call__(self, model):
        s = getattr(model, ATTR)
        sid, length, rank = s['sample_id'], s['length'], s['rank']
        if sid is None:
            raise RuntimeError('no active teacher sample')
        tensors = {}
        for name, chunks in s['fields'].items():
            positions = torch.cat([p for p, _ in chunks])
            data = torch.cat([x for _, x in chunks])
            order = positions.argsort()
            if not torch.equal(positions[order], torch.arange(length)):
                raise RuntimeError(f'{sid} {name}: missing, duplicate or extra predictor positions')
            data = data[order].contiguous()
            if not torch.isfinite(data).all():
                raise RuntimeError(f'{sid} {name}: nonfinite actual tensor')
            tensors[name] = data
        s.update(sample_id=None, fields={}, positions=None)
        out = Path(self.root)/self.variant/sid
        replica_hashes = {name: tensor_hash(t) for name, t in tensors.items() if '/cache_' not in name}
        result = {'rank': rank, 'sample_id': sid, 'length': length, 'replica_sha256': replica_hashes,
                  'layers': {}, 'kv': {}, 'logits': []}
        for layer in LAYERS:
            payload = {name.split('/')[1]: value for name, value in tensors.items()
                       if name.startswith(f'{layer}/') and '/cache_' not in name}
            if not {'input', 'output', 'attn', 'moe', 'moe_input', 'post_attn'}.issubset(payload):
                raise RuntimeError('teacher layer boundary coverage incomplete')
            if rank == 0:
                result['layers'][str(layer)] = save_tensor(out/f'L{layer}.pt', payload)
            if layer in (41, 43):
                result['kv'][str(layer)] = save_tensor(out/f'L{layer}_kv_rank{rank}.pt',
                    {name: tensors[f'{layer}/{name}'] for name in ('cache_k', 'cache_v')})
        if rank == 0:
            result['final_hidden'] = save_tensor(out/'final_hidden.pt', tensors['final_hidden'])
        if self.variant == 'E0':
            device = model.lm_head.weight.device
            for start in range(0, length, 256):
                stop = min(start+256, length)
                # All TP ranks participate in the real LM Head collective.
                logits = model.compute_logits(tensors['final_hidden'][start:stop].to(device))
                if rank == 0:
                    if logits is None or logits.shape[0] != stop-start or not torch.isfinite(logits).all():
                        raise RuntimeError('invalid actual LM Head teacher logits')
                    result['logits'].append({**save_tensor(out/f'logits_{start:06d}.pt', logits.detach().cpu()),
                                             'start': start, 'stop': stop, 'vocab': logits.shape[-1]})
                    if stop == length:
                        from .path_audit import difference
                        if s['generated_logits'] is None:
                            raise RuntimeError('actual generation LM Head observation missing')
                        result['last_logit_batch_shape_difference'] = difference(logits[-1:], s['generated_logits'])
                del logits
        return result


def run_worker(root, variant, model_path, phasea_root):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    torch.set_num_threads(4)
    out = phase_root(root)/'actual_teacher'
    protocol = read_json(phase_root(root)/'protocol.json')
    samples = load_objective_calibration_samples(protocol['train_ids']+protocol['val_ids'])
    from .corrected_phase import token_hash
    if {s.sample_id:token_hash(s.input_ids) for s in samples}!=protocol['sample_token_sha256']:
        raise RuntimeError('teacher input tokens changed from the frozen protocol')
    samples.sort(key=lambda sample: (-len(sample.input_ids), sample.sample_id))
    llm, runtime = build_real_vllm(variant, model_path=model_path, phasea_root=phasea_root,
        gpu_memory_utilization=.60, max_num_seqs=1, max_num_batched_tokens=2048,
        enable_forced_trajectory_processor=False)
    replies = llm.apply_model(InstallCapture(str(out), variant))
    if len(replies) != 2:
        raise RuntimeError('teacher requires actual TP2')
    # build_real_vllm already resolves the loaded model to a local directory.
    # The repository's HF resolver accepts a repo ID, not this absolute path.
    model_dir=Path(runtime['model_path']).resolve()
    if not model_dir.is_dir():
        raise RuntimeError('loaded teacher model directory is missing')
    model_files=sorted(model_dir.glob('*.safetensors'))
    if not model_files:
        raise RuntimeError('actual teacher model weights cannot be fingerprinted')
    for name in ('config.json','model.safetensors.index.json','hif4_runtime_spec.pt'):
        path=model_dir/name
        if path.exists():model_files.append(path)
    runtime['files_sha256']={str(path):sha256(path) for path in model_files}
    execution_sha=sha256(phase_root(root)/'preparation_execution.json')
    rows = []
    for sample in samples:
        sid, length = sample.sample_id, len(sample.input_ids)
        print(f'{variant} actual teacher START {sid} tokens={length}', flush=True)
        llm.apply_model(BeginCapture(sid, length))
        llm.generate([TokensPrompt(prompt_token_ids=sample.input_ids.tolist())],
                     SamplingParams(max_tokens=1, temperature=0, ignore_eos=True), use_tqdm=False)
        replies = llm.apply_model(FinishCapture(str(out), variant))
        replies.sort(key=lambda r:r['rank'])
        if [r['rank'] for r in replies] != [0, 1] or replies[0]['replica_sha256'] != replies[1]['replica_sha256']:
            raise RuntimeError('TP teacher replicated boundaries disagree')
        row = {'status': 'COMPLETE', 'variant': variant, 'sample_id': sid,
               'length': length, 'token_sha256': protocol['sample_token_sha256'][sid], 'ranks': replies,
               'execution_sha256':execution_sha}
        atomic_write_json(out/variant/sid/'complete.json', row)
        rows.append(row)
        atomic_write_json(out/f'{variant}_progress.json', {'status': 'RUNNING', 'completed': len(rows),
                          'total': len(samples), 'last_sample': sid})
        print(f'{variant} actual teacher COMPLETE {sid} ({len(rows)}/{len(samples)})', flush=True)
    atomic_write_json(out/f'{variant}_complete.json', {'status': 'COMPLETE', 'runtime': runtime, 'samples': rows,
        'training_path_version': VERSION, 'execution_path': protocol['teacher_execution_path'],
        'execution_sha256':execution_sha})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--variant', choices=['E0', 'E1'], required=True)
    p.add_argument('--model-path', default=DEFAULT_MODEL_PATH)
    p.add_argument('--phasea-root', type=Path, default=DEFAULT_PHASEA_ROOT)
    a = p.parse_args()
    run_worker(a.root, a.variant, a.model_path, a.phasea_root)
