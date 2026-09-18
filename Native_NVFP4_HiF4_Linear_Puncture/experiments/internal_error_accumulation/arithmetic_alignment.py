"""Controlled arithmetic diagnostics against immutable TP2 captures.

The expert-loop ablations here explain rounding; they are never substituted for
production intervention evidence or used to release the training gate.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import subprocess
import sys

import torch
import torch.nn.functional as F

from . import selected_layer_objective_trainer as tr
from .candidate_runtime import sha256
from .capture_states import load_rank_records
from .config import DEFAULT_MODEL_PATH
from .corrected_phase import phase_root, write_frozen
from .path_audit import difference, selected_states
from .residual_ledger import index_capture_records
from .run_state import atomic_write_json, read_json, read_jsonl, write_jsonl
from ..e2e_diag_reconstruction.core import moe_semantic_hif4 as sem

LAYERS = (0, 1, 19, 31, 39, 41, 43, 47)
MODES = ('legacy', 'route_fp32', 'fused_silu', 'weighted_accumulator', 'sum_fp32', 'tp2')


def output_root(root):
    return phase_root(root) / 'arithmetic_alignment_v1'


def moe_ablations(st, x, variant):
    """Cumulative, explicitly named changes; same input and natural routes."""
    logits = F.linear(x, st.router_weight.to(x.dtype))
    weights, ids = torch.topk(logits.float().softmax(-1), st.spec.top_k, dim=-1)
    if st.spec.norm_topk_prob:
        weights = weights / weights.sum(-1, keepdim=True)
    contributions = {mode: [] for mode in MODES}
    tp_contributions = [[], []]
    # Only a single actual predictor is being compared. Keep the expert order
    # identical to the training loop for the first five ablations.
    for pos in torch.argsort(ids[0]).tolist():
        expert = st.experts[int(ids[0, pos])]
        def w(proj):
            master = getattr(expert, proj)
            return (master.to(x.dtype) if variant == 'E0' else
                    sem.qdq_hif4_direct(master, output_dtype=x.dtype))
        def q(t, proj):
            if variant == 'E1':
                return sem.qdq_hif4_direct(t.float(), output_dtype=x.dtype)
            meta = getattr(expert, proj.replace('_proj', '_metadata'))
            return sem.qdq_native_nvfp4(t, meta.input_global_scale_inv)
        wg, wu, wd = w('gate_proj'), w('up_proj'), w('down_proj')
        gate = F.linear(q(x, 'gate_proj'), wg)
        up = F.linear(q(x, 'up_proj'), wu)
        ordinary = q(F.silu(gate) * up, 'down_proj')
        fused = q((F.silu(gate.float()) * up.float()).to(x.dtype), 'down_proj')
        rw = weights[:, pos:pos+1]
        down = F.linear(ordinary, wd)
        contributions['legacy'].append(down * rw.to(x.dtype))
        contributions['route_fp32'].append((down.float() * rw).to(x.dtype))
        down_fused = F.linear(fused, wd)
        contributions['fused_silu'].append((down_fused.float() * rw).to(x.dtype))
        weighted = (F.linear(fused.float(), wd.float()) * rw).to(x.dtype)
        contributions['weighted_accumulator'].append(weighted)
        contributions['sum_fp32'].append(weighted)
        for rank in (0, 1):
            a = fused.chunk(2, -1)[rank].contiguous()
            b = wd.chunk(2, -1)[rank].contiguous()
            tp_contributions[rank].append((F.linear(a.float(), b.float()) * rw).to(x.dtype))
    results = {}
    for mode in MODES[:-1]:
        values = contributions[mode]
        if mode == 'sum_fp32':
            results[mode] = torch.stack(values).float().sum(0).to(x.dtype)
        else:
            value = torch.zeros_like(x)
            for term in values:
                value = value + term
            results[mode] = value
    partials = [torch.stack(v).float().sum(0).to(x.dtype) for v in tp_contributions]
    results['tp2'] = partials[0] + partials[1]
    return results, weights, ids


def worker(root, variant, model_path):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    out = output_root(root)
    if (out / f'{variant}_complete.json').exists():
        raise RuntimeError('refusing to overwrite complete arithmetic worker')
    device = torch.device('cuda:0')
    snapshot = Path(tr.resolve_local_snapshot(model_path))
    states = selected_states(root)
    captures = {}
    capture_hashes = {}
    # Keep only the required boundaries/layers, but all historical predictors.
    for meta in states:
        key = meta['sample_key']
        if len(meta['prompt_token_ids']) != 1:
            raise RuntimeError('full-history diagnostic requires captured position zero')
        hooks = phase_root(root) / 'alignment' / variant / 'hooks'
        records = []
        for rank in (0, 1):
            path = hooks / variant / key / f'rank{rank}.pt'
            capture_hashes[str(path.relative_to(root))] = sha256(path)
            records.extend(r for r in load_rank_records(hooks, variant, key, rank)
                           if r['layer'] in LAYERS and r['boundary'] in {
                               'input_norm', 'post_attn_norm', 'router_logits',
                               'qkv_proj', 'attention_core', 'o_proj', 'moe_out'})
        captures[key] = index_capture_records(records)
    rows = []
    with torch.no_grad():
        for layer in LAYERS:
            st = tr.load_qwen3_moe_layer_state(snapshot, layer, device)
            try:
                # Materialize the actual E1 routed kernel's TP-local weights.
                kernel_weights = []
                if variant == 'E1':
                    for rank in (0, 1):
                        w13, w2 = [], []
                        for expert in st.experts:
                            gate = sem.qdq_hif4_direct(expert.gate_proj, output_dtype=torch.bfloat16)
                            up = sem.qdq_hif4_direct(expert.up_proj, output_dtype=torch.bfloat16)
                            down = sem.qdq_hif4_direct(expert.down_proj, output_dtype=torch.bfloat16)
                            w13.append(torch.cat([gate.chunk(2, 0)[rank], up.chunk(2, 0)[rank]], 0))
                            w2.append(down.chunk(2, -1)[rank].contiguous())
                        kernel_weights.append((torch.stack(w13), torch.stack(w2)))
                for meta in states:
                    key, di = meta['sample_key'], meta['decode_index']
                    idx = captures[key]
                    def ref(boundary, role, rank=0, pos=di):
                        return idx[(key, pos, layer, boundary, role, rank)].reshape(1, -1).to(device)
                    row = {k: meta[k] for k in ('sample_key', 'source', 'calibration_sample_id', 'prefix_length_j')}
                    row.update(layer=layer, variant=variant)
                    x = torch.cat([ref('attention_core', 'rank_local', r) for r in (0, 1)], -1)
                    if variant == 'E0':
                        aq = sem.qdq_native_nvfp4(x, st.attention_metadata['o_proj'].input_global_scale_inv)
                        wq = st.attention['o_proj'].to(x.dtype)
                    else:
                        aq = sem.qdq_hif4_direct(x.float(), output_dtype=x.dtype)
                        wq = sem.qdq_hif4_direct(st.attention['o_proj'], output_dtype=x.dtype)
                    truth = ref('o_proj', 'tp_reduced')
                    row['o_full'] = difference(F.linear(aq, wq), truth)
                    partials = [F.linear(a.contiguous(), b.contiguous()) for a, b in zip(aq.chunk(2, -1), wq.chunk(2, -1))]
                    row['o_tp2'] = difference(partials[0] + partials[1], truth)
                    x = ref('post_attn_norm', 'normalized')
                    modes, weights, ids = moe_ablations(st, x, variant)
                    truth = ref('moe_out', 'tp_reduced')
                    for mode, value in modes.items():
                        row['moe_' + mode] = difference(value, truth)
                    if variant == 'E1':
                        from vllm.model_executor.layers.fused_moe.experts.hif4_emulation_moe import _apply_hif4_triton_moe
                        from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
                        router = ref('router_logits', 'logits')
                        prod_weights, prod_ids, _ = fused_topk(x, router, st.spec.top_k, st.spec.norm_topk_prob)
                        row['router_weights'] = difference(weights, prod_weights)
                        row['router_ids_equal'] = torch.equal(ids, prod_ids)
                        actual_parts = [_apply_hif4_triton_moe(x, a, b, prod_weights, prod_ids) for a, b in kernel_weights]
                        row['moe_production_kernel_tp2'] = difference(actual_parts[0] + actual_parts[1], truth)
                    # Attention replay starts with ACTUAL QKV for EVERY history
                    # position, thus isolates QK norm/RoPE/attention from QDQ.
                    h = torch.cat([ref('input_norm', 'normalized', pos=p) for p in range(di+1)], 0).unsqueeze(0)
                    pe = tr.build_qwen3_moe_layer_call(str(snapshot), h).position_embeddings
                    attn_parts = []
                    for rank in (0, 1):
                        qkv = torch.cat([ref('qkv_proj', 'rank_local', rank, p) for p in range(di+1)], 0).unsqueeze(0)
                        qs = st.spec.num_attention_heads * st.spec.head_dim // 2
                        ks = st.spec.num_key_value_heads * st.spec.head_dim // 2
                        q, k, v = qkv.split([qs, ks, ks], -1)
                        shape = (1, di+1, -1, st.spec.head_dim)
                        q = sem._rms_norm(q.reshape(shape), st.q_norm_weight, 1e-6).transpose(1, 2)
                        k = sem._rms_norm(k.reshape(shape), st.k_norm_weight, 1e-6).transpose(1, 2)
                        v = v.reshape(shape).transpose(1, 2)
                        q, k = sem.apply_rotary_pos_emb(q, k, *pe)
                        groups = st.spec.num_attention_heads // st.spec.num_key_value_heads
                        k, v = sem.repeat_kv(k, groups), sem.repeat_kv(v, groups)
                        # One last query against its entire causal history.
                        value = F.scaled_dot_product_attention(q[:, :, -1:], k, v,
                            dropout_p=0.0, is_causal=False, scale=1/math.sqrt(st.spec.head_dim)).reshape(1, -1)
                        row[f'attention_history_rank{rank}'] = difference(value, ref('attention_core', 'rank_local', rank))
                        attn_parts.append(value)
                    rows.append(row)
                write_jsonl(out / f'{variant}_rows.jsonl', rows)
                print(f'{variant}: arithmetic/history layer {layer} complete', flush=True)
            finally:
                kernel_weights = []
                tr.release_qwen3_moe_layer_state(st)
        atomic_write_json(out / f'{variant}_complete.json', {
            'status': 'COMPLETE', 'variant': variant, 'rows': len(rows),
            'rows_sha256': sha256(out / f'{variant}_rows.jsonl'),
            'capture_sha256': capture_hashes, 'training_started': False})


def run_arithmetic_alignment(*, run_root, model_path):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '2,3':
        raise RuntimeError('authorized devices must be exactly 2,3')
    root = Path(run_root).resolve()
    out = output_root(root)
    out.mkdir(parents=True, exist_ok=True)
    write_frozen(out / 'execution_manifest.json', {
        'protocol_sha256': sha256(phase_root(root) / 'protocol.json'),
        'source_sha256': sha256(Path(__file__)), 'layers': list(LAYERS),
        'dependencies_sha256': {str(Path(m.__file__).resolve()): sha256(Path(m.__file__))
                                for m in (tr, sem)},
        'variants': ['E0', 'E1'], 'modes': list(MODES), 'gpus': [2, 3],
        'scope': 'same-input arithmetic and complete captured history diagnosis; no gate release',
        'legacy_gate_sha256': sha256(phase_root(root) / 'alignment_gate.json')})
    jobs = []
    try:
        for variant, gpu in [('E0', '2'), ('E1', '3')]:
            log = (out / f'{variant}.log').open('x')
            proc = subprocess.Popen([sys.executable, '-u', '-m', __name__,
                '--root', str(root), '--variant', variant, '--model-path', model_path],
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': gpu}, stdout=log, stderr=subprocess.STDOUT)
            jobs.append((variant, proc, log))
        for variant, proc, _ in jobs:
            if proc.wait() != 0:
                raise RuntimeError(f'{variant} arithmetic worker failed; see {out / (variant + ".log")}')
        summary = {}
        for variant in ('E0', 'E1'):
            complete = read_json(out / f'{variant}_complete.json')
            if sha256(out / f'{variant}_rows.jsonl') != complete['rows_sha256']:
                raise RuntimeError('arithmetic evidence hash mismatch')
            rows = read_jsonl(out / f'{variant}_rows.jsonl')
            if len(rows) != len(LAYERS) * len(selected_states(root)):
                raise RuntimeError('incomplete arithmetic matrix')
            summary[variant] = {key: {'exact': sum(r[key]['equal'] for r in rows),
                'total': len(rows), 'mean_nmse': sum(r[key]['nmse'] for r in rows)/len(rows),
                'max_abs': max(r[key]['max_abs'] for r in rows)}
                for key in rows[0] if isinstance(rows[0][key], dict) and 'equal' in rows[0][key]}
        result = {'status': 'COMPLETE', 'training_gate': 'BLOCKED', 'summary': summary,
            'limitations': ['diagnostic expert loops do not replace production interventions',
                'full-path and nonzero-DIAG deployment alignment still required'],
            'training_started': False}
        atomic_write_json(out / 'summary.json', result)
        lines = ['# TP2 运算顺序与完整历史诊断', '', '训练门控仍为 BLOCKED。', '',
                 '| 路径 | 对照 | 逐位一致 | 平均 NMSE |', '|---|---|---|---|']
        for variant, metrics in summary.items():
            for name, value in metrics.items():
                lines.append(f'| {variant} | {name} | {value["exact"]}/{value["total"]} | {value["mean_nmse"]:.9g} |')
        lines += ['', 'MoE 各项为按顺序累加的运算语义改变，不能将改善比例解释为独立因果贡献。',
                  'Attention 使用每个 TP rank 自己的全部真实历史 QKV；仍需逐算子定位未闭合的差异。',
                  '本阶段不修改训练实现，不发布新 checkpoint，不覆盖原路径门控。']
        (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
        return result
    finally:
        for _, proc, log in jobs:
            if proc.poll() is None:
                proc.terminate()
                proc.wait()
            log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--variant', choices=['E0', 'E1'], required=True)
    parser.add_argument('--model-path', default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    worker(args.root, args.variant, args.model_path)
