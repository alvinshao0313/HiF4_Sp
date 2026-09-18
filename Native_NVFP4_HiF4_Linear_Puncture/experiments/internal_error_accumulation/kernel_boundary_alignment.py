"""Trace actual MoE kernels and isolate full-history attention arithmetic."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import torch
import torch.nn.functional as F

from . import selected_layer_objective_trainer as tr
from .arithmetic_alignment import LAYERS
from .candidate_runtime import sha256
from .capture_states import load_rank_records
from .config import DEFAULT_MODEL_PATH
from .corrected_phase import phase_root, write_frozen
from .path_audit import difference, selected_states
from .residual_ledger import index_capture_records
from .run_state import atomic_write_json, read_jsonl, write_jsonl
from ..e2e_diag_reconstruction.core import moe_semantic_hif4 as sem


def output_root(root):
    return phase_root(root) / 'kernel_boundary_alignment_v1'


def captures(root, variant, boundaries):
    result, hashes = {}, {}
    hooks = phase_root(root) / 'alignment' / variant / 'hooks'
    for meta in selected_states(root):
        key = meta['sample_key']
        records = []
        for rank in (0, 1):
            path = hooks / variant / key / f'rank{rank}.pt'
            hashes[str(path.relative_to(root))] = sha256(path)
            records.extend(r for r in load_rank_records(hooks, variant, key, rank)
                           if r['layer'] in LAYERS and r['boundary'] in boundaries)
        result[key] = index_capture_records(records)
    return result, hashes


def trace_moe(kernel, x, w13, w2, weights, ids):
    """Observe the real kernel's temporary buffers before workspace reuse."""
    trace = {}
    invoke, quantize = kernel.invoke_fused_moe_triton_kernel, kernel.hif4_quantize_hifx4_triton
    calls = {'gemm': 0, 'qdq': 0}
    def gemm(*args, **kwargs):
        value = invoke(*args, **kwargs)
        calls['gemm'] += 1
        trace[f'gemm{calls["gemm"]}'] = args[2].clone()
        return value
    def qdq(x, *args, **kwargs):
        calls['qdq'] += 1
        n = calls['qdq']
        trace[f'qdq{n}_input'] = x.clone()
        value = quantize(x, *args, **kwargs)
        trace[f'qdq{n}_output'] = kwargs['out'].clone()
        return value
    with patch.object(kernel, 'invoke_fused_moe_triton_kernel', gemm), \
         patch.object(kernel, 'hif4_quantize_hifx4_triton', qdq):
        value = kernel._apply_hif4_triton_moe(x, w13, w2, weights, ids)
    if calls != {'gemm': 2, 'qdq': 2}:
        raise RuntimeError(f'unexpected production MoE call structure: {calls}')
    return value, trace


def moe_worker(root, snapshot, device):
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.fused_moe.experts import hif4_emulation_moe as kernel
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
    refs, hashes = captures(root, 'E1', {'post_attn_norm', 'router_logits', 'moe_out'})
    rows = []
    for layer in LAYERS:
        st = tr.load_qwen3_moe_layer_state(snapshot, layer, device)
        try:
            weights_by_rank = []
            for rank in (0, 1):
                w13, w2 = [], []
                for expert in st.experts:
                    q = lambda w: sem.qdq_hif4_direct(w, output_dtype=torch.bfloat16)
                    w13.append(torch.cat([q(expert.gate_proj).chunk(2, 0)[rank], q(expert.up_proj).chunk(2, 0)[rank]], 0))
                    w2.append(q(expert.down_proj).chunk(2, -1)[rank].contiguous())
                weights_by_rank.append((torch.stack(w13), torch.stack(w2)))
            for meta in selected_states(root):
                key, di = meta['sample_key'], meta['decode_index']
                ref = lambda b, r: refs[key][(key, di, layer, b, r, 0)].reshape(1, -1).to(device)
                x = ref('post_attn_norm', 'normalized')
                logits = ref('router_logits', 'logits')
                rw, ids, _ = fused_topk(x, logits, st.spec.top_k, st.spec.norm_topk_prob)
                probs = logits.float().softmax(-1)
                _, ti = probs.topk(st.spec.top_k, -1)
                # Fixed-production-route control. Natural route order/set
                # differences are reported separately rather than confounded
                # with expert arithmetic or silently treated as identical.
                tw = probs.gather(1, ids.long())
                if st.spec.norm_topk_prob:
                    tw /= tw.sum(-1, keepdim=True)
                row = {'sample_key': key, 'layer': layer, 'variant': 'E1',
                       'natural_route_order_equal': torch.equal(ids, ti),
                       'natural_route_set_equal': torch.equal(ids.sort(-1).values, ti.sort(-1).values),
                       'production_ids': ids[0].tolist(), 'training_ids': ti[0].tolist()}
                row['fixed_set_router_weights'] = difference(tw, rw)
                actual, reconstructed, torch_router = [], [], []
                for rank, (w13, w2) in enumerate(weights_by_rank):
                    value, t = trace_moe(kernel, x, w13, w2, rw, ids)
                    repeated = kernel._apply_hif4_triton_moe(x, w13, w2, rw, ids)
                    row[f'r{rank}_observer_noop'] = difference(value, repeated)
                    if not row[f'r{rank}_observer_noop']['equal']:
                        raise RuntimeError('MoE observer changed production output')
                    xq = sem.qdq_hif4_direct(x.float(), output_dtype=x.dtype)
                    row[f'r{rank}_input_qdq'] = difference(xq, t['qdq1_output'])
                    gates = torch.stack([F.linear(xq, w13[int(i)]) for i in ids[0]], 1)
                    row[f'r{rank}_gate_up'] = difference(gates, t['gemm1'])
                    g, u = t['gemm1'].reshape(-1, w13.shape[1]).chunk(2, -1)
                    activation = F.silu(g) * u
                    row[f'r{rank}_silu_mul'] = difference(activation, t['qdq2_input'])
                    aq = sem.qdq_hif4_direct(t['qdq2_input'].float(), output_dtype=x.dtype)
                    row[f'r{rank}_activation_qdq'] = difference(aq, t['qdq2_output'])
                    weighted = torch.stack([(F.linear(t['qdq2_output'][p:p+1].float(), w2[int(i)].float()) * rw[:, p:p+1]).to(x.dtype)
                                            for p, i in enumerate(ids[0])], 1)
                    row[f'r{rank}_weighted_down'] = difference(weighted, t['gemm2'])
                    summed = t['gemm2'].float().sum(1).to(x.dtype)
                    row[f'r{rank}_expert_sum'] = difference(summed, value)
                    # Reconstruct the entire expert path, keeping BF16 SiLU
                    # and using FP32 route weights before down-output casting.
                    g, u = gates.reshape(-1, w13.shape[1]).chunk(2, -1)
                    aq = sem.qdq_hif4_direct((F.silu(g) * u).float(), output_dtype=x.dtype)
                    terms = torch.stack([(F.linear(aq[p:p+1].float(), w2[int(i)].float()) * rw[:, p:p+1]).to(x.dtype)
                                         for p, i in enumerate(ids[0])], 1)
                    rebuilt = terms.float().sum(1).to(x.dtype)
                    terms_torch = torch.stack([(F.linear(aq[p:p+1].float(), w2[int(i)].float()) * tw[:, p:p+1]).to(x.dtype)
                                               for p, i in enumerate(ids[0])], 1)
                    row[f'r{rank}_chain'] = difference(rebuilt, value)
                    actual.append(value)
                    reconstructed.append(rebuilt)
                    torch_router.append(terms_torch.float().sum(1).to(x.dtype))
                truth = ref('moe_out', 'tp_reduced')
                row['production_identity'] = difference(actual[0]+actual[1], truth)
                if not row['production_identity']['equal']:
                    raise RuntimeError('actual kernel replay no longer matches captured TP2 output')
                row['reconstructed_chain'] = difference(reconstructed[0]+reconstructed[1], truth)
                row['torch_router_chain'] = difference(torch_router[0]+torch_router[1], truth)
                rows.append(row)
            write_jsonl(output_root(root)/'moe_rows.jsonl', rows)
            print(f'MoE layer {layer} complete', flush=True)
        finally:
            weights_by_rank = []
            tr.release_qwen3_moe_layer_state(st)
    return rows, hashes


def attention_worker(root, snapshot, device):
    from vllm import _custom_ops as ops
    from vllm.config import VllmConfig, CompilationConfig, set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.v1.attention.backends.fa_utils import get_flash_attn_version, flash_attn_varlen_func
    config = sem.qwen3_moe_config_from_snapshot(snapshot)
    # Match the recorded eager production configuration's enabled custom ops.
    with set_current_vllm_config(VllmConfig(compilation_config=CompilationConfig(mode=0, custom_ops=['all']))), torch.device(device):
        rope = get_rope(config.head_dim, config.max_position_embeddings,
                        rope_parameters=config.rope_parameters, dtype=torch.bfloat16)
    version = get_flash_attn_version(head_size=config.head_dim)
    rows, hashes = [], {}
    for variant in ('E0', 'E1'):
        refs, source_hashes = captures(root, variant, {'qkv_proj', 'attention_core'})
        hashes.update(source_hashes)
        for layer in LAYERS:
            st = tr.load_qwen3_moe_layer_state(snapshot, layer, device)
            try:
                for meta in selected_states(root):
                    key, di = meta['sample_key'], meta['decode_index']
                    if len(meta['prompt_token_ids']) != 1:
                        raise RuntimeError('missing prefix history')
                    for rank in (0, 1):
                        row = {'sample_key': key, 'layer': layer, 'rank': rank, 'variant': variant, 'fa_version': version}
                        qkv = torch.cat([refs[key][(key, p, layer, 'qkv_proj', 'rank_local', rank)].reshape(1, -1)
                                         for p in range(di+1)], 0).to(device)
                        qs = st.spec.num_attention_heads * st.spec.head_dim // 2
                        ks = st.spec.num_key_value_heads * st.spec.head_dim // 2
                        q, k, v = qkv.split([qs, ks, ks], -1)
                        q, k, v = [t.reshape(di+1, -1, st.spec.head_dim) for t in (q, k, v)]
                        qn = sem._rms_norm(q, st.q_norm_weight, 1e-6)
                        kn = sem._rms_norm(k, st.k_norm_weight, 1e-6)
                        qc, kc = torch.empty_like(q), torch.empty_like(k)
                        ops.rms_norm(qc, q.contiguous(), st.q_norm_weight, 1e-6)
                        ops.rms_norm(kc, k.contiguous(), st.k_norm_weight, 1e-6)
                        row['q_norm'] = difference(qn, qc)
                        row['k_norm'] = difference(kn, kc)
                        dummy = torch.empty((1, di+1, st.spec.hidden_size), device=device, dtype=q.dtype)
                        pe = tr.build_qwen3_moe_layer_call(str(snapshot), dummy).position_embeddings
                        qh, kh = sem.apply_rotary_pos_emb(qn.transpose(0,1).unsqueeze(0), kn.transpose(0,1).unsqueeze(0), *pe)
                        qh, kh = qh[0].transpose(0,1), kh[0].transpose(0,1)
                        positions = torch.arange(di+1, device=device)
                        qr, kr = rope.forward_cuda(positions, qc.reshape(di+1, -1).clone(), kc.reshape(di+1, -1).clone())
                        qr, kr = qr.reshape_as(q), kr.reshape_as(k)
                        row['q_rope'] = difference(qh, qr)
                        row['k_rope'] = difference(kh, kr)
                        truth = refs[key][(key, di, layer, 'attention_core', 'rank_local', rank)].to(device)
                        groups = st.spec.num_attention_heads // st.spec.num_key_value_heads
                        def sdpa(a, b):
                            aa = a.transpose(0,1).unsqueeze(0)[:, :, -1:]
                            bb = sem.repeat_kv(b.transpose(0,1).unsqueeze(0), groups)
                            vv = sem.repeat_kv(v.transpose(0,1).unsqueeze(0), groups)
                            return F.scaled_dot_product_attention(aa, bb, vv, dropout_p=0.0, is_causal=False).reshape(-1)
                        row['hf_rope_sdpa'] = difference(sdpa(qh, kh), truth)
                        row['production_rope_sdpa'] = difference(sdpa(qr, kr), truth)
                        cuq = torch.tensor([0, 1], dtype=torch.int32, device=device)
                        cuk = torch.tensor([0, di+1], dtype=torch.int32, device=device)
                        flash = flash_attn_varlen_func(q=qr[-1:].contiguous(), k=kr.contiguous(), v=v.contiguous(),
                            max_seqlen_q=1, cu_seqlens_q=cuq, max_seqlen_k=di+1, cu_seqlens_k=cuk,
                            causal=True, softmax_scale=1/math.sqrt(st.spec.head_dim), fa_version=version)
                        row['production_rope_flash_varlen'] = difference(flash, truth)
                        rows.append(row)
                write_jsonl(output_root(root)/'attention_rows.jsonl', rows)
                print(f'Attention {variant} layer {layer} complete', flush=True)
            finally:
                tr.release_qwen3_moe_layer_state(st)
    return rows, hashes


def worker(root, kind, model_path):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    out = output_root(root)
    if (out/f'{kind}_complete.json').exists():
        raise RuntimeError('refusing completed worker overwrite')
    snapshot = Path(tr.resolve_local_snapshot(model_path))
    with torch.no_grad():
        rows, hashes = (moe_worker if kind == 'moe' else attention_worker)(root, snapshot, torch.device('cuda:0'))
    atomic_write_json(out/f'{kind}_complete.json', {'status': 'COMPLETE', 'rows': len(rows),
        'rows_sha256': sha256(out/f'{kind}_rows.jsonl'), 'capture_sha256': hashes})


def run_kernel_alignment(*, run_root, model_path):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '2,3':
        raise RuntimeError('authorized devices are 2,3')
    root = Path(run_root).resolve()
    out = output_root(root)
    out.mkdir(exist_ok=False)
    write_frozen(out/'execution_manifest.json', {'source_sha256': sha256(Path(__file__)),
        'semantic_runtime_sha256': sha256(Path(sem.__file__)),
        'protocol_sha256': sha256(phase_root(root)/'protocol.json'), 'layers': list(LAYERS),
        'workers': {'moe': 2, 'attention': 3}, 'training_started': False})
    jobs = []
    try:
        for kind, gpu in [('moe', '2'), ('attention', '3')]:
            log = (out/f'{kind}.log').open('x')
            proc = subprocess.Popen([sys.executable, '-u', '-m', __name__, '--root', str(root),
                '--kind', kind, '--model-path', model_path], stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': gpu})
            jobs.append((kind, proc, log))
        for kind, proc, _ in jobs:
            if proc.wait() != 0:
                raise RuntimeError(f'{kind} boundary worker failed: {out / (kind + ".log")}')
        summary = {}
        for kind, count in [('moe', 128), ('attention', 512)]:
            rows = read_jsonl(out/f'{kind}_rows.jsonl')
            if len(rows) != count:
                raise RuntimeError('incomplete fixed audit matrix')
            summary[kind] = {key: {'exact': sum(r[key]['equal'] for r in rows), 'total': len(rows),
                'mean_nmse': sum(r[key]['nmse'] for r in rows)/len(rows)}
                for key in rows[0] if isinstance(rows[0][key], dict)}
        atomic_write_json(out/'summary.json', {'status': 'COMPLETE', 'training_gate': 'BLOCKED', 'summary': summary})
        lines = ['# 生产核心逐边界诊断', '', '训练门控保持 BLOCKED。', '',
                 '| 路径 | 边界 | 逐位一致 | 平均 NMSE |', '|---|---|---|---|']
        for kind, metrics in summary.items():
            for key, val in metrics.items():
                lines.append(f'| {kind} | {key} | {val["exact"]}/{val["total"]} | {val["mean_nmse"]:.9g} |')
        lines += ['', 'MoE 中间输出来自实际 routed Triton 核心，并检查观察器 no-op 和 TP2 输出 identity。',
                  'Attention 使用真实全部历史 QKV；本地生产 norm/RoPE 与训练实现逐项对照。',
                  'Flash varlen 与实际 paged decode 的布局差异仍需以最终 identity 结果判断；本阶段不放行训练。']
        (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    finally:
        for _, proc, log in jobs:
            if proc.poll() is None:
                proc.terminate()
                proc.wait()
            log.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--kind', choices=['moe', 'attention'], required=True)
    p.add_argument('--model-path', default=DEFAULT_MODEL_PATH)
    a = p.parse_args()
    worker(a.root, a.kind, a.model_path)
