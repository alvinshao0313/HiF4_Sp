"""Production CUDA forwards with explicit derivatives through BF16 rounding.

No output correction or reference-tensor substitution: each forward calls the
same kernel as vLLM. Quantization uses STE; other backward rules differentiate
the represented norm, rotation, attention and routed matrix operations.
"""
import torch
import torch.nn.functional as F
from torch.nn.attention.bias import causal_lower_right


class RMS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps, fused):
        from vllm import _custom_ops as ops
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        w = weight.to(x.dtype).contiguous()
        if fused:
            out, residual = torch.zeros_like(x), x.contiguous().clone()
            ops.fused_add_rms_norm(out, residual, w, eps)
        else:
            out = torch.empty_like(x, memory_format=torch.contiguous_format)
            ops.rms_norm(out, x, w, eps)
        return out

    @staticmethod
    def backward(ctx, grad):
        from ..e2e_diag_reconstruction.core.moe_semantic_hif4 import _rms_norm
        x, w = ctx.saved_tensors
        with torch.enable_grad():
            a, b = x.detach().requires_grad_(), w.detach().requires_grad_()
            value = _rms_norm(a, b, ctx.eps)
            dx, dw = torch.autograd.grad(value, (a, b), grad)
        return dx, dw, None, None


def rms_norm(x, weight, eps, *, fused=True):
    if not x.is_cuda:
        raise ValueError("production RMSNorm requires CUDA")
    return RMS.apply(x, weight, eps, fused)


def final_norm(hidden, weight, eps):
    from .config import PROTOCOL
    flat = hidden.reshape(-1, hidden.shape[-1])
    return torch.cat([rms_norm(flat[i:i+PROTOCOL.prefill_chunk], weight, eps)
                      for i in range(0, len(flat), PROTOCOL.prefill_chunk)]).reshape_as(hidden)


class RotateQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        from vllm.model_executor.layers.quantization.hif4_transform_triton import hif4_r64_quantize_hifx4_triton
        ctx.dtype = x.dtype
        return hif4_r64_quantize_hifx4_triton(x.contiguous())

    @staticmethod
    def backward(ctx, grad):
        from ..diag_gradient.r64_transform import r64_matrix
        # R64 is symmetric, but retain the transpose in the derivative explicitly.
        r = r64_matrix(device=grad.device)
        return (grad.float().reshape(-1, 64) @ r.T).reshape_as(grad).to(ctx.dtype)


class Rotary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin):
        from vllm import _custom_ops as ops
        ctx.save_for_backward(cos, sin)
        n, heads, width = q.shape
        a, b = q.contiguous().clone(), k.contiguous().clone()
        cache = torch.cat((cos[:, :width//2], sin[:, :width//2]), -1).contiguous()
        positions = torch.arange(n, device=q.device)
        ops.rotary_embedding(positions, a.reshape(n, -1), b.reshape(n, -1), width, cache, True)
        return a, b

    @staticmethod
    def backward(ctx, dq, dk):
        cos, sin = ctx.saved_tensors
        def back(g):
            c, s = cos[:, None], sin[:, None]
            a, b = g.chunk(2, -1)
            return g * c + torch.cat((b, -a), -1) * s
        return back(dq), back(dk), None, None


class PagedAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        from vllm.vllm_flash_attn import flash_attn_varlen_func
        ctx.save_for_backward(q, k, v)
        nq, nk, width = len(q), len(k), q.shape[-1]
        block = 16
        padded = ((nk + block - 1) // block) * block
        kp = F.pad(k, (0, 0, 0, 0, 0, padded-nk)).reshape(-1, block, k.shape[1], width)
        vp = F.pad(v, (0, 0, 0, 0, 0, padded-nk)).reshape_as(kp)
        cu = torch.tensor([0, nq], dtype=torch.int32, device=q.device)
        used = torch.tensor([nk], dtype=torch.int32, device=q.device)
        table = torch.arange(len(kp), dtype=torch.int32, device=q.device).unsqueeze(0)
        return flash_attn_varlen_func(q=q.contiguous(), k=kp, v=vp,
            cu_seqlens_q=cu, seqused_k=used, max_seqlen_q=nq, max_seqlen_k=nk,
            block_table=table, causal=True, softmax_scale=width**-.5, fa_version=2)

    @staticmethod
    def backward(ctx, grad):
        q, k, v = ctx.saved_tensors
        with torch.enable_grad():
            a, b, c = [t.detach().requires_grad_() for t in (q, k, v)]
            groups = q.shape[1] // k.shape[1]
            query = a.transpose(0, 1).unsqueeze(0)
            key = b.repeat_interleave(groups, dim=1).transpose(0, 1).unsqueeze(0)
            value = c.repeat_interleave(groups, dim=1).transpose(0, 1).unsqueeze(0)
            out = F.scaled_dot_product_attention(query, key, value,
                attn_mask=causal_lower_right(len(q), len(k)), dropout_p=0., scale=q.shape[-1]**-.5)
            out = out[0].transpose(0, 1)
            grads = torch.autograd.grad(out, (a, b, c), grad)
        return grads


class Routing(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, top_k):
        from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
        weights, ids, _ = fused_topk(logits, logits, top_k, True)
        ctx.save_for_backward(weights, ids)
        ctx.shape, ctx.dtype = logits.shape, logits.dtype
        ctx.mark_non_differentiable(ids)
        return weights, ids

    @staticmethod
    def backward(ctx, grad, ignored):
        w, ids = ctx.saved_tensors
        # Renormalized top-k softmax has support only on the selected experts.
        local = w * (grad.float() - (w * grad.float()).sum(-1, keepdim=True))
        out = torch.zeros(ctx.shape, dtype=torch.float32, device=w.device)
        out.scatter_(1, ids.long(), local)
        return out.to(ctx.dtype), None


class SiluMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, packed):
        from vllm import _custom_ops as ops
        ctx.save_for_backward(packed)
        out = packed.new_empty((*packed.shape[:-1], packed.shape[-1]//2))
        torch.ops._C.silu_and_mul(out, packed)
        return out

    @staticmethod
    def backward(ctx, grad):
        packed, = ctx.saved_tensors
        with torch.enable_grad():
            x = packed.detach().requires_grad_()
            gate, up = x.chunk(2, -1)
            value = F.silu(gate) * up
            dx, = torch.autograd.grad(value, x, grad)
        return dx


class RoutedMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, route_weights, ids, sorted_ids, experts, padded, config, down):
        from vllm.model_executor.layers.fused_moe.fused_moe import invoke_fused_moe_triton_kernel
        from vllm.triton_utils import tl
        ctx.save_for_backward(x, weight, route_weights, ids)
        ctx.down = down
        out = x.new_empty((len(ids), ids.shape[1], weight.shape[1]))
        invoke_fused_moe_triton_kernel(x, weight, out, None, None,
            route_weights if down else None, sorted_ids, experts, padded,
            down, 1 if down else ids.shape[1], config, compute_type=tl.bfloat16,
            use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False,
            use_int4_w4a16=False, per_channel_quant=False, block_shape=None)
        return out

    @staticmethod
    def backward(ctx, grad):
        x, w, rw, ids = ctx.saved_tensors
        need_x, need_w, need_rw = ctx.needs_input_grad[:3]
        # Each routed slot is unique. Reduce the K contributions deterministically,
        # avoiding atomic BF16 scatter-adds on duplicated input token indices.
        dx = x.new_zeros((len(ids), ids.shape[1], x.shape[-1]), dtype=torch.float32) if need_x else None
        dw = torch.zeros_like(w) if need_w else None
        drw = torch.zeros_like(rw) if need_rw and ctx.down else None
        for expert in ids.unique(sorted=True).tolist():
            rows, positions = torch.where(ids == expert)
            source = rows * ids.shape[1] + positions if ctx.down else rows
            a, b = x[source].float(), w[expert].float()
            g = grad[rows, positions].float()
            if ctx.down:
                if drw is not None:
                    drw[rows, positions] = (g * (a @ b.T)).sum(-1)
                g = g * rw[rows, positions, None]
            if dx is not None:
                dx[rows, positions] = g @ b
            if dw is not None:
                dw[expert] = (g.T @ a).to(w.dtype)
        if dx is not None:
            dx = (dx.reshape_as(x) if ctx.down else dx.sum(1)).to(x.dtype)
        return dx, dw, drw, None, None, None, None, None, None


class SumExperts(torch.autograd.Function):
    @staticmethod
    def forward(ctx, routed):
        from vllm import _custom_ops as ops
        ctx.top_k = routed.shape[1]
        out = routed.new_empty((len(routed), routed.shape[-1]))
        ops.moe_sum(routed, out)
        return out

    @staticmethod
    def backward(ctx, grad):
        return grad[:, None].expand(-1, ctx.top_k, -1)


def moe_rank(x, w13, w2, weights, ids):
    from vllm.model_executor.layers.fused_moe.config import FUSED_MOE_UNQUANTIZED_CONFIG
    from vllm.model_executor.layers.fused_moe.fused_moe import try_get_optimal_moe_config
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
    config = try_get_optimal_moe_config(w13.size(), w2.size(), ids.shape[1],
        FUSED_MOE_UNQUANTIZED_CONFIG.config_name(x.dtype), len(x), block_shape=None)
    sorted_ids, experts, padded = moe_align_block_size(ids, config['BLOCK_SIZE_M'], len(w13))
    q = RotateQuantize.apply(x)
    packed = RoutedMM.apply(q, w13, weights, ids, sorted_ids, experts, padded, config, False)
    activated = SiluMultiply.apply(packed.reshape(-1, packed.shape[-1]))
    down_q = RotateQuantize.apply(activated)
    down = RoutedMM.apply(down_q, w2, weights, ids, sorted_ids, experts, padded, config, True)
    return SumExperts.apply(down)
