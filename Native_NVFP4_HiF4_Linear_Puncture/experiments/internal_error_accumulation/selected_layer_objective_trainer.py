"""Selected-layer O0/O1/O2/O3/O4 trainer for IEA (plan O3.1 D).

Reuses e2e MoE runtime/calibration/optimizer primitives only.
Does not alter moe_trainer.py defaults or old Router KL semantics.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, IO

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.config import (
    DEFAULT_OPTIMIZER,
    DEFAULT_WEIGHT_DECAY,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.modelopt_moe_checkpoint import (
    load_qwen3_moe_layer_state,
    release_qwen3_moe_layer_state,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_fold import (
    folded_router_logits_from_pre_dgu_input,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_semantic_hif4 import (
    MoEFusableDiagState,
    NativeQwen3MoELayerRuntime,
    StudentMoEOutput,
    StudentQwen3MoELayerRuntime,
    StudentStepCache,
    build_moe_diag_state,
    forward_student_routed_moe,
    _rms_norm,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.data.batching import (
    DynamicCalibrationCollator,
    build_length_bucket_batches,
    build_validation_batches,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.data.calibration import (
    CalibrationSample,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.training.layer_runtime import (
    ProgressiveHiddenCache,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.training.losses import (
    finalize_reconstruction_loss,
    masked_reconstruction_components,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.training.moe_layer_runtime import (
    build_qwen3_moe_layer_call,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.training.moe_trainer import (
    build_initial_moe_hidden_cache,
)
from Native_NVFP4_HiF4_Linear_Puncture.src.checkpoint import resolve_local_snapshot
from Native_NVFP4_HiF4_Linear_Puncture.src.io_utils import ensure_dir

from .router_objective import (
    outside_teacher_topk_mass,
    router_full_kl,
    router_topk_total,
    topk_id_match_ratio,
    topk_margin_mean,
    topk_overlap_mean,
)
from .router_teacher_cache import (
    load_objective_calibration_samples,
    load_router_teacher_split,
    reshape_flat_router_logits,
    slice_router_logits_by_lengths,
)
from .run_state import atomic_write_json, read_json, read_jsonl


LOSS_O0 = "O0"
LOSS_O1_A = "O1_A"
LOSS_O1_M = "O1_M"
LOSS_O2_A = "O2_A"
LOSS_O2_M = "O2_M"
LOSS_O3_FULL = "O3_full"
LOSS_O3_TOPK = "O3_topk"
LOSS_O4 = "O4"
LOSS_JOINT = "joint"

O3_LOSSES = frozenset({LOSS_O3_FULL, LOSS_O3_TOPK})
ALL_LOSSES = frozenset(
    {
        LOSS_O0,
        LOSS_O1_A,
        LOSS_O1_M,
        LOSS_O2_A,
        LOSS_O2_M,
        LOSS_O3_FULL,
        LOSS_O3_TOPK,
        LOSS_O4,
        LOSS_JOINT,
    }
)

PARAM_TO_COMPONENT = {
    "D_QKV": "qkv",
    "D_VO": "vo",
    "D_GU": "gu",
    "D_UD": "ud",
}


_RECIPE_ARTIFACTS = (
    "checkpoint.pt",
    "train_metrics.jsonl",
    "val_metrics.json",
    "cost.json",
    "recipe.json",
)


def recipe_artifacts_complete(cand: Path) -> bool:
    cand = Path(cand)
    return all((cand / name).is_file() for name in _RECIPE_ARTIFACTS)


def _exclusive_recipe_lock(out_dir: Path) -> IO[str]:
    """Block until this candidate dir is exclusive; released when the file is closed."""
    import fcntl

    path = Path(out_dir) / "train.lock"
    handle = path.open("a+", encoding="utf-8")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return handle


def _reject_legacy_o3_tokens(recipe: dict) -> None:
    if recipe.get("loss") == "O3_conditional":
        raise RuntimeError("O3_conditional is forbidden; use O3_full or O3_topk")
    if "enable_o3" in recipe:
        raise RuntimeError("enable_o3 is forbidden on recipes; use o3_layers / O3_full|O3_topk")


def _params_to_active_components(params: list[str]) -> frozenset[str]:
    active: set[str] = set()
    for name in params:
        key = str(name)
        if key not in PARAM_TO_COMPONENT:
            raise ValueError(f"unknown DIAG param {key!r}; expected {sorted(PARAM_TO_COMPONENT)}")
        active.add(PARAM_TO_COMPONENT[key])
    if not active:
        raise ValueError("recipe.params must be non-empty")
    return frozenset(active)


def _configure_diag_from_params(diag_state: MoEFusableDiagState, params: list[str]) -> None:
    active = _params_to_active_components(params)
    # Bypass named presets so arbitrary unions (e.g. both-scope) are exact.
    diag_state.z_qkv.requires_grad_("qkv" in active)
    diag_state.z_vo.requires_grad_("vo" in active)
    diag_state.z_gu.requires_grad_("gu" in active)
    diag_state.z_ud.requires_grad_("ud" in active)


def _candidate_dir(run_root: Path, layer: int, loss: str) -> Path:
    return Path(run_root) / "60_objective" / "objective_candidates" / f"L{int(layer):02d}" / str(loss)


def _assemble_tensor_dict(
    store: dict[str, torch.Tensor],
    sample_ids: list[str],
    device: torch.device,
) -> torch.Tensor:
    hs = [store[sid] for sid in sample_ids]
    tmax = max(int(h.shape[0]) for h in hs)
    hidden = int(hs[0].shape[1])
    out = torch.zeros(len(hs), tmax, hidden, dtype=torch.bfloat16, device=device)
    for i, h in enumerate(hs):
        out[i, : h.shape[0]] = h.to(device=device)
    return out


def _masked_nmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    num, den = masked_reconstruction_components(
        pred, target, loss_mask, attention_mask, "block_output_nmse"
    )
    return finalize_reconstruction_loss(num, den, "block_output_nmse")


def _masked_block_delta_nmse(
    y_pred: torch.Tensor,
    y_tgt: torch.Tensor,
    x: torch.Tensor,
    loss_mask: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    num, den = masked_reconstruction_components(
        y_pred.float() - x.float(),
        y_tgt.float() - x.float(),
        loss_mask,
        attention_mask,
        "block_delta_nmse",
    )
    return finalize_reconstruction_loss(num, den, "block_delta_nmse")


def _strip_valid_rows(tensor_bth: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    chunks = [tensor_bth[i, : int(lengths[i].item())] for i in range(tensor_bth.shape[0])]
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def _native_layer_parts(
    runtime: NativeQwen3MoELayerRuntime,
    hidden: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, torch.Tensor]:
    residual = hidden
    normed = _rms_norm(hidden, runtime.state.input_layernorm_weight, runtime.rms_norm_eps)
    attn = runtime.attention_projections(normed, None, position_embeddings).o
    post_attn = residual + attn
    normed_moe = _rms_norm(post_attn, runtime.state.post_attention_layernorm_weight, runtime.rms_norm_eps)
    moe = runtime.routed_moe(normed_moe)
    output = post_attn + moe.output
    return {
        "attn_branch": attn,
        "post_attn": post_attn,
        "moe_branch": moe.output,
        "output": output,
        "router_logits": moe.router_logits,
    }


def _student_layer_parts(
    runtime: StudentQwen3MoELayerRuntime,
    hidden: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    *,
    use_ste: bool,
) -> dict[str, torch.Tensor | StudentMoEOutput]:
    """Single student forward; expose attn/moe parts without double compute."""
    cache = StudentStepCache.new()
    post_attn, normed = runtime.forward_to_router_input(
        hidden,
        attention_mask=None,
        position_embeddings=position_embeddings,
        step_cache=cache,
        use_ste=use_ste,
    )
    moe = forward_student_routed_moe(
        normed,
        runtime.state,
        runtime.diag_state,
        use_r64=runtime.use_r64,
        rot_order=runtime.rot_order,
        step_cache=cache,
        use_ste=use_ste,
    )
    output = post_attn + moe.output
    out = StudentMoEOutput(
        output,
        moe.router_logits,
        moe.routing_weights,
        moe.selected_experts,
        moe.per_expert_routed_token_count,
        router_input=moe.router_input,
    )
    return {
        "attn_branch": post_attn - hidden,
        "post_attn": post_attn,
        "moe_branch": moe.output,
        "output": output,
        "router_logits": moe.router_logits,
        "router_input": moe.router_input,
        "student_out": out,
    }


def _propagate_identity_student(
    *,
    snapshot: Path,
    samples: list[CalibrationSample],
    collator: DynamicCalibrationCollator,
    x_cache: ProgressiveHiddenCache,
    device: torch.device,
    batch_size: int,
    layer_idx: int,
) -> ProgressiveHiddenCache:
    """One progressive step with identity HiF4 student (z=0), no STE."""
    state = load_qwen3_moe_layer_state(snapshot, layer_idx, device)
    try:
        diag = build_moe_diag_state(state.spec, "fusable").to(device)
        assert isinstance(diag, MoEFusableDiagState)
        for p in diag.parameters():
            p.requires_grad_(False)
        student = StudentQwen3MoELayerRuntime(state, diag, use_r64=False, rot_order="diag_then_r64").to(device).eval()
        nxt = ProgressiveHiddenCache()
        for batch in build_validation_batches(samples, batch_size):
            packed = collator(batch)
            sample_ids = [s.sample_id for s in batch]
            hidden, _ = x_cache.assemble(sample_ids, device)
            call = build_qwen3_moe_layer_call(str(snapshot), hidden)
            with torch.no_grad():
                y = student(
                    hidden,
                    attention_mask=None,
                    position_embeddings=call.position_embeddings,
                    step_cache=StudentStepCache.new(),
                    use_ste=False,
                ).output
            for i, sample in enumerate(batch):
                n = int(packed["lengths"][i].item())
                nxt.store(sample.sample_id, y[i, :n], n)
        return nxt
    finally:
        release_qwen3_moe_layer_state(state)


def _build_layer_caches(
    *,
    snapshot: Path,
    samples: list[CalibrationSample],
    collator: DynamicCalibrationCollator,
    device: torch.device,
    batch_size: int,
    layer: int,
    need_e1: bool,
) -> dict[str, Any]:
    """Build frozen E0 (and optional E1) caches at the selected layer."""
    x_e0 = build_initial_moe_hidden_cache(snapshot, samples, collator, device, batch_size)
    x_e1 = None
    if need_e1:
        x_e1 = ProgressiveHiddenCache()
        for sid in [s.sample_id for s in samples]:
            x_e1.store(sid, x_e0.get(sid), int(x_e0.get(sid).shape[0]))

    e0_input: dict[str, torch.Tensor] = {}
    e0_output: dict[str, torch.Tensor] = {}
    e0_attn: dict[str, torch.Tensor] = {}
    e0_post_attn: dict[str, torch.Tensor] = {}
    e0_moe: dict[str, torch.Tensor] = {}
    e1_input: dict[str, torch.Tensor] = {}

    for layer_idx in range(0, layer + 1):
        state = load_qwen3_moe_layer_state(snapshot, layer_idx, device)
        try:
            native = NativeQwen3MoELayerRuntime(state).to(device).eval()
            nxt_e0 = ProgressiveHiddenCache()
            for batch in build_validation_batches(samples, batch_size):
                packed = collator(batch)
                sample_ids = [s.sample_id for s in batch]
                hidden, _ = x_e0.assemble(sample_ids, device)
                call = build_qwen3_moe_layer_call(str(snapshot), hidden)
                parts = _native_layer_parts(native, hidden, call.position_embeddings)
                for i, sample in enumerate(batch):
                    n = int(packed["lengths"][i].item())
                    sid = sample.sample_id
                    if layer_idx == layer:
                        e0_input[sid] = hidden[i, :n].detach().cpu().to(torch.bfloat16).contiguous()
                        e0_output[sid] = parts["output"][i, :n].detach().cpu().to(torch.bfloat16).contiguous()
                        e0_attn[sid] = parts["attn_branch"][i, :n].detach().cpu().to(torch.bfloat16).contiguous()
                        e0_post_attn[sid] = parts["post_attn"][i, :n].detach().cpu().to(torch.bfloat16).contiguous()
                        e0_moe[sid] = parts["moe_branch"][i, :n].detach().cpu().to(torch.bfloat16).contiguous()
                    nxt_e0.store(sid, parts["output"][i, :n], n)
            x_e0 = nxt_e0
        finally:
            release_qwen3_moe_layer_state(state)

        if need_e1:
            assert x_e1 is not None
            if layer_idx < layer:
                x_e1 = _propagate_identity_student(
                    snapshot=snapshot,
                    samples=samples,
                    collator=collator,
                    x_cache=x_e1,
                    device=device,
                    batch_size=batch_size,
                    layer_idx=layer_idx,
                )
            else:
                for batch in build_validation_batches(samples, batch_size):
                    packed = collator(batch)
                    sample_ids = [s.sample_id for s in batch]
                    hidden, _ = x_e1.assemble(sample_ids, device)
                    for i, sample in enumerate(batch):
                        n = int(packed["lengths"][i].item())
                        e1_input[sample.sample_id] = (
                            hidden[i, :n].detach().cpu().to(torch.bfloat16).contiguous()
                        )

    return {
        "e0_input": e0_input,
        "e0_output": e0_output,
        "e0_attn": e0_attn,
        "e0_post_attn": e0_post_attn,
        "e0_moe": e0_moe,
        "e1_input": e1_input,
    }


def _load_e0_teacher_batch(
    teacher_split: dict[str, dict[str, Any]],
    sample_ids: list[str],
    lengths: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits_chunks: list[torch.Tensor] = []
    ids_chunks: list[torch.Tensor] = []
    w_chunks: list[torch.Tensor] = []
    for i, sid in enumerate(sample_ids):
        if sid not in teacher_split:
            raise RuntimeError(f"missing E0 router teacher for sample {sid}")
        payload = teacher_split[sid]
        n = int(lengths[i].item())
        if int(payload["length"]) != n:
            raise RuntimeError(
                f"teacher length mismatch for {sid}: cache={payload['length']} batch={n}"
            )
        logits = payload["router_logits"]
        topk_ids = payload["topk_ids"]
        topk_weights = payload["topk_weights"]
        if int(logits.shape[0]) != n or int(topk_ids.shape[0]) != n or int(topk_weights.shape[0]) != n:
            raise RuntimeError(f"teacher tensor length mismatch for {sid}")
        logits_chunks.append(logits.to(device=device, dtype=torch.float32))
        ids_chunks.append(topk_ids.to(device=device, dtype=torch.int64))
        w_chunks.append(topk_weights.to(device=device, dtype=torch.float32))
    e0_logits = torch.cat(logits_chunks, dim=0)
    topk_ids_t = torch.cat(ids_chunks, dim=0)
    topk_w_t = torch.cat(w_chunks, dim=0)
    return e0_logits.detach(), topk_ids_t.detach(), topk_w_t.detach()


def _assert_router_helper_identity(
    *,
    out_router_logits: torch.Tensor,
    router_input_bth: torch.Tensor,
    lengths: torch.Tensor,
    router_weight: torch.Tensor,
    diag_state: MoEFusableDiagState,
) -> None:
    bsz, seq_len, _ = router_input_bth.shape
    logits_bte = reshape_flat_router_logits(
        out_router_logits.detach(), batch_size=bsz, seq_len=seq_len
    )
    real = torch.cat(slice_router_logits_by_lengths(logits_bte, lengths), dim=0)
    valid_input = _strip_valid_rows(router_input_bth, lengths).detach()
    with torch.no_grad():
        helper = folded_router_logits_from_pre_dgu_input(valid_input, router_weight, diag_state)
    if real.shape != helper.shape:
        raise RuntimeError(
            f"router identity shape mismatch real={tuple(real.shape)} helper={tuple(helper.shape)}"
        )
    if not torch.allclose(real.float(), helper.float(), rtol=1e-3, atol=1e-3):
        delta = (real.float() - helper.float()).abs().max().item()
        raise RuntimeError(
            f"router helper identity failed: max_abs={delta}; refusing surrogate Router training"
        )


def _lookup_router_causal_contribution(run_root: Path, layer: int) -> float | None:
    path = Path(run_root) / "40_router" / "router_intervention_rows.jsonl"
    if not path.is_file():
        return None
    rows = read_jsonl(path)
    vals = [float(r["C_router"]) for r in rows if int(r.get("layer", -1)) == int(layer) and "C_router" in r]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _load_final_norm_and_lm_head(snapshot: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    from safetensors import safe_open

    index = json.loads((snapshot / "model.safetensors.index.json").read_text(encoding="utf-8"))
    keys = ("model.norm.weight", "lm_head.weight")
    tensors: dict[str, torch.Tensor] = {}
    for key in keys:
        shard = index["weight_map"][key]
        with safe_open(str(snapshot / shard), framework="pt", device="cpu") as handle:
            tensors[key] = handle.get_tensor(key)
    norm_w = tensors["model.norm.weight"].to(device=device, dtype=torch.bfloat16)
    lm_head = tensors["lm_head.weight"].to(device=device, dtype=torch.bfloat16)
    return norm_w, lm_head


def _final_logits_from_hidden(
    hidden: torch.Tensor,
    lengths: torch.Tensor,
    norm_weight: torch.Tensor,
    lm_head_weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    token_chunk: int = 256,
) -> torch.Tensor:
    """Return concatenated valid-token logits [N_valid, V] without one giant matmul."""
    chunks: list[torch.Tensor] = []
    for i in range(hidden.shape[0]):
        n = int(lengths[i].item())
        h = hidden[i, :n]
        h_n = _rms_norm(h, norm_weight, eps).to(dtype=lm_head_weight.dtype)
        for start in range(0, n, int(token_chunk)):
            end = min(start + int(token_chunk), n)
            chunks.append(F.linear(h_n[start:end], lm_head_weight))
    return torch.cat(chunks, dim=0)


def _logsumexp_vocab_chunked(logits: torch.Tensor, *, vocab_chunk: int = 4096) -> torch.Tensor:
    """Exact logsumexp over vocab, chunked to avoid materializing full exp(V)."""
    if logits.ndim != 2:
        raise ValueError(f"expected [N,V] logits, got shape={tuple(logits.shape)}")
    n_tok, vocab = logits.shape
    chunk = max(1, int(vocab_chunk))
    # Running max then second pass — exact logsumexp.
    row_max = logits[:, : min(chunk, vocab)].max(dim=-1).values
    for start in range(chunk, vocab, chunk):
        end = min(start + chunk, vocab)
        row_max = torch.maximum(row_max, logits[:, start:end].max(dim=-1).values)
    acc = torch.zeros(n_tok, device=logits.device, dtype=torch.float32)
    for start in range(0, vocab, chunk):
        end = min(start + chunk, vocab)
        acc = acc + (logits[:, start:end].float() - row_max.unsqueeze(-1)).exp().sum(dim=-1)
    return row_max.float() + acc.clamp_min(torch.finfo(torch.float32).tiny).log()


def _kl_mean(p_logits: torch.Tensor, q_logits: torch.Tensor, *, vocab_chunk: int = 4096) -> torch.Tensor:
    """Exact token-mean KL(p||q); never materializes full float32 softmax([N,V])."""
    if p_logits.shape != q_logits.shape:
        raise RuntimeError(f"KL shape mismatch: p={tuple(p_logits.shape)} q={tuple(q_logits.shape)}")
    if p_logits.ndim != 2:
        raise ValueError(f"expected [N,V] logits, got p={tuple(p_logits.shape)}")
    n_tok, vocab = p_logits.shape
    if n_tok == 0:
        return p_logits.new_zeros(())
    chunk = max(1, int(vocab_chunk))
    lse_p = _logsumexp_vocab_chunked(p_logits, vocab_chunk=chunk)
    lse_q = _logsumexp_vocab_chunked(q_logits, vocab_chunk=chunk)
    kl_sum = torch.zeros((), device=p_logits.device, dtype=torch.float32)
    for start in range(0, vocab, chunk):
        end = min(start + chunk, vocab)
        lp = p_logits[:, start:end].float() - lse_p.unsqueeze(-1)
        lq = q_logits[:, start:end].float() - lse_q.unsqueeze(-1)
        kl_sum = kl_sum + (lp.exp() * (lp - lq)).sum()
    return kl_sum / float(n_tok)


def _o4_final_kl_mean(
    *,
    hidden: torch.Tensor,
    lengths: torch.Tensor,
    norm_weight: torch.Tensor,
    lm_head_weight: torch.Tensor,
    e0_final_logits: dict[str, torch.Tensor],
    sample_ids: list[str],
    eps: float = 1e-6,
    token_chunk: int = 128,
    vocab_chunk: int = 4096,
) -> torch.Tensor:
    """Exact O4 final-logit KL without allocating full-batch [N_valid, V] logits.

    Processes one sample / token microbatch at a time. Reduction matches
    ``F.kl_div(..., reduction='batchmean')`` over valid tokens.
    """
    token_chunk_i = max(1, int(token_chunk))
    if lm_head_weight.device != hidden.device:
        raise RuntimeError(
            f"O4 LM must stay on student device: lm={lm_head_weight.device} hidden={hidden.device}"
        )
    kl_sum = torch.zeros((), device=hidden.device, dtype=torch.float32)
    n_valid = 0
    for i, sid in enumerate(sample_ids):
        n = int(lengths[i].item())
        if n <= 0:
            continue
        p_full = e0_final_logits[sid]
        if int(p_full.shape[0]) != n:
            raise RuntimeError(f"O4 E0 logits length mismatch for {sid}: {p_full.shape[0]} vs {n}")
        h = hidden[i, :n]
        h_n = _rms_norm(h, norm_weight.to(device=h.device), eps)
        for start in range(0, n, token_chunk_i):
            end = min(start + token_chunk_i, n)
            h_chunk = h_n[start:end].to(dtype=lm_head_weight.dtype)
            q_logits = F.linear(h_chunk, lm_head_weight)
            p_logits = p_full[start:end].to(device=hidden.device, dtype=q_logits.dtype)
            # Scale token-mean KL back to sum over this microbatch.
            n_chunk = int(end - start)
            kl_sum = kl_sum + _kl_mean(p_logits.detach(), q_logits, vocab_chunk=vocab_chunk) * float(n_chunk)
            n_valid += n_chunk
            del q_logits, p_logits, h_chunk
    if n_valid <= 0:
        raise RuntimeError("O4 N_valid is 0")
    return kl_sum / float(n_valid)


def _o4_one_sample_kl(
    *,
    student: StudentQwen3MoELayerRuntime,
    snapshot: Path,
    x_store: dict[str, torch.Tensor],
    sample_id: str,
    n_tokens: int,
    layer: int,
    num_layers: int,
    device: torch.device,
    train_mode: bool,
    norm_w: torch.Tensor,
    lm_w: torch.Tensor,
    e0_final_logits: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, int]:
    """Forward one sample through student+subsequent+KL. Returns (token_mean_kl, n)."""
    if n_tokens <= 0:
        raise RuntimeError(f"O4 sample {sample_id} has n_tokens={n_tokens}")
    h_i = _assemble_tensor_dict(x_store, [sample_id], device)
    if int(h_i.shape[1]) != int(n_tokens):
        h_i = h_i[:, :n_tokens, :].contiguous()
    else:
        h_i = h_i.contiguous()
    call_i = build_qwen3_moe_layer_call(str(snapshot), h_i)
    parts_i = _student_layer_parts(
        student,
        h_i,
        call_i.position_embeddings,
        use_ste=bool(train_mode),
    )
    y_final_i = _o4_subsequent_native(
        snapshot=snapshot,
        hidden=parts_i["output"].contiguous(),
        start_layer=layer + 1,
        num_layers=num_layers,
        device=device,
        use_checkpoint=True,
    )
    lengths_i = torch.tensor([n_tokens], device=device, dtype=torch.long)
    kl_i = _o4_final_kl_mean(
        hidden=y_final_i,
        lengths=lengths_i,
        norm_weight=norm_w,
        lm_head_weight=lm_w,
        e0_final_logits=e0_final_logits,
        sample_ids=[sample_id],
        token_chunk=32,
        vocab_chunk=2048,
    )
    return kl_i, int(n_tokens)


def _o4_subsequent_native(
    *,
    snapshot: Path,
    hidden: torch.Tensor,
    start_layer: int,
    num_layers: int,
    device: torch.device,
    use_checkpoint: bool | None = None,
) -> torch.Tensor:
    """Forward through subsequent native layers; grads flow to `hidden` only.

    Always checkpoints by default. Entire O4 autograd must stay on one device —
    never put LM/KL on another GPU in the same loss graph as checkpoint.
    """
    from torch.utils.checkpoint import checkpoint

    if hidden.ndim != 3:
        raise ValueError(f"O4 subsequent expects [B,S,H], got {tuple(hidden.shape)}")
    y = hidden.contiguous()
    if use_checkpoint is None:
        use_checkpoint = True

    def _one(h: torch.Tensor, layer_idx: int) -> torch.Tensor:
        st = load_qwen3_moe_layer_state(snapshot, layer_idx, device)
        try:
            native = NativeQwen3MoELayerRuntime(st).to(device).eval()
            call = build_qwen3_moe_layer_call(str(snapshot), h)
            return native(
                h,
                attention_mask=None,
                position_embeddings=call.position_embeddings,
            ).output
        finally:
            release_qwen3_moe_layer_state(st)

    for layer_idx in range(start_layer, num_layers):
        if use_checkpoint:
            y = checkpoint(
                lambda h, li=int(layer_idx): _one(h, li),
                y,
                use_reentrant=False,
            )
        else:
            y = _one(y, int(layer_idx))
    return y


def compute_o3_router_aux(
    *,
    loss_name: str,
    router_input_bth: torch.Tensor,
    lengths: torch.Tensor,
    router_weight: torch.Tensor,
    diag_state: MoEFusableDiagState,
    e0_logits: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Detach router input, recompute folded logits, return aux loss + metrics."""
    valid_input = _strip_valid_rows(router_input_bth, lengths).detach()
    candidate = folded_router_logits_from_pre_dgu_input(valid_input, router_weight, diag_state)
    n_valid = int(candidate.shape[0])
    if n_valid != int(e0_logits.shape[0]) or n_valid != int(topk_ids.shape[0]) or n_valid != int(topk_weights.shape[0]):
        raise RuntimeError(
            f"N_valid mismatch: cand={n_valid} e0={e0_logits.shape[0]} "
            f"ids={topk_ids.shape[0]} w={topk_weights.shape[0]}"
        )
    metrics: dict[str, float] = {}
    if loss_name == LOSS_O3_FULL:
        aux = router_full_kl(e0_logits, candidate)
        metrics["router_full_kl"] = float(aux.detach().item())
    elif loss_name == LOSS_O3_TOPK:
        total, weight_kl, hinge = router_topk_total(topk_ids, topk_weights, candidate)
        aux = total
        metrics["router_topk_weight_kl"] = float(weight_kl.detach().item())
        metrics["router_topk_support_hinge"] = float(hinge.detach().item())
        metrics["router_topk_total"] = float(total.detach().item())
    else:
        raise ValueError(f"not an O3 loss: {loss_name}")
    with torch.no_grad():
        metrics["topk_id_match_ratio"] = float(topk_id_match_ratio(topk_ids, candidate).item())
        metrics["topk_overlap_mean"] = float(topk_overlap_mean(topk_ids, candidate).item())
        metrics["topk_margin_mean"] = float(topk_margin_mean(topk_ids, candidate).item())
        metrics["outside_teacher_topk_mass"] = float(outside_teacher_topk_mass(topk_ids, candidate).item())
    return aux, metrics


def train_selected_layer_recipe(
    *,
    recipe: dict,
    run_root: Path,
    model_path: str,
    phasea_root: Path,
) -> dict:
    """Train one selected-layer recipe; save final checkpoint (no best-epoch pick)."""
    _reject_legacy_o3_tokens(recipe)
    run_root = Path(run_root)
    layer = int(recipe["layer"])
    loss_name = str(recipe["loss"])
    out_dir = ensure_dir(_candidate_dir(run_root, layer, loss_name))
    lock_fh = _exclusive_recipe_lock(out_dir)
    try:
        if recipe_artifacts_complete(out_dir):
            return {
                "status": "SKIPPED_COMPLETE",
                "layer": layer,
                "loss": loss_name,
                "out_dir": str(out_dir),
            }
        return _train_selected_layer_recipe_unlocked(
            recipe=recipe,
            run_root=run_root,
            model_path=model_path,
            phasea_root=phasea_root,
        )
    finally:
        lock_fh.close()


def _train_selected_layer_recipe_unlocked(
    *,
    recipe: dict,
    run_root: Path,
    model_path: str,
    phasea_root: Path,
) -> dict:
    """Train one selected-layer recipe; save final checkpoint (no best-epoch pick)."""
    _ = phasea_root  # reserved for deploy/materialize paths; fair init is identity zeros
    _reject_legacy_o3_tokens(recipe)

    run_root = Path(run_root)
    layer = int(recipe["layer"])
    loss_name = str(recipe["loss"])
    if loss_name not in ALL_LOSSES:
        raise ValueError(f"unsupported loss={loss_name!r}")
    params = list(recipe["params"])
    train_ids = [str(x) for x in recipe["train_ids"]]
    val_ids = [str(x) for x in recipe["val_ids"]]
    if set(train_ids) & set(val_ids):
        raise RuntimeError("train_ids and val_ids must be disjoint")

    seed = int(recipe.get("seed", 20260909))
    lr = float(recipe.get("lr", 5e-3))
    weight_decay = float(recipe.get("weight_decay", DEFAULT_WEIGHT_DECAY))
    optimizer_name = str(recipe.get("optimizer", DEFAULT_OPTIMIZER))
    batch_size = int(recipe.get("batch_size", recipe.get("diag_batch_size", 4)))
    epochs = int(recipe.get("epochs", recipe.get("diag_epochs", 20)))
    max_steps = recipe.get("steps", None)
    max_steps_i = int(max_steps) if max_steps is not None else None
    router_lambda = float(recipe.get("router_lambda", 0.0))
    use_r64 = bool(recipe.get("use_r64", False))
    rot_order = str(recipe.get("rot_order", "diag_then_r64"))

    if loss_name in O3_LOSSES:
        if "router_lambda" not in recipe:
            raise RuntimeError("O3 recipe requires router_lambda")
        if "router_teacher_cache_dir" not in recipe:
            raise RuntimeError("O3 recipe requires router_teacher_cache_dir")
        cache_dir = Path(recipe["router_teacher_cache_dir"])
        manifest = read_json(cache_dir / "manifest.json")
        if manifest.get("status") != "COMPLETE":
            raise RuntimeError(
                f"router teacher cache status={manifest.get('status')!r}; need COMPLETE"
            )
        if int(layer) not in {int(x) for x in manifest.get("layers", [])}:
            raise RuntimeError(f"layer {layer} absent from router teacher cache layers")
    else:
        cache_dir = None

    out_dir = ensure_dir(_candidate_dir(run_root, layer, loss_name))
    atomic_write_json(out_dir / "recipe.json", dict(recipe))

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    snapshot = Path(resolve_local_snapshot(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    collator = DynamicCalibrationCollator(int(tokenizer.pad_token_id))

    train_samples = load_objective_calibration_samples(train_ids)
    val_samples = load_objective_calibration_samples(val_ids)
    all_for_cache = list(train_samples) + list(val_samples)

    need_e1 = loss_name in {
        LOSS_O2_A,
        LOSS_O2_M,
        LOSS_O3_FULL,
        LOSS_O3_TOPK,
        LOSS_O4,
        LOSS_JOINT,
    }
    t_cache0 = time.perf_counter()
    caches = _build_layer_caches(
        snapshot=snapshot,
        samples=all_for_cache,
        collator=collator,
        device=device,
        batch_size=batch_size,
        layer=layer,
        need_e1=need_e1,
    )
    cache_build_s = time.perf_counter() - t_cache0

    teacher_train = teacher_val = None
    if loss_name in O3_LOSSES:
        assert cache_dir is not None
        teacher_train = load_router_teacher_split(cache_dir, layer=layer, split="train")
        teacher_val = load_router_teacher_split(cache_dir, layer=layer, split="val")

    state = load_qwen3_moe_layer_state(snapshot, layer, device)
    train_metrics_path = out_dir / "train_metrics.jsonl"
    if train_metrics_path.exists():
        train_metrics_path.unlink()

    cost: dict[str, Any] = {
        "cache_build_seconds": cache_build_s,
        "train_seconds": 0.0,
        "val_seconds": 0.0,
        "optimizer_steps": 0,
        "tokens_seen": 0,
        "loss": loss_name,
        "layer": layer,
    }

    try:
        diag_state = build_moe_diag_state(state.spec, "fusable").to(device)
        assert isinstance(diag_state, MoEFusableDiagState)
        _configure_diag_from_params(diag_state, params)
        # Router weight must stay frozen for O3 aux.
        state.router_weight.requires_grad_(False)

        student = StudentQwen3MoELayerRuntime(
            state,
            diag_state,
            use_r64=use_r64,
            rot_order=rot_order,
        ).to(device)

        trainable = [p for p in diag_state.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError(f"no trainable DIAG params for params={params}")
        if optimizer_name != "AdamW":
            raise ValueError(f"unsupported optimizer={optimizer_name!r}")
        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)

        # Precompute E0 final logits for O4.
        e0_final_logits: dict[str, torch.Tensor] | None = None
        norm_w = lm_w = None
        if loss_name == LOSS_O4:
            # Entire O4 autograd (student/subsequent/LM/KL) must stay on one device.
            # Cross-device LM + checkpoint subsequent caused CUDA illegal memory access.
            norm_w, lm_w = _load_final_norm_and_lm_head(snapshot, device)
            e0_final_logits = {}
            # Continue E0 from selected-layer output through remaining native layers once.
            x_rest = ProgressiveHiddenCache()
            for sid, h in caches["e0_output"].items():
                x_rest.store(sid, h, int(h.shape[0]))
            for layer_idx in range(layer + 1, state.spec.num_layers):
                st = load_qwen3_moe_layer_state(snapshot, layer_idx, device)
                try:
                    native = NativeQwen3MoELayerRuntime(st).to(device).eval()
                    nxt = ProgressiveHiddenCache()
                    for batch in build_validation_batches(all_for_cache, batch_size):
                        packed = collator(batch)
                        sids = [s.sample_id for s in batch]
                        hidden, _ = x_rest.assemble(sids, device)
                        call = build_qwen3_moe_layer_call(str(snapshot), hidden)
                        with torch.no_grad():
                            y = native(
                                hidden,
                                attention_mask=None,
                                position_embeddings=call.position_embeddings,
                            ).output
                        for i, sample in enumerate(batch):
                            n = int(packed["lengths"][i].item())
                            nxt.store(sample.sample_id, y[i, :n], n)
                    x_rest = nxt
                finally:
                    release_qwen3_moe_layer_state(st)
            for batch in build_validation_batches(all_for_cache, max(1, min(batch_size, 2))):
                packed = collator(batch)
                sids = [s.sample_id for s in batch]
                hidden, _ = x_rest.assemble(sids, device)
                with torch.no_grad():
                    lengths_b = packed["lengths"]
                    for i, sample in enumerate(batch):
                        n = int(lengths_b[i].item())
                        h_i = hidden[i : i + 1, :n].contiguous()
                        logits_i = _final_logits_from_hidden(
                            h_i,
                            torch.tensor([n], device=device),
                            norm_w,
                            lm_w,
                            token_chunk=128,
                        )
                        e0_final_logits[sample.sample_id] = logits_i.detach().cpu()
                        del h_i, logits_i
                del hidden
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            # O4 train path only needs E1 inputs + CPU E0 final logits.
            for drop_key in ("e0_output", "e0_moe", "e0_attn", "e0_post_attn", "e0_input"):
                caches.pop(drop_key, None)
            del x_rest
            if device.type == "cuda":
                torch.cuda.empty_cache()

        def _input_store_for_loss() -> dict[str, torch.Tensor]:
            if loss_name in {LOSS_O0, LOSS_O1_A, LOSS_O1_M}:
                return caches["e0_input"]
            return caches["e1_input"]

        def _compute_base_and_aux(
            *,
            batch: list[CalibrationSample],
            packed: dict[str, Any],
            train_mode: bool,
        ) -> tuple[torch.Tensor, dict[str, float]]:
            sample_ids = [s.sample_id for s in batch]
            lengths = packed["lengths"].to(device)
            loss_mask = packed["loss_mask"].to(device)
            attn_mask = packed["attention_mask"].to(device)
            x_store = _input_store_for_loss()
            metrics: dict[str, float] = {}

            if train_mode:
                student.train()
            else:
                student.eval()

            # O4 val / metric path: per-sample forward (no retained multi-sample graph).
            # Train uses true per-sample backward in the optimizer loop below.
            if loss_name == LOSS_O4:
                assert norm_w is not None and lm_w is not None and e0_final_logits is not None
                kl_sum = torch.zeros((), device=device, dtype=torch.float32)
                n_valid_total = 0
                for i, sid in enumerate(sample_ids):
                    n = int(lengths[i].item())
                    if n <= 0:
                        continue
                    kl_i, n_i = _o4_one_sample_kl(
                        student=student,
                        snapshot=snapshot,
                        x_store=x_store,
                        sample_id=sid,
                        n_tokens=n,
                        layer=layer,
                        num_layers=state.spec.num_layers,
                        device=device,
                        train_mode=bool(train_mode),
                        norm_w=norm_w,
                        lm_w=lm_w,
                        e0_final_logits=e0_final_logits,
                    )
                    kl_sum = kl_sum + kl_i * float(n_i)
                    n_valid_total += n_i
                    del kl_i
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                if n_valid_total <= 0:
                    raise RuntimeError("O4 N_valid is 0")
                base = kl_sum / float(n_valid_total)
                metrics["final_logit_kl"] = float(base.detach().item())
                metrics.setdefault("loss", float(base.detach().item()))
                return base, metrics

            hidden = _assemble_tensor_dict(x_store, sample_ids, device)
            call = build_qwen3_moe_layer_call(str(snapshot), hidden)
            parts = _student_layer_parts(
                student,
                hidden,
                call.position_embeddings,
                use_ste=bool(train_mode),
            )
            out = parts["student_out"]

            if loss_name == LOSS_O0:
                tgt = _assemble_tensor_dict(caches["e0_output"], sample_ids, device)
                base = _masked_block_delta_nmse(parts["output"], tgt, hidden, loss_mask, attn_mask)
            elif loss_name == LOSS_O1_M:
                tgt = _assemble_tensor_dict(caches["e0_moe"], sample_ids, device)
                base = _masked_nmse(parts["moe_branch"], tgt, loss_mask, attn_mask)
            elif loss_name == LOSS_O1_A:
                tgt = _assemble_tensor_dict(caches["e0_attn"], sample_ids, device)
                base = _masked_nmse(parts["attn_branch"], tgt, loss_mask, attn_mask)
            elif loss_name == LOSS_O2_M:
                tgt = _assemble_tensor_dict(caches["e0_output"], sample_ids, device)
                # R_A1 + M_theta vs R_{l+1}^0  <=> full output vs E0 output on E1 input path
                base = _masked_nmse(parts["output"], tgt, loss_mask, attn_mask)
                metrics["cumulative_residual_nmse"] = float(base.detach().item())
            elif loss_name == LOSS_O2_A:
                tgt = _assemble_tensor_dict(caches["e0_post_attn"], sample_ids, device)
                base = _masked_nmse(parts["post_attn"], tgt, loss_mask, attn_mask)
                metrics["cumulative_residual_nmse"] = float(base.detach().item())
            elif loss_name == LOSS_JOINT:
                tgt_m = _assemble_tensor_dict(caches["e0_output"], sample_ids, device)
                tgt_a = _assemble_tensor_dict(caches["e0_post_attn"], sample_ids, device)
                loss_m = _masked_nmse(parts["output"], tgt_m, loss_mask, attn_mask)
                loss_a = _masked_nmse(parts["post_attn"], tgt_a, loss_mask, attn_mask)
                base = loss_m + loss_a
                metrics["cumulative_residual_nmse"] = float(base.detach().item())
            elif loss_name in O3_LOSSES:
                tgt = _assemble_tensor_dict(caches["e0_output"], sample_ids, device)
                base = _masked_nmse(parts["output"], tgt, loss_mask, attn_mask)
                metrics["cumulative_residual_nmse"] = float(base.detach().item())
                if out.router_input is None:
                    raise RuntimeError("O3 requires out.router_input")
                _assert_router_helper_identity(
                    out_router_logits=out.router_logits,
                    router_input_bth=out.router_input,
                    lengths=lengths,
                    router_weight=state.router_weight,
                    diag_state=diag_state,
                )
                teacher = teacher_train if train_mode else teacher_val
                assert teacher is not None
                e0_logits, topk_ids, topk_w = _load_e0_teacher_batch(
                    teacher, sample_ids, lengths.cpu(), device
                )
                aux, router_metrics = compute_o3_router_aux(
                    loss_name=loss_name,
                    router_input_bth=out.router_input,
                    lengths=lengths,
                    router_weight=state.router_weight,
                    diag_state=diag_state,
                    e0_logits=e0_logits,
                    topk_ids=topk_ids,
                    topk_weights=topk_w,
                )
                metrics.update(router_metrics)
                total = base + float(router_lambda) * aux
                metrics["base_loss"] = float(base.detach().item())
                metrics["router_lambda"] = float(router_lambda)
                return total, metrics
            else:
                raise RuntimeError(f"unhandled loss {loss_name}")

            metrics.setdefault("loss", float(base.detach().item()))
            return base, metrics

        # Fixed step budget (fairness): epochs * n_batches, or explicit steps.
        train_batches = build_length_bucket_batches(train_samples, batch_size, seed)
        if not train_batches:
            raise RuntimeError("no training batches")
        planned_steps = int(max_steps_i) if max_steps_i is not None else int(epochs) * len(train_batches)

        t_train0 = time.perf_counter()
        steps_done = 0
        epoch = 0
        while steps_done < planned_steps:
            epoch_batches = build_length_bucket_batches(
                train_samples, batch_size, seed + epoch
            )
            for batch in epoch_batches:
                if steps_done >= planned_steps:
                    break
                packed = collator(batch)
                optimizer.zero_grad(set_to_none=True)
                if loss_name == LOSS_O4:
                    # True per-sample backward: each sample frees its graph before the next.
                    # Grad of sum_i (kl_i * n_i / N) equals grad of token-mean KL.
                    assert norm_w is not None and lm_w is not None and e0_final_logits is not None
                    sample_ids = [s.sample_id for s in batch]
                    lengths = packed["lengths"].to(device)
                    x_store = caches["e1_input"]
                    n_valid_total = 0
                    for i in range(len(sample_ids)):
                        n_i = int(lengths[i].item())
                        if n_i > 0:
                            n_valid_total += n_i
                    if n_valid_total <= 0:
                        raise RuntimeError("O4 N_valid is 0")
                    student.train()
                    kl_weighted_sum = 0.0
                    for i, sid in enumerate(sample_ids):
                        n = int(lengths[i].item())
                        if n <= 0:
                            continue
                        kl_i, n_i = _o4_one_sample_kl(
                            student=student,
                            snapshot=snapshot,
                            x_store=x_store,
                            sample_id=sid,
                            n_tokens=n,
                            layer=layer,
                            num_layers=state.spec.num_layers,
                            device=device,
                            train_mode=True,
                            norm_w=norm_w,
                            lm_w=lm_w,
                            e0_final_logits=e0_final_logits,
                        )
                        if not torch.isfinite(kl_i):
                            raise RuntimeError(f"non-finite O4 kl at step={steps_done} sample={sid}")
                        scaled = kl_i * (float(n_i) / float(n_valid_total))
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                        scaled.backward()
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                        kl_weighted_sum += float(kl_i.detach().item()) * float(n_i)
                        del kl_i, scaled
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                    loss_value = kl_weighted_sum / float(n_valid_total)
                    metrics = {
                        "loss": float(loss_value),
                        "final_logit_kl": float(loss_value),
                    }
                else:
                    loss, metrics = _compute_base_and_aux(
                        batch=batch, packed=packed, train_mode=True
                    )
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"non-finite train loss at step={steps_done}")
                    if device.type == "cuda":
                        # Surface async CUDA faults at the true step instead of empty_cache.
                        torch.cuda.synchronize()
                    loss.backward()
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    loss_value = float(loss.detach().item())
                    del loss
                optimizer.step()
                diag_state.clamp_log2_()
                steps_done += 1
                tokens = int(packed["lengths"].sum().item())
                cost["tokens_seen"] = int(cost["tokens_seen"]) + tokens
                row = {
                    "step": steps_done,
                    "epoch": epoch,
                    "loss": float(loss_value),
                    **{k: float(v) for k, v in metrics.items()},
                }
                with train_metrics_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                if device.type == "cuda" and loss_name == LOSS_O4 and (steps_done % 8 == 0):
                    torch.cuda.empty_cache()
            epoch += 1
        cost["train_seconds"] = time.perf_counter() - t_train0
        cost["optimizer_steps"] = steps_done
        # Final checkpoint only (no objective-specific best-epoch selection).
        final_snapshot = diag_state.snapshot()
        torch.save(
            {
                "layer": layer,
                "loss": loss_name,
                "params": params,
                "diag": final_snapshot,
                "steps": steps_done,
                "seed": seed,
            },
            out_dir / "checkpoint.pt",
        )

        t_val0 = time.perf_counter()
        val_totals: dict[str, float] = {}
        val_counts = 0
        with torch.no_grad():
            for batch in build_validation_batches(val_samples, batch_size):
                packed = collator(batch)
                loss, metrics = _compute_base_and_aux(batch=batch, packed=packed, train_mode=False)
                val_totals["loss"] = val_totals.get("loss", 0.0) + float(loss.item())
                for k, v in metrics.items():
                    val_totals[k] = val_totals.get(k, 0.0) + float(v)
                val_counts += 1
        cost["val_seconds"] = time.perf_counter() - t_val0
        val_metrics = {k: (v / max(val_counts, 1)) for k, v in val_totals.items()}
        val_metrics["optimizer_steps"] = steps_done
        val_metrics["layer"] = layer
        val_metrics["loss_name"] = loss_name
        c_router = _lookup_router_causal_contribution(run_root, layer)
        val_metrics["router_causal_contribution"] = c_router

        # Ensure O3 required keys exist (plan F).
        if loss_name in O3_LOSSES:
            required = [
                "cumulative_residual_nmse",
                "router_causal_contribution",
                "topk_id_match_ratio",
                "topk_overlap_mean",
                "topk_margin_mean",
                "outside_teacher_topk_mass",
            ]
            if loss_name == LOSS_O3_TOPK:
                required += [
                    "router_topk_weight_kl",
                    "router_topk_support_hinge",
                    "router_topk_total",
                ]
            if loss_name == LOSS_O3_FULL:
                required += ["router_full_kl"]
            for key in required:
                if key not in val_metrics:
                    val_metrics[key] = None
            # final_logit_kl is deferred to holdout eval; record explicit null if not computed.
            val_metrics.setdefault("final_logit_kl", None)

        atomic_write_json(out_dir / "val_metrics.json", val_metrics)
        atomic_write_json(out_dir / "cost.json", cost)
        return {
            "status": "COMPLETE",
            "layer": layer,
            "loss": loss_name,
            "out_dir": str(out_dir),
            "steps": steps_done,
            "val_metrics": val_metrics,
            "cost": cost,
        }
    finally:
        try:
            release_qwen3_moe_layer_state(state)
        except Exception:
            # Don't mask the original training fault with a secondary CUDA cleanup error.
            pass
        if device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
