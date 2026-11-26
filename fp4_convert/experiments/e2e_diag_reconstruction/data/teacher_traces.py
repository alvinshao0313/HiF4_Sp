"""Native NVFP4 teacher CoT generation and self-judge."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from fp4_convert.experiments.e2e_diag_reconstruction.core.config import (
    E2ETrainConfig,
)
from fp4_convert.experiments.e2e_diag_reconstruction.core.semantic_hif4 import (
    set_model_runtime_mode,
)

TEACHER_TEMPERATURE = 0.6
TEACHER_TOP_P = 0.95
TEACHER_TOP_K = 20
TEACHER_MIN_P = 0.0
JUDGE_MAX_NEW_TOKENS = 8
QWEN3_CHAT_TEMPLATE_SOURCE = "Qwen/Qwen3-8B"


def teacher_generation_seed(calib_seed: int, split_id: int, source_index: int, attempt_index: int) -> int:
    return int(calib_seed + 100000 * split_id + 1000 * source_index + attempt_index)


def _load_official_qwen3_chat_template() -> str:
    from transformers import AutoTokenizer

    src = AutoTokenizer.from_pretrained(
        QWEN3_CHAT_TEMPLATE_SOURCE,
        local_files_only=True,
        trust_remote_code=True,
    )
    tmpl = getattr(src, "chat_template", None)
    if not tmpl:
        raise RuntimeError(f"{QWEN3_CHAT_TEMPLATE_SOURCE} tokenizer has no chat_template")
    return str(tmpl)


def require_qwen3_chat_template(tokenizer) -> None:
    """NVFP4 snapshot 未携带 chat_template 时，挂上官方 Qwen3 模板。缺失则 fail。"""
    if getattr(tokenizer, "chat_template", None):
        return
    tokenizer.chat_template = _load_official_qwen3_chat_template()
    print(
        "NVFP4 tokenizer has no chat_template; "
        f"attached official {QWEN3_CHAT_TEMPLATE_SOURCE} chat_template",
        flush=True,
    )


def _as_1d_token_ids(ids) -> torch.Tensor:
    """Normalize apply_chat_template output to a 1-D CPU long tensor."""
    if torch.is_tensor(ids):
        t = ids
    elif isinstance(ids, Mapping):
        if "input_ids" not in ids:
            raise TypeError(
                "apply_chat_template mapping has no input_ids, "
                f"keys={list(ids.keys())}"
            )
        t = ids["input_ids"]
    else:
        raise TypeError(
            "apply_chat_template must return a Tensor or BatchEncoding with input_ids, "
            f"got {type(ids)}"
        )
    if not torch.is_tensor(t):
        t = torch.as_tensor(t)
    return t.reshape(-1).to(dtype=torch.long, device="cpu")


def build_teacher_prompt_ids(tokenizer, question: str) -> torch.Tensor:
    require_qwen3_chat_template(tokenizer)
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
        return_tensors="pt",
    )
    return _as_1d_token_ids(ids)


def build_judge_prompt_ids(tokenizer, question: str, solution: str, candidate: str) -> torch.Tensor:
    require_qwen3_chat_template(tokenizer)
    prompt = (
        "You are a strict grader.\n"
        f"Question:\n{question}\n\n"
        f"Reference solution:\n{solution}\n\n"
        f"Candidate response:\n{candidate}\n\n"
        "Is the candidate response correct? Reply with exactly one token: CORRECT or INCORRECT."
    )
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    )
    return _as_1d_token_ids(ids)


def _move_ids(ids: torch.Tensor, device: torch.device) -> torch.Tensor:
    return ids.to(device=device).unsqueeze(0)


def generate_teacher_trace(
    *,
    tokenizer,
    native_model,
    question: str,
    max_new_tokens: int,
    seed: int,
) -> dict[str, Any]:
    prompt_ids = build_teacher_prompt_ids(tokenizer, question)
    device = next(native_model.parameters()).device
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("tokenizer has neither pad_token_id nor eos_token_id")
    torch.manual_seed(seed)
    set_model_runtime_mode(native_model, "native_nvfp4")
    native_model.eval()
    with torch.no_grad():
        out = native_model.generate(
            input_ids=_move_ids(prompt_ids, device),
            attention_mask=torch.ones(1, int(prompt_ids.numel()), device=device, dtype=torch.long),
            do_sample=True,
            temperature=TEACHER_TEMPERATURE,
            top_p=TEACHER_TOP_P,
            top_k=TEACHER_TOP_K,
            min_p=TEACHER_MIN_P,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=pad_id,
        )
    full = out[0].detach().to(device="cpu", dtype=torch.long)
    prompt_len = int(prompt_ids.numel())
    if int(full.numel()) < prompt_len:
        raise RuntimeError("teacher generate returned fewer tokens than the prompt")
    generated = full[prompt_len:]
    eos = tokenizer.eos_token_id
    truncated = int(generated.numel()) >= max_new_tokens and (
        eos is None or int(generated[-1].item()) != int(eos)
    )
    text = tokenizer.decode(generated, skip_special_tokens=False)
    return {
        "prompt_ids": prompt_ids.cpu(),
        "generated_ids": generated.contiguous(),
        "full_ids": full.contiguous(),
        "generated_text": text,
        "truncated": bool(truncated),
        "prompt_len": prompt_len,
        "n_generated": int(generated.numel()),
    }


def judge_teacher_trace(
    *,
    tokenizer,
    native_model,
    question: str,
    solution: str,
    candidate: str,
) -> str:
    prompt_ids = build_judge_prompt_ids(tokenizer, question, solution, candidate)
    device = next(native_model.parameters()).device
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    set_model_runtime_mode(native_model, "native_nvfp4")
    native_model.eval()
    with torch.no_grad():
        out = native_model.generate(
            input_ids=_move_ids(prompt_ids, device),
            attention_mask=torch.ones(1, int(prompt_ids.numel()), device=device, dtype=torch.long),
            do_sample=False,
            max_new_tokens=JUDGE_MAX_NEW_TOKENS,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=pad_id,
        )
    gen = out[0, int(prompt_ids.numel()) :].detach().to("cpu")
    text = tokenizer.decode(gen, skip_special_tokens=True).strip()
    if text not in {"CORRECT", "INCORRECT"}:
        raise RuntimeError(f"unknown judge output: {text!r}")
    return text


def _trace_to_sample(
    source_index: int,
    trace: dict[str, Any],
    meta: dict[str, Any],
):
    from fp4_convert.experiments.e2e_diag_reconstruction.data.calibration import (
        sample_from_ids_and_mask,
    )
    prompt_len = int(trace["prompt_len"])
    full = trace["full_ids"]
    mask = torch.zeros_like(full)
    mask[prompt_len:] = 1
    if int(full.numel()) == prompt_len:
        raise RuntimeError(f"teacher produced empty continuation for source_index={source_index}")
    return sample_from_ids_and_mask(
        f"s1k_teacher_cot_{source_index}",
        source_index,
        full,
        mask,
        meta,
    )


def generate_split_teacher_traces(
    *,
    cfg: E2ETrainConfig,
    tokenizer,
    native_model,
    dataset,
    base_ids: list[int],
    unused_ids: list[int],
    split_name: str,
    split_id: int,
) -> tuple[list, list[dict[str, Any]]]:
    require_qwen3_chat_template(tokenizer)
    samples = []
    traces: list[dict[str, Any]] = []
    from fp4_convert.experiments.e2e_diag_reconstruction.data.calibration import (
        require_s1k_fields,
    )

    unused = list(unused_ids)
    unused_pos = 0

    def _one(source_index: int, attempt: int) -> dict[str, Any]:
        row = dataset[source_index]
        require_s1k_fields(row, source_index)
        seed = teacher_generation_seed(cfg.calib_seed, split_id, source_index, attempt)
        print(
            f"[teacher] {split_name} source_index={source_index} attempt={attempt} generating",
            flush=True,
        )
        trace = generate_teacher_trace(
            tokenizer=tokenizer,
            native_model=native_model,
            question=row["question"],
            max_new_tokens=cfg.teacher_max_new_tokens,
            seed=seed,
        )
        judge = None
        if not trace["truncated"] or cfg.teacher_trace_policy == "all":
            if not trace["truncated"]:
                judge = judge_teacher_trace(
                    tokenizer=tokenizer,
                    native_model=native_model,
                    question=row["question"],
                    solution=row["solution"],
                    candidate=trace["generated_text"],
                )
        trace["judge"] = judge
        trace["source_index"] = source_index
        trace["attempt"] = attempt
        trace["question"] = row["question"]
        print(
            f"[teacher] {split_name} source_index={source_index} "
            f"n_generated={trace['n_generated']} truncated={trace['truncated']} "
            f"judge={trace['judge']}",
            flush=True,
        )
        return trace

    if cfg.teacher_trace_policy == "all":
        for source_index in base_ids:
            tr = _one(source_index, 0)
            sample = _trace_to_sample(
                source_index,
                tr,
                {
                    "source": "s1k_teacher_cot",
                    "truncated": tr["truncated"],
                    "judge": tr["judge"],
                    "attempts": 1,
                    "question_source_index": source_index,
                },
            )
            samples.append(sample)
            traces.append(_trace_meta(split_name, tr))
        return samples, traces

    if cfg.teacher_trace_policy == "regenerate_correct":
        for source_index in base_ids:
            last = None
            accepted = None
            for attempt in range(cfg.teacher_max_attempts):
                last = _one(source_index, attempt)
                traces.append(_trace_meta(split_name, last))
                if (not last["truncated"]) and last["judge"] == "CORRECT":
                    accepted = last
                    n_try = attempt + 1
                    break
            if accepted is None:
                raise RuntimeError(
                    f"regenerate_correct failed for source_index={source_index} "
                    f"after {cfg.teacher_max_attempts} attempts"
                )
            samples.append(
                _trace_to_sample(
                    source_index,
                    accepted,
                    {
                        "source": "s1k_teacher_cot",
                        "truncated": False,
                        "judge": "CORRECT",
                        "attempts": n_try,
                        "question_source_index": source_index,
                    },
                )
            )
        return samples, traces

    if cfg.teacher_trace_policy == "replace_question_correct":
        for _slot, source_index in enumerate(base_ids):
            candidate_ids = [source_index]
            chosen = None
            while chosen is None:
                if not candidate_ids:
                    if unused_pos >= len(unused):
                        raise RuntimeError(
                            f"replace_question_correct exhausted the unused S1K pool on {split_name}"
                        )
                    candidate_ids = [unused[unused_pos]]
                    unused_pos += 1
                cid = candidate_ids.pop(0)
                tr = _one(cid, 0)
                traces.append(_trace_meta(split_name, tr))
                if (not tr["truncated"]) and tr["judge"] == "CORRECT":
                    chosen = tr
            samples.append(
                _trace_to_sample(
                    int(chosen["source_index"]),
                    chosen,
                    {
                        "source": "s1k_teacher_cot",
                        "truncated": False,
                        "judge": "CORRECT",
                        "attempts": 1,
                        "question_source_index": int(chosen["source_index"]),
                    },
                )
            )
        train_val_overlap_guard(samples)
        return samples, traces

    raise ValueError(cfg.teacher_trace_policy)


def train_val_overlap_guard(samples: list) -> None:
    ids = [s.source_index for s in samples]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate source_index in teacher-cot split")


def _trace_meta(split_name: str, tr: dict[str, Any]) -> dict[str, Any]:
    return {
        "split": split_name,
        "source_index": tr["source_index"],
        "attempt": tr["attempt"],
        "truncated": tr["truncated"],
        "judge": tr["judge"],
        "prompt_len": tr["prompt_len"],
        "n_generated": tr["n_generated"],
    }
