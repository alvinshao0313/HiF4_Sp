from __future__ import annotations

import pytest
import torch

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.config import (
    E2ETrainConfig,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.data.batching import (
    DynamicCalibrationCollator,
    build_length_bucket_batches,
    build_validation_batches,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.data.calibration import (
    CalibrationSample,
    build_s1k_original_sample,
    build_s1k_question_sample,
    require_s1k_fields,
    sample_from_ids_and_mask,
    split_source_ids,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.data import teacher_traces as traces


class FakeTok:
    pad_token_id = 0
    eos_token_id = 1
    chat_template = "fake"

    def __call__(self, text, add_special_tokens=False, return_tensors=None, truncation=False):
        n = max(len(str(text).split()), 1)
        ids = torch.arange(2, 2 + n, dtype=torch.long).unsqueeze(0)
        return {"input_ids": ids}

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, enable_thinking=True, return_tensors="pt"):
        n = 5
        return torch.arange(n, dtype=torch.long).unsqueeze(0)

    def decode(self, ids, skip_special_tokens=True):
        return self._decode_text


class FakeLM(torch.nn.Module):
    def __init__(self, truncated=False, judge="CORRECT"):
        super().__init__()
        self.truncated = truncated
        self.judge = judge
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def generate(self, input_ids, max_new_tokens, do_sample=False, **kwargs):
        b, t = input_ids.shape
        extra = torch.arange(10, 10 + max_new_tokens, dtype=torch.long).view(1, -1).expand(b, -1).clone()
        if max_new_tokens != traces.JUDGE_MAX_NEW_TOKENS and not self.truncated:
            extra[:, -1] = 1
        return torch.cat([input_ids.cpu(), extra], dim=1)


def test_s1k_schema_rejects_missing_fields():
    with pytest.raises(ValueError, match="solution"):
        require_s1k_fields({"question": "q", "text": "t"}, 0)


def test_as_1d_token_ids_accepts_tensor_and_batch_encoding():
    t = traces._as_1d_token_ids(torch.arange(4).view(1, 4))
    assert t.tolist() == [0, 1, 2, 3]
    class BatchEnc(dict):
        def __getattr__(self, item):
            try:
                return self[item]
            except KeyError as e:
                raise AttributeError(item) from e
    be = BatchEnc(input_ids=torch.tensor([[9, 8, 7]]))
    assert traces._as_1d_token_ids(be).tolist() == [9, 8, 7]
    with pytest.raises(TypeError, match="apply_chat_template"):
        traces._as_1d_token_ids("not-ids")


def test_require_qwen3_chat_template_attaches_official(monkeypatch):
    class Tok:
        chat_template = None

    monkeypatch.setattr(traces, "_load_official_qwen3_chat_template", lambda: "OFFICIAL_QWEN3")
    tok = Tok()
    traces.require_qwen3_chat_template(tok)
    assert tok.chat_template == "OFFICIAL_QWEN3"
    traces.require_qwen3_chat_template(tok)
    assert tok.chat_template == "OFFICIAL_QWEN3"


def test_require_qwen3_chat_template_fails_when_source_missing(monkeypatch):
    class Tok:
        chat_template = None

    def boom():
        raise RuntimeError("Qwen/Qwen3-8B tokenizer has no chat_template")

    monkeypatch.setattr(traces, "_load_official_qwen3_chat_template", boom)
    with pytest.raises(RuntimeError, match="no chat_template"):
        traces.require_qwen3_chat_template(Tok())


def test_teacher_cot_mask_prompt_zero_generated_one():
    prompt = torch.arange(5)
    gen = torch.arange(5, 12)
    full = torch.cat([prompt, gen])
    mask = torch.zeros_like(full)
    mask[5:] = 1
    sample = sample_from_ids_and_mask("t0", 0, full, mask, {"source": "s1k_teacher_cot"})
    assert sample.loss_mask.tolist() == [0] * 5 + [1] * 7
    assert sample.input_ids.numel() == 12


def test_original_question_window_masks_all_ones():
    tok = FakeTok()
    row = {"question": "why sky", "solution": "because", "text": "full original text here"}
    orig = build_s1k_original_sample(tok, row, 3)
    q = build_s1k_question_sample(tok, row, 3)
    assert torch.equal(orig.loss_mask, torch.ones_like(orig.input_ids))
    assert torch.equal(q.loss_mask, torch.ones_like(q.input_ids))
    wiki = sample_from_ids_and_mask("w", 0, torch.arange(8), torch.ones(8, dtype=torch.long), {"source": "wikitext2"})
    c4 = sample_from_ids_and_mask("c", 1, torch.arange(6), torch.ones(6, dtype=torch.long), {"source": "c4"})
    assert torch.equal(wiki.loss_mask, torch.ones(8, dtype=torch.long))
    assert torch.equal(c4.loss_mask, torch.ones(6, dtype=torch.long))


def test_fixed_id_split_is_seeded():
    a, b = split_source_ids(20, 4, 2, 42)
    c, d = split_source_ids(20, 4, 2, 42)
    e, f = split_source_ids(20, 4, 2, 43)
    assert a == c and b == d
    assert a != e or b != f
    assert len(set(a) & set(b)) == 0


def test_teacher_all_keeps_incorrect(monkeypatch):
    tok = FakeTok()
    tok._decode_text = "INCORRECT"
    model = FakeLM(truncated=False, judge="INCORRECT")
    ds = [
        {"question": "q0", "solution": "s0", "text": "t0"},
        {"question": "q1", "solution": "s1", "text": "t1"},
    ]
    cfg = E2ETrainConfig.for_test(
        calib_source="s1k_teacher_cot",
        teacher_trace_policy="all",
        teacher_max_new_tokens=7,
        calib_nsamples=1,
        calib_val_nsamples=1,
    )
    samples, meta = traces.generate_split_teacher_traces(
        cfg=cfg,
        tokenizer=tok,
        native_model=model,
        dataset=ds,
        base_ids=[0],
        unused_ids=[1],
        split_name="train",
        split_id=0,
    )
    assert len(samples) == 1
    assert samples[0].meta["judge"] == "INCORRECT"
    assert samples[0].loss_mask[:5].sum() == 0
    assert samples[0].loss_mask[5:].sum() == 7


def test_regenerate_correct_retries_same_question(monkeypatch):
    tok = FakeTok()
    calls = {"n": 0}

    def fake_judge(**kwargs):
        calls["n"] += 1
        return "INCORRECT" if calls["n"] < 2 else "CORRECT"

    monkeypatch.setattr(traces, "judge_teacher_trace", fake_judge)
    model = FakeLM(truncated=False)
    tok._decode_text = "x"
    ds = [{"question": "q0", "solution": "s0", "text": "t0"}]
    cfg = E2ETrainConfig.for_test(
        teacher_trace_policy="regenerate_correct",
        teacher_max_attempts=4,
        teacher_max_new_tokens=7,
    )
    samples, meta = traces.generate_split_teacher_traces(
        cfg=cfg,
        tokenizer=tok,
        native_model=model,
        dataset=ds,
        base_ids=[0],
        unused_ids=[],
        split_name="train",
        split_id=0,
    )
    assert samples[0].meta["judge"] == "CORRECT"
    assert samples[0].meta["attempts"] == 2
    assert calls["n"] == 2


def test_regenerate_correct_fails_after_max_attempts(monkeypatch):
    monkeypatch.setattr(traces, "judge_teacher_trace", lambda **k: "INCORRECT")
    tok = FakeTok()
    tok._decode_text = "x"
    cfg = E2ETrainConfig.for_test(
        teacher_trace_policy="regenerate_correct",
        teacher_max_attempts=4,
        teacher_max_new_tokens=7,
    )
    with pytest.raises(RuntimeError, match="regenerate_correct"):
        traces.generate_split_teacher_traces(
            cfg=cfg,
            tokenizer=tok,
            native_model=FakeLM(),
            dataset=[{"question": "q", "solution": "s", "text": "t"}],
            base_ids=[0],
            unused_ids=[],
            split_name="train",
            split_id=0,
        )


def test_replace_question_correct_uses_unused(monkeypatch):
    tok = FakeTok()
    tok._decode_text = "x"
    judges = {0: "INCORRECT", 2: "CORRECT"}
    monkeypatch.setattr(
        traces,
        "judge_teacher_trace",
        lambda question, **k: judges[{"q0": 0, "q2": 2}[question.split()[0].replace("q", "q") if False else {"q0": 0, "q2": 2}[question]]],
    )

    def judge(*, question, **kwargs):
        return {"q0": "INCORRECT", "q2": "CORRECT"}[question]

    monkeypatch.setattr(traces, "judge_teacher_trace", judge)
    ds = [
        {"question": "q0", "solution": "s0", "text": "t0"},
        {"question": "q1", "solution": "s1", "text": "t1"},
        {"question": "q2", "solution": "s2", "text": "t2"},
    ]
    cfg = E2ETrainConfig.for_test(
        teacher_trace_policy="replace_question_correct",
        teacher_max_new_tokens=7,
    )
    samples, meta = traces.generate_split_teacher_traces(
        cfg=cfg,
        tokenizer=tok,
        native_model=FakeLM(),
        dataset=ds,
        base_ids=[0],
        unused_ids=[2],
        split_name="train",
        split_id=0,
    )
    assert samples[0].source_index == 2
    assert samples[0].meta["judge"] == "CORRECT"


def test_unknown_judge_fails(monkeypatch):
    tok = FakeTok()
    tok._decode_text = "MAYBE"
    with pytest.raises(RuntimeError, match="unknown judge"):
        traces.judge_teacher_trace(
            tokenizer=tok,
            native_model=FakeLM(),
            question="q",
            solution="s",
            candidate="c",
        )


def test_dynamic_padding_masks_pad():
    s1 = sample_from_ids_and_mask("a", 0, torch.arange(3), torch.ones(3, dtype=torch.long), {})
    s2 = sample_from_ids_and_mask("b", 1, torch.arange(5), torch.ones(5, dtype=torch.long), {})
    packed = DynamicCalibrationCollator(pad_token_id=9)([s1, s2])
    assert packed["input_ids"].shape == (2, 5)
    assert packed["input_ids"][0, 3:].tolist() == [9, 9]
    assert packed["attention_mask"][0].tolist() == [1, 1, 1, 0, 0]
    assert packed["loss_mask"][0].tolist() == [1, 1, 1, 0, 0]


def test_length_bucketing_and_val_sort_only():
    samples = [
        sample_from_ids_and_mask(str(i), i, torch.arange(i + 1), torch.ones(i + 1, dtype=torch.long), {})
        for i in range(10)
    ]
    batches = build_length_bucket_batches(samples, batch_size=2, seed=0)
    assert sum(len(b) for b in batches) == 10
    val = build_validation_batches(samples, 3)
    lengths = [int(b[0].input_ids.numel()) for b in val]
    assert lengths == sorted(lengths)
    other = build_length_bucket_batches(samples, batch_size=2, seed=0)
    assert [[s.sample_id for s in b] for b in batches] == [[s.sample_id for s in b] for b in other]


def test_shared_calib_cache_reuses_without_teacher_generation(tmp_path, monkeypatch):
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.data import calibration as cal

    class DS:
        def __len__(self):
            return 20

        def __getitem__(self, i):
            return {"question": f"q{i}", "solution": f"s{i}", "text": f"t{i}"}

    gen_calls = {"n": 0}

    def fake_gen(*, cfg, tokenizer, native_model, dataset, base_ids, unused_ids, split_name, split_id):
        gen_calls["n"] += 1
        samples = []
        metas = []
        for src in base_ids:
            ids = torch.arange(src + 3, dtype=torch.long)
            mask = torch.zeros_like(ids)
            mask[-2:] = 1
            samples.append(sample_from_ids_and_mask(f"s1k_teacher_cot_{src}", src, ids, mask, {"source": "s1k_teacher_cot"}))
            metas.append({"split": split_name, "source_index": src})
        return samples, metas

    monkeypatch.setattr(cal, "load_s1k_dataset", lambda: DS())
    monkeypatch.setattr(cal, "generate_split_teacher_traces", fake_gen)
    cache = tmp_path / "shared"
    cfg = E2ETrainConfig.for_test(
        calib_source="s1k_teacher_cot",
        calib_nsamples=2,
        calib_val_nsamples=1,
        teacher_trace_policy="all",
    )
    train1, val1 = cal.build_or_load_calibration(cfg, FakeTok(), FakeLM(), cache)
    assert gen_calls["n"] == 2
    train2, val2 = cal.build_or_load_calibration(cfg, FakeTok(), FakeLM(), cache)
    assert gen_calls["n"] == 2
    assert [s.sample_id for s in train1] == [s.sample_id for s in train2]
    assert torch.equal(train1[0].input_ids, train2[0].input_ids)
    assert torch.equal(train1[0].loss_mask, train2[0].loss_mask)
    assert torch.equal(val1[0].input_ids, val2[0].input_ids)


def test_nearest_rank_percentile():
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.data.calibration import (
        nearest_rank_percentile,
    )

    assert nearest_rank_percentile([10, 20, 30, 40], 0.50) == 20
    assert nearest_rank_percentile([10, 20, 30, 40], 0.95) == 40
    with pytest.raises(ValueError, match="empty"):
        nearest_rank_percentile([], 0.50)


def test_s1k_original_builds_with_native_model_none(tmp_path, monkeypatch):
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.data import calibration as cal

    class DS:
        def __len__(self):
            return 20

        def __getitem__(self, i):
            return {"question": f"q{i}", "solution": f"s{i}", "text": f"text for {i} extra tokens"}

    monkeypatch.setattr(cal, "load_s1k_dataset", lambda: DS())

    def boom(**kwargs):
        raise AssertionError("teacher generation must not run")

    monkeypatch.setattr(cal, "generate_split_teacher_traces", boom)
    cache = tmp_path / "shared_original"
    cfg = E2ETrainConfig.for_test(
        calib_source="s1k_original",
        calib_nsamples=2,
        calib_val_nsamples=1,
    )
    train1, val1 = cal.build_or_load_calibration(cfg, FakeTok(), None, cache)
    train2, val2 = cal.build_or_load_calibration(cfg, FakeTok(), None, cache)
    assert [s.sample_id for s in train1] == [s.sample_id for s in train2]
    assert torch.equal(train1[0].input_ids, train2[0].input_ids)
    assert torch.equal(train1[0].loss_mask, train2[0].loss_mask)
    assert torch.equal(val1[0].input_ids, val2[0].input_ids)
    man = (cache / "calibration" / "train_manifest.json").read_text(encoding="utf-8")
    assert "teacher_trace_policy" not in man


def test_prepare_cli_rejects_teacher_cot():
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.cli import (
        prepare_calibration as prep,
    )

    with pytest.raises(ValueError, match="s1k_teacher_cot"):
        prep.require_prepare_source("s1k_teacher_cot")
    with pytest.raises(SystemExit):
        prep.main(["--calib_cache_dir", "/tmp/x", "--calib_source", "s1k_teacher_cot"])
