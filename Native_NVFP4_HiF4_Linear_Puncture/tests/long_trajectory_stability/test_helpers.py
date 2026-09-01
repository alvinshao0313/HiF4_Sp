from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path

import pytest
import torch

from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.build_probe_plan import (
    evenly_spaced,
    select_samples,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.compare_free_runs import (
    first_divergence,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.metrics import (
    hidden_metrics,
    logit_metrics,
    router_metrics,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.capture_main import (
    coerce_unrestricted_topk,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.launch_capture import (
    CHAT_TEMPLATE_NAME,
    ensure_chat_template_model_path,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.run_semantic_matrix import (
    require_existing_e0_parity_pass,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.trajectory_io import (
    load_detail_trajectories,
    prompt_key,
)


def test_failed_e0_parity_file_is_not_skippable_success(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    assert require_existing_e0_parity_pass(missing, 0.99) is False
    ok = tmp_path / "ok.json"
    ok.write_text('{"top1_parity": 0.995}', encoding="utf-8")
    assert require_existing_e0_parity_pass(ok, 0.99) is True
    bad = tmp_path / "bad.json"
    bad.write_text('{"top1_parity": 0.8571}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="do not continue E1-E4"):
        require_existing_e0_parity_pass(bad, 0.99)


def test_chat_template_view_is_local_and_does_not_rewrite_source(tmp_path: Path) -> None:
    source = tmp_path / "sidecar"
    source.mkdir()
    (source / "config.json").write_text("{}", encoding="utf-8")
    native_template = tmp_path / "native" / CHAT_TEMPLATE_NAME
    native_template.parent.mkdir()
    native_template.write_text("template", encoding="utf-8")
    output_dir = tmp_path / "capture"
    view = ensure_chat_template_model_path(source, native_template, output_dir)
    assert view == (output_dir / "model_view").resolve()
    assert (view / CHAT_TEMPLATE_NAME).is_file()
    assert (view / "config.json").is_symlink()
    assert not (source / CHAT_TEMPLATE_NAME).exists()
    already = tmp_path / "native_ok"
    already.mkdir()
    (already / CHAT_TEMPLATE_NAME).write_text("template", encoding="utf-8")
    assert ensure_chat_template_model_path(already, native_template, output_dir) == already.resolve()


def test_unrestricted_topk_maps_negative_sentinel_to_none() -> None:
    assert coerce_unrestricted_topk(-1) is None
    assert coerce_unrestricted_topk(0) == 0
    assert coerce_unrestricted_topk(20) == 20
    assert coerce_unrestricted_topk(None) is None


def test_first_divergence_exact_and_prefix() -> None:
    assert first_divergence([1, 2, 3], [1, 2, 3]) is None
    assert first_divergence([1, 2, 3], [1, 9, 3]) == 1
    assert first_divergence([1, 2], [1, 2, 3]) == 2


def test_evenly_spaced_stays_inside_half_open_interval() -> None:
    values = evenly_spaced(128, 512, 4)
    assert values == sorted(set(values))
    assert len(values) == 4
    assert all(128 <= x < 512 for x in values)


def test_select_samples_includes_longest_and_length_spread() -> None:
    rows = [
        {"prompt_key": f"p{i}", "output_len": i + 1}
        for i in range(20)
    ]
    selected = select_samples(rows, 8)
    keys = {x["prompt_key"] for x in selected}
    assert {"p16", "p17", "p18", "p19"}.issubset(keys)
    assert len(selected) == 8


def test_metrics_identity_is_zero_error() -> None:
    hidden = torch.tensor([0.5, -1.0, 2.0])
    h = hidden_metrics(hidden, hidden.clone())
    assert h["hidden_rel_l2"] == 0.0
    assert abs(h["hidden_cosine"] - 1.0) < 1e-6

    router = torch.tensor([0.1, 0.8, 0.3, -0.2])
    r = router_metrics(router, router.clone(), top_k=2)
    assert abs(float(r["router_kl_e0_to_variant"])) < 1e-7
    assert r["router_topk_overlap"] == 1.0
    assert r["router_topk_exact"] is True

    logits = torch.tensor([0.2, 1.4, -0.5, 0.1])
    l = logit_metrics(logits, logits.clone(), target_token_id=1)
    assert abs(float(l["logit_kl_e0_to_variant"])) < 1e-7
    assert abs(float(l["logit_js"])) < 1e-7
    assert l["top1_agreement"] is True
    assert l["e0_target_rank"] == 1
    assert l["variant_target_rank"] == 1


def test_all_experiment_python_files_parse() -> None:
    exp_dir = Path(__file__).resolve().parents[2] / "experiments/long_trajectory_stability"
    for path in sorted(exp_dir.glob("*.py")):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_causal_replay_module_imports() -> None:
    pytest.importorskip("transformers")
    importlib.import_module(
        "Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.causal_replay"
    )


def test_capture_loader_keeps_exact_ids(tmp_path: Path) -> None:
    detail = tmp_path / "details_mmlu_pro|0_test.json"
    payload = [
        {
            "doc": {"id": "row0", "specific": {"category": "math"}},
            "metric": {"extractive_match": 1.0},
            "gold": ["A"],
            "model_response": {
                "input_tokens": [10, 11, 12],
                "output_tokens": [[20, 21, 22]],
                "text": ["raw reasoning"],
            },
        }
    ]
    detail.write_text(json.dumps(payload), encoding="utf-8")
    rows = load_detail_trajectories(detail)
    assert len(rows) == 1
    assert rows[0].input_ids == [10, 11, 12]
    assert rows[0].output_ids == [20, 21, 22]
    assert rows[0].prompt_key == prompt_key([10, 11, 12])
