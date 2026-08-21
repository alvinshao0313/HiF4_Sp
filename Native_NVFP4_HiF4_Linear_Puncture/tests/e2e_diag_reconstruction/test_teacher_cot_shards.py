from __future__ import annotations

import pytest
import torch

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.config import (
    E2ETrainConfig,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.data.calibration import (
    sample_from_ids_and_mask,
    split_source_ids,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.data import (
    teacher_cot_shards as shards,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.cli import (
    build_teacher_cot_cache as cli,
)


def _fake_sample(source_index: int) -> object:
    prompt = torch.arange(3, dtype=torch.long)
    gen = torch.arange(3, 3 + source_index % 5 + 1, dtype=torch.long)
    full = torch.cat([prompt, gen])
    mask = torch.zeros_like(full)
    mask[3:] = 1
    return sample_from_ids_and_mask(
        f"s1k_teacher_cot_{source_index}",
        source_index,
        full,
        mask,
        {
            "source": "s1k_teacher_cot",
            "truncated": False,
            "judge": "CORRECT",
            "attempts": 1,
            "question_source_index": source_index,
        },
    )


def _fake_trace(split_name: str, source_index: int) -> dict:
    return {
        "split": split_name,
        "source_index": source_index,
        "attempt": 0,
        "truncated": False,
        "judge": "CORRECT",
        "prompt_len": 3,
        "n_generated": source_index % 5 + 1,
    }


def test_shard_source_ids_partition_is_disjoint_union():
    train_ids, val_ids = split_source_ids(200, 128, 32, 42)
    train_parts = shards.assert_shards_partition(train_ids, 4)
    val_parts = shards.assert_shards_partition(val_ids, 4)
    assert sum(len(p) for p in train_parts) == 128
    assert sum(len(p) for p in val_parts) == 32
    assert train_parts[0] == [train_ids[i] for i in range(0, 128, 4)]
    assert train_parts[1] == [train_ids[i] for i in range(1, 128, 4)]
    assert val_parts[0] == [val_ids[i] for i in range(0, 32, 4)]


def test_require_all_policy_rejects_other_policies():
    cfg = E2ETrainConfig.for_test(teacher_trace_policy="regenerate_correct")
    with pytest.raises(ValueError, match="teacher_trace_policy=all"):
        shards.require_all_policy(cfg)
    cfg2 = E2ETrainConfig.for_test(calib_source="s1k_original", teacher_trace_policy="all")
    with pytest.raises(ValueError, match="s1k_teacher_cot"):
        shards.require_all_policy(cfg2)


def test_worker_cli_rejects_non_all_policy(tmp_path):
    with pytest.raises(ValueError, match="teacher_trace_policy=all"):
        cli.main(
            [
                "worker",
                "--calib_cache_dir",
                str(tmp_path / "cache"),
                "--output_shard_dir",
                str(tmp_path / "shard_0"),
                "--shard_id",
                "0",
                "--num_shards",
                "4",
                "--teacher_trace_policy",
                "regenerate_correct",
            ]
        )


def test_merge_restores_original_order_and_masks(tmp_path):
    train_ids = [10, 20, 30, 40, 50, 60, 70, 80]
    val_ids = [11, 21, 31, 41]
    num_shards = 4
    shards_root = tmp_path / "shards"
    for k in range(num_shards):
        out = shards_root / f"shard_{k}"
        t_ids = shards.shard_source_ids(train_ids, num_shards, k)
        v_ids = shards.shard_source_ids(val_ids, num_shards, k)
        shards.write_split_shard(
            output_shard_dir=out,
            split_name="train",
            split_id=0,
            shard_id=k,
            num_shards=num_shards,
            base_ids=t_ids,
            samples=[_fake_sample(i) for i in t_ids],
            traces=[_fake_trace("train", i) for i in t_ids],
        )
        shards.write_split_shard(
            output_shard_dir=out,
            split_name="val",
            split_id=1,
            shard_id=k,
            num_shards=num_shards,
            base_ids=v_ids,
            samples=[_fake_sample(i) for i in v_ids],
            traces=[_fake_trace("val", i) for i in v_ids],
        )

    cache = tmp_path / "cache"
    cfg = E2ETrainConfig.for_test(
        calib_source="s1k_teacher_cot",
        teacher_trace_policy="all",
        calib_nsamples=len(train_ids),
        calib_val_nsamples=len(val_ids),
    )
    shards.merge_teacher_cot_shards(
        cfg=cfg,
        calib_cache_dir=cache,
        shard_dirs=[shards_root / f"shard_{k}" for k in range(num_shards)],
        train_ids=train_ids,
        val_ids=val_ids,
    )

    from Native_NVFP4_HiF4_Linear_Puncture.src.io_utils import load_pt

    train = load_pt(cache / "calibration" / "train.pt", map_location="cpu")
    val = load_pt(cache / "calibration" / "val.pt", map_location="cpu")
    assert [s.source_index for s in train] == train_ids
    assert [s.source_index for s in val] == val_ids
    for s in train + val:
        assert int(s.loss_mask[:3].sum().item()) == 0
        assert int(s.loss_mask[3:].sum().item()) == int(s.input_ids.numel()) - 3
    assert (cache / "calibration" / "train_manifest.json").is_file()
    assert (cache / "calibration" / "val_manifest.json").is_file()
    assert (cache / "calibration" / "teacher_traces" / "train.jsonl").is_file()
    assert (cache / "calibration" / "teacher_traces" / "val.jsonl").is_file()


def test_merge_fails_on_missing_sample(tmp_path):
    train_ids = [1, 2, 3, 4]
    val_ids = [5, 6, 7, 8]
    num_shards = 2
    shards_root = tmp_path / "shards"
    for k in range(num_shards):
        out = shards_root / f"shard_{k}"
        t_ids = shards.shard_source_ids(train_ids, num_shards, k)
        v_ids = shards.shard_source_ids(val_ids, num_shards, k)
        shards.write_split_shard(
            output_shard_dir=out,
            split_name="train",
            split_id=0,
            shard_id=k,
            num_shards=num_shards,
            base_ids=t_ids,
            samples=[_fake_sample(i) for i in t_ids],
            traces=[_fake_trace("train", i) for i in t_ids],
        )
        shards.write_split_shard(
            output_shard_dir=out,
            split_name="val",
            split_id=1,
            shard_id=k,
            num_shards=num_shards,
            base_ids=v_ids,
            samples=[_fake_sample(i) for i in v_ids],
            traces=[_fake_trace("val", i) for i in v_ids],
        )
    # Drop one train sample after writing so meta still claims the full shard.
    bad = shards_root / "shard_0" / "train" / "samples.pt"
    from Native_NVFP4_HiF4_Linear_Puncture.src.io_utils import load_pt, save_pt

    samples = load_pt(bad, map_location="cpu")
    save_pt(bad, samples[:-1])

    cfg = E2ETrainConfig.for_test(
        calib_source="s1k_teacher_cot",
        teacher_trace_policy="all",
        calib_nsamples=4,
        calib_val_nsamples=4,
    )
    with pytest.raises(RuntimeError, match="samples"):
        shards.merge_teacher_cot_shards(
            cfg=cfg,
            calib_cache_dir=tmp_path / "cache",
            shard_dirs=[shards_root / f"shard_{k}" for k in range(num_shards)],
            train_ids=train_ids,
            val_ids=val_ids,
        )
