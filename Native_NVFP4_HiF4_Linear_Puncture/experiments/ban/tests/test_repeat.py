from pathlib import Path

import pytest

from ..repeat import pool_scores, repeat_command


def test_pool_preserves_original_and_every_round():
    rows = [{"round": 3, "score": .6}, {"round": 1, "score": .4}, {"round": 2, "score": .8}]
    report = pool_scores(rows)
    assert [row["round"] for row in report["rounds"]] == [1, 2, 3]
    assert report["mean_score"] == pytest.approx(.6)
    assert report["min_score"] == .4 and report["max_score"] == .8
    with pytest.raises(ValueError, match="unique"):
        pool_scores([rows[0], rows[0]])


def test_repeat_only_changes_cache_and_output_locations():
    original = ["python", "-m", "ban.additional_main", "--dataset_manifest", "/old/data.json",
                "--output_dir", "/old/output", "--ban_stats_dir", "/old/routing",
                "--ban_lambda", "0.7", "--ban_k_min", "5", "--max_new_tokens", "38912"]
    command = repeat_command(original, Path("/new/round_02/aime25"), Path("/new/data.json"))
    assert command[command.index("--output_dir") + 1] == "/new/round_02/aime25/evaluation"
    assert command[command.index("--ban_stats_dir") + 1] == "/new/round_02/aime25/routing"
    assert command[-6:] == original[-6:]
    assert original[original.index("--output_dir") + 1] == "/old/output"
    with pytest.raises(ValueError, match="full"):
        repeat_command([*original, "--max_samples", "1"], Path("/new"), Path("/new/data.json"))
