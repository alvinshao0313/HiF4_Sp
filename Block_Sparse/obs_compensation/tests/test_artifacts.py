from __future__ import annotations

import json

import pytest
import torch

from obs_compensation.artifacts import (
    load_source_artifacts,
    validate_source_artifacts_against_targets,
)
from obs_compensation.tests.helpers import (
    TinyCausalLM,
    make_block_masks_for_targets,
    make_descending_permutation_payload,
    make_targets_from_tiny,
    write_source_artifacts,
)


def test_load_valid_none_permutation(tmp_path):
    model = TinyCausalLM()
    targets = make_targets_from_tiny(model)
    masks = make_block_masks_for_targets(targets, 2, 2)
    root = write_source_artifacts(tmp_path / "src", masks=masks, mlp_permutation="none")
    arts = load_source_artifacts(root)
    assert arts.permutation_payload is None
    assert arts.metadata.mlp_permutation == "none"
    validate_source_artifacts_against_targets(arts, targets)


def test_load_valid_wanda_shared(tmp_path):
    model = TinyCausalLM(d_ff=8)
    targets = make_targets_from_tiny(model)
    masks = make_block_masks_for_targets(targets, 2, 2)
    payload = make_descending_permutation_payload(targets, intermediate_size=8)
    root = write_source_artifacts(
        tmp_path / "src",
        masks=masks,
        mlp_permutation="wanda_shared",
        permutation_payload=payload,
    )
    arts = load_source_artifacts(root)
    assert arts.permutation_payload is not None
    validate_source_artifacts_against_targets(arts, targets)


def test_missing_summary(tmp_path):
    root = tmp_path / "src"
    root.mkdir()
    torch.save({}, root / "block_masks.pt")
    with pytest.raises(FileNotFoundError, match="pruning_summary"):
        load_source_artifacts(root)


def test_missing_mask_file(tmp_path):
    root = tmp_path / "src"
    root.mkdir()
    (root / "pruning_summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="block_masks"):
        load_source_artifacts(root)


def test_mask_not_dict(tmp_path):
    root = write_source_artifacts(tmp_path / "src", masks={})
    torch.save([torch.ones(1, 1, dtype=torch.bool)], root / "block_masks.pt")
    # rewrite summary for non-empty expectation path - empty masks fail earlier
    summary = json.loads((root / "pruning_summary.json").read_text())
    (root / "pruning_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(TypeError, match="dict"):
        load_source_artifacts(root)


def test_mask_non_bool(tmp_path):
    masks = {"a": torch.tensor([[True, False], [True, True]])}
    root = write_source_artifacts(tmp_path / "src", masks=masks)
    torch.save({"a": torch.ones(2, 2)}, root / "block_masks.pt")
    with pytest.raises(TypeError, match="bool"):
        load_source_artifacts(root)


def test_zero_kept_blocks(tmp_path):
    masks = {"a": torch.zeros(2, 2, dtype=torch.bool)}
    root = write_source_artifacts(tmp_path / "src", masks=masks)
    with pytest.raises(ValueError, match="zero kept"):
        load_source_artifacts(root)


def test_all_false_block_row_is_allowed(tmp_path):
    mask = torch.tensor(
        [
            [False, False],
            [True, False],
        ],
        dtype=torch.bool,
    )
    root = write_source_artifacts(tmp_path / "src", masks={"a": mask})
    artifacts = load_source_artifacts(root)
    assert torch.equal(artifacts.masks["a"], mask)
    assert bool(artifacts.masks["a"].any().item())


def test_reject_rounds_and_residual(tmp_path):
    masks = {"a": torch.ones(2, 2, dtype=torch.bool)}
    root = write_source_artifacts(
        tmp_path / "src", masks=masks, num_pruning_rounds=2
    )
    with pytest.raises(ValueError, match="num_pruning_rounds"):
        load_source_artifacts(root)
    root2 = write_source_artifacts(
        tmp_path / "src2", masks=masks, residual_permutation="block_loss"
    )
    with pytest.raises(ValueError, match="residual_permutation"):
        load_source_artifacts(root2)


def test_unsupported_mlp_permutation(tmp_path):
    masks = {"a": torch.ones(2, 2, dtype=torch.bool)}
    root = write_source_artifacts(
        tmp_path / "src",
        masks=masks,
        mlp_permutation="weird",
        summary_overrides={"mlp_permutation": "weird"},
    )
    with pytest.raises(ValueError, match="Unsupported mlp_permutation"):
        load_source_artifacts(root)


def test_wanda_missing_permutation_file(tmp_path):
    masks = {"a": torch.tensor([[True, False], [True, True]])}
    root = write_source_artifacts(
        tmp_path / "src", masks=masks, mlp_permutation="wanda_shared"
    )
    with pytest.raises(FileNotFoundError, match="mlp_permutations"):
        load_source_artifacts(root)


def test_permutation_payload_validation(tmp_path):
    model = TinyCausalLM(d_ff=8)
    targets = make_targets_from_tiny(model)
    masks = make_block_masks_for_targets(targets, 2, 2)
    payload = make_descending_permutation_payload(targets, 8)
    # missing field
    bad = {k: dict(v) for k, v in payload.items()}
    del bad["0"]["combined_score"]
    root = write_source_artifacts(
        tmp_path / "src",
        masks=masks,
        mlp_permutation="wanda_shared",
        permutation_payload=bad,
    )
    with pytest.raises(ValueError, match="missing permutation fields"):
        load_source_artifacts(root)

    # non-bijective
    bad2 = {k: dict(v) for k, v in payload.items()}
    bad2["0"]["permutation"] = torch.zeros(8, dtype=torch.int64)
    root2 = write_source_artifacts(
        tmp_path / "src2",
        masks=masks,
        mlp_permutation="wanda_shared",
        permutation_payload=bad2,
    )
    with pytest.raises(ValueError, match="bijective"):
        load_source_artifacts(root2)

    # ascending importance after reorder
    bad3 = {k: dict(v) for k, v in payload.items()}
    bad3["0"]["combined_score"] = torch.arange(8, 0, -1, dtype=torch.float32)
    root3 = write_source_artifacts(
        tmp_path / "src3",
        masks=masks,
        mlp_permutation="wanda_shared",
        permutation_payload=bad3,
    )
    with pytest.raises(ValueError, match="descending"):
        load_source_artifacts(root3)


def test_sparsity_mismatch(tmp_path):
    masks = {"a": torch.tensor([[True, False], [True, True]])}
    root = write_source_artifacts(
        tmp_path / "src",
        masks=masks,
        summary_overrides={"actual_block_sparsity": 0.9},
    )
    with pytest.raises(ValueError, match="does not match"):
        load_source_artifacts(root)
