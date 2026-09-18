from pathlib import Path
from types import SimpleNamespace

from Native_NVFP4_HiF4_Linear_Puncture.experiments.kl_direction_reuse.scripts.smoke_pipeline import (
    smoke_tasks,
)


def test_smoke_has_four_train_jobs_and_only_intra_phase_dependencies(tmp_path):
    train_ids = [f"train{i}" for i in range(4)]
    test_ids = ["test_short", "test_long"]
    samples = {sid: {"input_ids": list(range(i + 1))} for i, sid in enumerate(train_ids + test_ids)}
    ds = SimpleNamespace(
        root=Path(tmp_path),
        samples=samples,
        protocol={"batches": [train_ids], "splits": {"train": train_ids, "test": test_ids}},
        ids=lambda split: train_ids if split == "train" else test_ids,
    )
    config = {"train_sample_ids": train_ids, "eval_sample_ids": test_ids}
    tasks = smoke_tasks(tmp_path, ds, config)
    by_key = {task.key: task for group in tasks.values() for task in group}
    assert len(tasks['train']) == 5
    assert {t.key for t in tasks['train']} == {'L08_mse', 'L08_direct_kl', 'L24_mse', 'L24_cached_kl_k1', 'L24_direct_kl'}
    for group in tasks.values():
        keys = {t.key for t in group}
        assert all(set(t.dependencies) <= keys for t in group)
    for recipe in ("L24_mse", "L24_cached_kl_k1", "L24_direct_kl"):
        assert by_key[recipe].pool == "train" and by_key[recipe].slots == 1
        assert by_key[recipe].dependencies == ()
        export = by_key[f"{recipe}_final_export"]
        evaluate = by_key[f"{recipe}_final_evaluate"]
        assert export.dependencies == () and export.pool == "cpu"
        assert evaluate.dependencies == (export.key,) and evaluate.pool == "eval" and evaluate.slots == 2
