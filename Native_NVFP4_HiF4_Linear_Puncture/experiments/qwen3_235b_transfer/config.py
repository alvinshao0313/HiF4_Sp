"""Fixed two-stage transfer recipe; smoke only reduces samples/epochs/layers."""
from pathlib import Path

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.config import E2ETrainConfig, validate_train_config
from Native_NVFP4_HiF4_Linear_Puncture.experiments.non_equivalent_reconstruction.config import Config
from .fetch_model import REVISION


def source_snapshot() -> Path:
    return Path.home() / ".cache/huggingface/hub/models--nvidia--Qwen3-235B-A22B-NVFP4/snapshots" / REVISION


def diag_config(run_root: Path, *, smoke: bool = False) -> E2ETrainConfig:
    cfg = E2ETrainConfig(
        model_path=str(source_snapshot()),
        output_dir=str(run_root / ("smoke/e4" if smoke else "e4")),
        use_r64=True, num_layers=94, end_layer=0 if smoke else 93,
        loss_rollback="on", router_rollback="on",
        diag_epochs=2 if smoke else 20,
        calib_nsamples=4 if smoke else 128,
        calib_val_nsamples=4 if smoke else 32,
        calib_cache_dir=str(run_root / ("smoke/data" if smoke else "data")),
    )
    validate_train_config(cfg)
    return cfg


def group_config(run_root: Path) -> Config:
    cfg = Config(
        model_path=str(source_snapshot()), output_dir=str(run_root / "group_top_mass"),
        init_artifact=str(run_root / "e4/checkpoint/final_model/conversion_state.pt"),
        matrix_sharing="group", router_loss="top_mass", calib_cache_dir=str(run_root / "data"),
    )
    cfg.validate()
    return cfg
