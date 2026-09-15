"""Frozen protocol for the internal error accumulation mechanism study."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = (
    REPO_ROOT
    / "Native_NVFP4_HiF4_Linear_Puncture"
    / "results"
    / "internal_error_accumulation"
    / "qwen3_30b_a3b_e0_e1_formal"
)
RUNS_ROOT = RESULTS_ROOT / "runs"
ANALYSIS_ROOT = RESULTS_ROOT / "analysis"

DEFAULT_MODEL_PATH = "nvidia/Qwen3-30B-A3B-NVFP4"
DEFAULT_PHASEA_ROOT = (
    REPO_ROOT
    / "Native_NVFP4_HiF4_Linear_Puncture"
    / "results"
    / "e2e_diag_reconstruction"
    / "phaseA_refactor_20260825T035730Z"
)
SHARED_CALIB_ROOT = (
    REPO_ROOT
    / "Native_NVFP4_HiF4_Linear_Puncture"
    / "results"
    / "e2e_diag_reconstruction"
    / "shared_calibration"
)
S1K_SHARED_CALIB = SHARED_CALIB_ROOT / "s1k_original_n128_v32_seed42"
WIKITEXT2_SHARED_CALIB = SHARED_CALIB_ROOT / "wikitext2_n128_v32_seed42_len1024"

NUM_LAYERS = 48
TP_SIZE = 2
PREFIX_LENGTHS_J = (8, 32, 64, 128)
MIN_CALIBRATION_LENGTH = 129  # must cover max j=128 predictor
COHORT_SEED = 20260909
DISCOVERY_PER_SOURCE = 4
HOLDOUT_PER_SOURCE = 4
OBJECTIVE_SOURCE_RATIO = (0.5, 0.5)  # WikiText2 : S1K
OBJECTIVE_TRAIN_SEED = 20260909

STAGE_ORDER = (
    "S0_INFRA",
    "S1_CAPTURE_STRUCTURAL",
    "S2_LAYER_CAUSAL",
    "S3_SUBSTRUCTURE_CAUSAL",
    "S3_FULL48_CAUSAL",
    "S4_PROTECTION_OBJECTIVE",
    "S5_TOPK_STRUCTURAL_VALIDATE",
    "S6_E2E_VALIDATE",
    "S7_REPORT",
)

RUN_SUBDIRS = (
    "logs",
    "00_protocol",
    "10_capture",
    "20_structural",
    "30_moe_qp",
    "40_router",
    "50_protection",
    "60_objective",
    "70_variant_validation",
    "analysis",
    "tmp",
)

REPLICA_BOUNDARIES = (
    ("input_norm", "updated_residual"),
    ("post_attn_norm", "updated_residual"),
    ("o_proj", "tp_reduced"),
    ("router_logits", "logits"),
    ("moe_out", "tp_reduced"),
    ("final_norm", "updated_residual"),
    ("final_norm", "normalized"),
)

STATUS_RUNNING = "RUNNING"
STATUS_WAITING_REVIEW = "WAITING_REVIEW"
STATUS_FAILED = "FAILED"
STATUS_COMPLETED = "COMPLETED"
