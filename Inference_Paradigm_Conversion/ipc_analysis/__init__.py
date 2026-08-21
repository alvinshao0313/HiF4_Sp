"""IPC analysis library: format adapters, metrics, capture, and reporting."""

SCHEMA_VERSION = 1
DEFAULT_SEED = 20260810
ANALYSIS_SEED = 20260809

PATH_IDS = frozenset(
    {
        "P1_semantic",
        "P1_runtime",
        "P2_matched_semantic",
        "P2_matched_runtime",
        "P2_deployment_semantic",
        "P2_deployment_runtime",
        "W_storage_probe",
    }
)

SOURCE_SEMANTICS_ALLOWED = frozenset({"nvfp4_qat_fake_dequant_bf16"})
