"""Stable public API for HiF4 MLP hierarchical permutation search.

Future main.py hook (not wired yet): after from_pretrained, before
hif4_rtn_quant / gptq_fwrd, call ``reorder_model_mlps`` or
``apply_permutations_from_file``.
"""

from .config import LayerSearchResult, MLPLayerSpec, SearchConfig
from .hierarchical_greedy import optimize_layer_permutation
from .model_permutation import (
    apply_mlp_permutation_,
    apply_permutations_from_dict,
    apply_permutations_from_file,
    discover_swiglu_mlps,
    get_mlp_modules,
    validate_permutation,
)
from .pipeline import reorder_model_mlps

__all__ = [
    "SearchConfig",
    "MLPLayerSpec",
    "LayerSearchResult",
    "discover_swiglu_mlps",
    "get_mlp_modules",
    "validate_permutation",
    "apply_mlp_permutation_",
    "apply_permutations_from_dict",
    "apply_permutations_from_file",
    "optimize_layer_permutation",
    "reorder_model_mlps",
]
