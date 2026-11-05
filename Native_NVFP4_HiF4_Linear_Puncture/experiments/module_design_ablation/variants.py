"""The confirmed 15-model matrix; optimizer budgets are configured separately."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Variant:
    name: str
    transform: str
    train_norm: bool = False
    train_router: bool = False
    kl_weight: float = 0.0

    @property
    def is_reference(self) -> bool:
        return self.transform in {"native", "historical"}

    @property
    def needs_training(self) -> bool:
        return self.train_norm or self.transform in {"diag", "rotation", "matrix"}


VARIANTS = {variant.name: variant for variant in (
    Variant("native_nvfp4", "native"),
    Variant("direct_hif4", "direct"),
    Variant("fixed_r64", "fixed"),
    Variant("norm", "fixed", train_norm=True),
    Variant("norm_kl", "fixed", train_norm=True, kl_weight=1.0),
    Variant("diag", "diag"),
    Variant("rotation", "rotation"),
    Variant("rotation_norm", "rotation", train_norm=True),
    Variant("rotation_norm_kl", "rotation", train_norm=True, kl_weight=1.0),
    Variant("matrix", "matrix"),
    Variant("matrix_norm", "matrix", train_norm=True),
    Variant("matrix_norm_kl", "matrix", train_norm=True, kl_weight=1.0),
    Variant("current", "matrix", train_norm=True, train_router=True),
    Variant("current_kl", "matrix", train_norm=True, train_router=True, kl_weight=1.0),
    Variant("existing_two_stage", "historical"),
)}


def get_variant(name: str) -> Variant:
    try:
        return VARIANTS[name]
    except KeyError:
        raise ValueError(f"unknown variant {name!r}; expected one of {tuple(VARIANTS)}") from None
