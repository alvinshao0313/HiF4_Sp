"""Metric helpers: NMSE / SQNR / recovery."""

from __future__ import annotations

import math

import torch

from Native_NVFP4_HiF4_Linear_Puncture.src.metrics import (
    aggregate_global_nmse,
    compute_nmse,
    compute_recovery,
    compute_sqnr_db,
)


def test_nmse_basic_correctness():
    y_ref = torch.tensor([1.0, 2.0, 3.0])
    y_hat = torch.tensor([1.0, 2.0, 4.0])
    nmse = compute_nmse(y_hat, y_ref)
    assert abs(float(nmse) - (1.0 / 14.0)) < 1e-12


def test_sqnr_basic_correctness():
    nmse = 1e-2
    sqnr = compute_sqnr_db(nmse)
    assert abs(float(sqnr) - (-10.0 * math.log10(nmse))) < 1e-12


def test_recovery_basic_and_nan_on_zero_direct_error():
    r = compute_recovery(direct_error_energy=4.0, improved_error_energy=1.0)
    assert abs(float(r) - 0.75) < 1e-12

    r0 = compute_recovery(direct_error_energy=0.0, improved_error_energy=0.0)
    assert math.isnan(float(r0))


def test_aggregate_global_nmse_is_energy_weighted():
    # module A: err=1, ref=1 -> nmse=1
    # module B: err=1, ref=100 -> nmse=0.01
    # mean of nmse = 0.505; energy aggregate = 2/101
    global_nmse = aggregate_global_nmse(
        error_energies=[1.0, 1.0],
        reference_energies=[1.0, 100.0],
    )
    assert abs(float(global_nmse) - (2.0 / 101.0)) < 1e-12
    assert abs(float(global_nmse) - 0.505) > 0.1
