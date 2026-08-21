from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path('/home/shaoyuantian/program/HiF4_Sp')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Inference_Paradigm_Conversion.ipc_analysis.analysis.activation_grid_occupancy import (
    build_theoretical_grid_json,
    enumerate_hif4_full_internal_grid,
    enumerate_nvfp4_full_internal_grid,
)

OUT = ROOT / 'Inference_Paradigm_Conversion/results/20260811T_ax_final_consolidated'


def test_recomputed_ax3_exactly_matches_saved_core():
    saved = json.loads((OUT / 'ax3_theoretical_grid.json').read_text())
    now = build_theoretical_grid_json(out_dir=None)
    for key in ('nvfp4_full_stats', 'hif4_full_stats', 'nvfp4_scale_format', 'hif4_s0_format'):
        assert now[key] == saved[key], (key, now[key], saved[key])
    assert now['nvfp4_full_internal_grid'] == saved['nvfp4_full_internal_grid']
    assert now['hif4_full_internal_grid'] == saved['hif4_full_internal_grid']
    print('AX3 JSON recompute exact match; NV unique=', now['nvfp4_full_stats']['num_unique_values'])


def test_saved_pt_grids_match_current_enumerators_exactly():
    nv_saved = torch.load(OUT / 'ax3_nvfp4_full_internal_grid.pt', map_location='cpu')
    hf_saved = torch.load(OUT / 'ax3_hif4_full_internal_grid.pt', map_location='cpu')
    nv_now, _ = enumerate_nvfp4_full_internal_grid()
    hf_now, _ = enumerate_hif4_full_internal_grid()
    assert torch.equal(nv_saved.cpu(), nv_now.cpu())
    assert torch.equal(hf_saved.cpu(), hf_now.cpu())
    print('AX3 PT grids exact match current enumerators:', nv_now.numel(), hf_now.numel())
