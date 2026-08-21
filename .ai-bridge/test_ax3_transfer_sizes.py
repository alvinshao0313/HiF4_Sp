import base64, gzip
from pathlib import Path

BASE=Path('Inference_Paradigm_Conversion/results/20260811T_ax_final_consolidated/figures')
NAMES=['fig_ax3_full_internal_grid_hist_pm0p1.png','fig_ax3_full_internal_grid_hist_pm0p01.png','fig_ax3_full_internal_grid_hist_pm0p001.png','fig_ax3_full_internal_grid_hist_pm1.png','fig_ax3_full_internal_grid_hist_pm10.png','fig_ax3_full_internal_grid_hist_pm100.png']

def test_sizes():
    for n in NAMES:
        raw=(BASE/n).read_bytes(); gz=gzip.compress(raw, compresslevel=9)
        print(n, len(raw), len(gz), len(base64.b64encode(gz)))
