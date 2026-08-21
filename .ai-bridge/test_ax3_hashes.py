from pathlib import Path
import hashlib
BASE=Path('Inference_Paradigm_Conversion/results/20260811T_ax_final_consolidated/figures')
NAMES=['fig_ax3_full_internal_grid_hist_pm0p1.png','fig_ax3_full_internal_grid_hist_pm0p01.png','fig_ax3_full_internal_grid_hist_pm0p001.png','fig_ax3_full_internal_grid_hist_pm1.png','fig_ax3_full_internal_grid_hist_pm10.png','fig_ax3_full_internal_grid_hist_pm100.png']
def test_hashes():
    for n in NAMES:
        b=(BASE/n).read_bytes()
        print(n, len(b), hashlib.sha256(b).hexdigest())
