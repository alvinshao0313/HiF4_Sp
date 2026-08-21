import base64, gzip
from pathlib import Path
BASE=Path('Inference_Paradigm_Conversion/results/20260811T_ax_final_consolidated/figures')
FILES={
'pm0p1':'fig_ax3_full_internal_grid_hist_pm0p1.png',
'pm0p01':'fig_ax3_full_internal_grid_hist_pm0p01.png',
'pm0p001':'fig_ax3_full_internal_grid_hist_pm0p001.png',
'pm1':'fig_ax3_full_internal_grid_hist_pm1.png',
'pm10':'fig_ax3_full_internal_grid_hist_pm10.png',
'pm100':'fig_ax3_full_internal_grid_hist_pm100.png'}
def emit(k):
    raw=(BASE/FILES[k]).read_bytes()
    print(base64.b64encode(gzip.compress(raw,compresslevel=9)).decode('ascii'))
def test_pm0p1(): emit('pm0p1')
def test_pm0p01(): emit('pm0p01')
def test_pm0p001(): emit('pm0p001')
def test_pm1(): emit('pm1')
def test_pm10(): emit('pm10')
def test_pm100(): emit('pm100')
