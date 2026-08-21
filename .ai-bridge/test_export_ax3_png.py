import base64
from pathlib import Path

BASE = Path('Inference_Paradigm_Conversion/results/20260811T_ax_final_consolidated/figures')
FILES = {
    'pm0p1': 'fig_ax3_full_internal_grid_hist_pm0p1.png',
    'pm0p01': 'fig_ax3_full_internal_grid_hist_pm0p01.png',
    'pm0p001': 'fig_ax3_full_internal_grid_hist_pm0p001.png',
    'pm1': 'fig_ax3_full_internal_grid_hist_pm1.png',
    'pm10': 'fig_ax3_full_internal_grid_hist_pm10.png',
    'pm100': 'fig_ax3_full_internal_grid_hist_pm100.png',
}

def emit(key: str, part: int):
    raw = (BASE / FILES[key]).read_bytes()
    data = base64.b64encode(raw).decode('ascii')
    cut = len(data) // 2
    chunk = data[:cut] if part == 0 else data[cut:]
    print(f'B64BEGIN:{key}:{part}')
    print(chunk)
    print(f'B64END:{key}:{part}')

def test_pm0p1_a(): emit('pm0p1',0)
def test_pm0p1_b(): emit('pm0p1',1)
def test_pm0p01_a(): emit('pm0p01',0)
def test_pm0p01_b(): emit('pm0p01',1)
def test_pm0p001_a(): emit('pm0p001',0)
def test_pm0p001_b(): emit('pm0p001',1)
def test_pm1_a(): emit('pm1',0)
def test_pm1_b(): emit('pm1',1)
def test_pm10_a(): emit('pm10',0)
def test_pm10_b(): emit('pm10',1)
def test_pm100_a(): emit('pm100',0)
def test_pm100_b(): emit('pm100',1)
