from __future__ import annotations

import csv
import gzip
import json
import sys
from pathlib import Path

import torch

ROOT = Path('/home/shaoyuantian/program/HiF4_Sp')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS = ROOT / 'Inference_Paradigm_Conversion' / 'results'


def test_inventory_fingerprints():
    p = RESULTS / '20260811T_ax_final_consolidated' / 'ax3_theoretical_grid.json'
    d = json.loads(p.read_text())
    nv = d['nvfp4_full_stats']
    sf = d['nvfp4_scale_format']
    assert sf['num_scale_values'] == 127
    assert sf['max_scale'] == 448.0
    assert 'E4M3FN' in sf['name']
    assert nv['num_raw_combinations'] == 1905.0
    assert nv['num_unique_values'] == 475.0
    assert nv['min_nonzero_abs'] == 0.0009765625
    assert nv['max_abs'] == 2688.0
    print('AX3 fingerprint OK:', sf, nv)


def test_all_formal_result_text_has_no_positive_e8m0_semantics():
    hits = []
    for p in RESULTS.rglob('*'):
        if not p.is_file() or p.suffix.lower() not in {'.json', '.md', '.csv', '.txt', '.log'}:
            continue
        try:
            txt = p.read_text(errors='ignore')
        except Exception:
            continue
        if 'E8M0' in txt:
            hits.append((str(p.relative_to(ROOT)), [ln.strip() for ln in txt.splitlines() if 'E8M0' in ln][:4]))
    print('E8M0 hits in result text:', hits)
    assert all(any(tok in line for tok in ('not E8M0', '不是 E8M0')) for _, lines in hits for line in lines)


def test_ax_csv_fingerprint_is_e4m3():
    p = RESULTS / '20260811T_ax_final_consolidated' / 'ax3_local_scale_distribution.csv'
    with p.open(newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        first = next(reader)
    assert 'nv_e4m3_local_mean' in fields
    assert 'nv_e4m3_local_std' in fields
    assert 'nv_raw_local_mean' in fields
    assert float(first['nv_e4m3_local_mean']) != 0.0
    print('AX CSV fields fingerprint OK; first NV local:', first['nv_e4m3_local_mean'], first['nv_raw_local_mean'])


def test_repr_raw_record_shape_and_keys():
    p = RESULTS / '20260811T015135Z_repr_al' / 'repr_activation_raw_shard0.jsonl.gz'
    with gzip.open(p, 'rt') as f:
        obj = json.loads(next(f))
    print('repr raw keys:', sorted(obj.keys()))
    for k,v in obj.items():
        if isinstance(v, list):
            print('LIST', k, 'len=', len(v), 'head=', v[:3])
        elif isinstance(v, dict):
            print('DICT', k, 'keys=', list(v)[:20])
        else:
            print('SCALAR', k, type(v).__name__, v if isinstance(v,(str,int,float,bool,type(None))) else '<obj>')
    assert obj
