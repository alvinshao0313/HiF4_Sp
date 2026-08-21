from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

ROOT = Path('/home/shaoyuantian/program/HiF4_Sp')
A2 = ROOT / 'Inference_Paradigm_Conversion/results/20260811T032247Z_a2'


def test_a2_artifact_fingerprint():
    with (A2 / 'a2_variants.csv').open(newline='') as f:
        r = csv.DictReader(f)
        print('A2 CSV fields:', r.fieldnames)
        first = next(r)
        print('A2 first row:', first)
    with gzip.open(A2 / 'activation_group_records_shard0.jsonl.gz', 'rt') as f:
        obj = json.loads(next(f))
    print('A2 group record keys:', sorted(obj.keys()))
    print('A2 group record:', obj)
    txt = json.dumps(obj).lower()
    assert 'e8m0' not in txt
