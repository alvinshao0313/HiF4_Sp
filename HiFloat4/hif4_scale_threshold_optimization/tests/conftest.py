"""Pytest path bootstrap for hif4_scale_threshold_optimization."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_HIFLOAT4 = _ROOT.parent
_REPO = _HIFLOAT4.parent

for p in (_ROOT, _HIFLOAT4, _REPO / "ChuanCi", _REPO):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)
