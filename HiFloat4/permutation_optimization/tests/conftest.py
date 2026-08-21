"""Pytest path setup: HiFloat4 root on sys.path."""

from __future__ import annotations

import sys
from pathlib import Path

_HIFLOAT4 = Path(__file__).resolve().parents[1]
if str(_HIFLOAT4) not in sys.path:
    sys.path.insert(0, str(_HIFLOAT4))
