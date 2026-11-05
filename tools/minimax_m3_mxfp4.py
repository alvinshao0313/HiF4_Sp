#!/usr/bin/env python
"""Opt-in MXFP4 registration in the parent and every spawned model worker."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / '3rdparty/lighteval/src'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from tools.minimax_mxfp4.integration import register
register()

if __name__ == '__main__':
    from tools.minimax_m3 import main
    main()
