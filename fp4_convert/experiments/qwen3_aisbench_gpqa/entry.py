"""Register the existing quantizers in parent and spawn workers, then run main.py."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for directory in (ROOT, ROOT / "3rdparty/lighteval/src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from fp4_convert.experiments.qwen3_mxfp4.integration import register as base
from fp4_convert.experiments.qwen3_mxfp4_r16x2.integration import register as r16x2
from fp4_convert.experiments.qwen3_mxfp4_uos.integration import register as uos
from fp4_convert.experiments.qwen3_mxfp4_dualscale.integration import register as dualscale

for register in (base, r16x2, uos, dualscale):
    register()

if __name__ == "__main__":
    from main import main
    main()
