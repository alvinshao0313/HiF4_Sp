from pathlib import Path
import sys

BLOCK_SPARSE_ROOT = Path(__file__).resolve().parents[2]
if str(BLOCK_SPARSE_ROOT) not in sys.path:
    sys.path.insert(0, str(BLOCK_SPARSE_ROOT))
