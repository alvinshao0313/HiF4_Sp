"""Pinned task data, then the repository's real main.py."""
import os
from pathlib import Path
import runpy

from ..matrix_norm_efficiency.pinned import install
from ..matrix_norm_efficiency.reasoning_main import install_engine_audit
from .config import REPO

if __name__ == "__main__":
    install(Path(os.environ["HIF4_SUPPLEMENT_RUN"]))
    install_engine_audit()
    runpy.run_path(str(REPO / "main.py"), run_name="__main__")
