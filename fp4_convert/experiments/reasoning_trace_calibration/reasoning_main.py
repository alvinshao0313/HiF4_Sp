"""Invoke the real main.py after installing only this process's pinned-data loader."""
import runpy

from .config import REPO
from .pinned import install

if __name__ == "__main__":
    install()
    runpy.run_path(str(REPO / "main.py"), run_name="__main__")
