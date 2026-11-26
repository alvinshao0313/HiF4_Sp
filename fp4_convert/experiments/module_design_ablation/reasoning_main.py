"""Run the repository's main.py with explicit learned-rotation workers."""
from pathlib import Path
import runpy


if __name__ == "__main__":
    from .runtime import install_engine_hook
    install_engine_hook()
    runpy.run_path(str(Path(__file__).resolve().parents[3] / "main.py"), run_name="__main__")
