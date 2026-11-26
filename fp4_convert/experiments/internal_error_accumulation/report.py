"""Final report assembly."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .run_state import atomic_write_json, read_json


def write_final_report(run_root: Path, analysis_dir: Path) -> Path:
    analysis_dir = Path(analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    state = read_json(Path(run_root) / "run_state.json")
    lines = [
        "# INTERNAL ERROR ACCUMULATION REPORT\n",
        f"- run_id: {state.get('run_id')}\n",
        f"- status: {state.get('status')}\n",
        f"- completed_stages: {state.get('completed_stages')}\n",
        "\n## Required answers (Q1–Q7)\n",
        "Filled as stages complete; see stage artifacts under this RUN_ID.\n",
    ]
    path = analysis_dir / "INTERNAL_ERROR_ACCUMULATION_REPORT.md"
    path.write_text("".join(lines), encoding="utf-8")
    atomic_write_json(analysis_dir / "report_index.json", {"report": str(path), "run_state": state})
    return path
