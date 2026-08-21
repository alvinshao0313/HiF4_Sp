#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/shaoyuantian/program/HiF4_Sp"
PY="/home/shaoyuantian/anaconda3/envs/hif4/bin/python"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
OUT="${OUT_DIR:-Inference_Paradigm_Conversion/results}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_report}"
mkdir -p "$OUT/$RUN_ID"
"$PY" - <<PY
from pathlib import Path
from Inference_Paradigm_Conversion.ipc_analysis.reporting.report import build_report
from Inference_Paradigm_Conversion.ipc_analysis.io_utils import write_text
out = Path("$OUT") / "$RUN_ID"
summary = build_report(Path("$OUT"), out)
print(summary)
write_text(Path("$OUT") / "latest_report_run_id.txt", "$RUN_ID")
PY
echo "REPORT → $OUT/$RUN_ID/report.html"
