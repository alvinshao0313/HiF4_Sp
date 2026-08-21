#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/shaoyuantian/program/HiF4_Sp"
SKIP_AX1=1 SKIP_AX2=1 SKIP_AX3=1 exec "$ROOT/Inference_Paradigm_Conversion/scripts/run_ax_all.sh" "$@"
