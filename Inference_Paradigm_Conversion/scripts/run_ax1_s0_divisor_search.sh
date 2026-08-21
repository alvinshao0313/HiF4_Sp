#!/usr/bin/env bash
# Thin wrapper: run full AX pipeline (all AX1–AX4).
set -euo pipefail
ROOT="/home/shaoyuantian/program/HiF4_Sp"
exec "$ROOT/Inference_Paradigm_Conversion/scripts/run_ax_all.sh" "$@"
