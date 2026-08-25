#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
ensure_run_dir
write_manifest
KEEP="${1:-}"
LOG="${RUN_DIR}/logs/m8_${KEEP:-all}.log"
exec > >(tee -a "${LOG}") 2>&1
PIN_GPU="${PIN_GPU:-}"
if [[ -n "${KEEP}" ]]; then
  case "${KEEP}" in
    0.75|0.750) tag=m8_keep075 ;;
    0.5|0.50|0.500) tag=m8_keep050 ;;
    0.25|0.250) tag=m8_keep025 ;;
    *) tag="m8_keep${KEEP}" ;;
  esac
  run_method_full "${tag}" m8_energy "${KEEP}" "${PIN_GPU}"
else
  run_method_full m8_keep075 m8_energy 0.75 "${PIN_GPU}"
  run_method_full m8_keep050 m8_energy 0.50 "${PIN_GPU}"
  run_method_full m8_keep025 m8_energy 0.25 "${PIN_GPU}"
fi
echo "m8 DONE"
