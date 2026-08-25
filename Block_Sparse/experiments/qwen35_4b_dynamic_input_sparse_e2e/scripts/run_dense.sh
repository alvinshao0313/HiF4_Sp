#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
ensure_run_dir
write_manifest
LOG="${RUN_DIR}/logs/dense.log"
exec > >(tee -a "${LOG}") 2>&1
run_method_full dense none 1.0
echo "dense DONE"
