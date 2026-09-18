#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 6 ]]; then
    echo "Usage: bash $0 SOURCE_ROOT RUN_ROOT TRAIN_GPUS EVAL_GPUS SMOKE_REPORT TMUX_SESSION" >&2
    exit 2
fi
source_root=$(realpath -e -- "$1")
run_root=$(realpath -m -- "$2")
train_gpus=$3
eval_gpus=$4
smoke_report=$(realpath -e -- "$5")
session=$6
if [[ ! $session =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo "Invalid tmux session name" >&2
    exit 2
fi
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../../.." && pwd)
command -v tmux >/dev/null
if tmux has-session -t "=$session" 2>/dev/null; then
    echo "tmux session already exists: $session" >&2
    exit 1
fi
if [[ -e "$run_root" ]]; then
    echo "RUN_ROOT must not exist; formal inputs are cloned into a new root" >&2
    exit 1
fi
mkdir -p -- "$(dirname -- "$run_root")"
cd -- "$repo_root"
/home/shaoyuantian/anaconda3/envs/hif4/bin/python -u -c \
    'from pathlib import Path; import sys; from Native_NVFP4_HiF4_Linear_Puncture.experiments.kl_direction_reuse.restart import reuse_inputs; reuse_inputs(Path(sys.argv[1]), Path(sys.argv[2]), verification=True)' \
    "$source_root" "$run_root"
control="${run_root}.long_${session}.control"
mkdir -- "$control"
{
    printf '#!/usr/bin/env bash\nset -euo pipefail\n'
    printf 'cd -- %q\n' "$repo_root"
    printf 'export CUDA_VISIBLE_DEVICES="" CUDA_DEVICE_ORDER=PCI_BUS_ID\n'
    printf 'export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1\n'
    printf 'exec /home/shaoyuantian/anaconda3/envs/hif4/bin/python -u %q ' "$script_dir/long_pipeline.py"
    printf -- '--root %q --control %q --train-gpus %q --eval-gpus %q --smoke-report %q --reuse-verified' \
        "$run_root" "$control" "$train_gpus" "$eval_gpus" "$smoke_report"
    printf ' > %q 2>&1\n' "$control/pipeline.log"
} > "$control/run.sh"
chmod +x "$control/run.sh"
tmux new-session -d -s "$session" -c "$repo_root" "exec bash $(printf '%q' "$control/run.sh")"
printf 'Session: %s\nStatus: %s/status.json\nLog: %s/pipeline.log\n' "$session" "$control" "$control"
