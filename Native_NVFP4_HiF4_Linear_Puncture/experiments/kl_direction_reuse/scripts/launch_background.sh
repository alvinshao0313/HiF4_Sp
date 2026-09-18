#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 && $# -ne 5 ]]; then
    echo "Usage: bash $0 RUN_ROOT TRAIN_GPU EVAL_GPUS TMUX_SESSION [--after-baseline|--after-captures]" >&2
    exit 2
fi
run_root=$(realpath -m -- "$1")
train_gpu=$2
eval_gpus=$3
session=$4
extra_args=()
if [[ $# -eq 5 ]]; then
    if [[ $5 != --after-baseline && $5 != --after-captures ]]; then
        echo "Select --after-baseline or --after-captures as the explicit continuation boundary." >&2
        exit 2
    fi
    extra_args=("$5")
fi
if [[ ! $session =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo "Use letters, digits, underscores or hyphens for the tmux session." >&2
    exit 2
fi
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../../.." && pwd)
conda_exe=$(command -v conda)
command -v tmux >/dev/null
if tmux has-session -t "=$session" 2>/dev/null; then
    echo "tmux session already exists: $session" >&2
    exit 1
fi
control="${run_root}.control"
if [[ ${#extra_args[@]} -ne 0 ]]; then
    control="${run_root}.resume_${session}.control"
fi
mkdir -p -- "$(dirname -- "$run_root")"
mkdir -- "$control"
{
    printf '#!/usr/bin/env bash\nset -euo pipefail\n'
    printf 'cd -- %q\n' "$repo_root"
    printf 'export CUDA_VISIBLE_DEVICES="" CUDA_DEVICE_ORDER=PCI_BUS_ID\n'
    printf 'export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1\n'
    printf 'exec '
    printf '%q ' "$conda_exe" run --no-capture-output -n hif4 python -u -m \
        Native_NVFP4_HiF4_Linear_Puncture.experiments.kl_direction_reuse.host \
        --root "$run_root" --control "$control" --train-gpu "$train_gpu" --eval-gpus "$eval_gpus" "${extra_args[@]}"
    printf '> %q 2>&1\n' "$control/pipeline.log"
} > "$control/run.sh"
tmux new-session -d -s "$session" -c "$repo_root" "exec bash $(printf '%q' "$control/run.sh")"
printf 'Session: %s\nRun: %s\nStatus: %s/status.json\nCurrent command: %s/current.json\nLog: %s/pipeline.log\n' \
    "$session" "$run_root" "$control" "$control" "$control"
