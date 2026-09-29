#!/bin/bash -l
# =============================================================================
# SLURM array launcher: depth sweep on the punctured orientability pair.
#
# 24 tasks = 8 depths (L in {1,...,8}) x 3 model seeds.
# One task per (L, seed); each task evaluates all 50 punctured pairs.
#
#   sbatch slurm/submit_depth_sweep.slurm
#
# Resumable: completed depth/seed runs are skipped by run_depth_sweep.sh.
#
# After the array finishes:
#   python3 aggregate_depth_sweep.py
# =============================================================================
#SBATCH --job-name=depth_orient
#SBATCH --array=0-23
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --time=0-06:00:00
#SBATCH -p gpu
#SBATCH --qos short
#SBATCH --output=slurm-%A_%a-%x.out
#SBATCH --error=slurm-%A_%a-%x.err

set -euo pipefail

SLURM_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_DIR="$(cd -- "$SLURM_DIR/.." && pwd)"
REPO_ROOT="$(cd -- "$SWEEP_DIR/../../.." && pwd)"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python3" ]]; then
    PYTHON="$VIRTUAL_ENV/bin/python3"
  elif [[ -x "$REPO_ROOT/../smcn-env/bin/python3" ]]; then
    PYTHON="$REPO_ROOT/../smcn-env/bin/python3"
  else
    PYTHON="$(command -v python3)"
  fi
fi
export PYTHON
[[ -x "$PYTHON" ]] || { echo "Missing Python executable: $PYTHON" >&2; exit 2; }
mkdir -p "$SWEEP_DIR/logs" "$SWEEP_DIR/runs"
cd "$SWEEP_DIR"

DEPTHS=(1 2 3 4 5 6 7 8)
SEEDS=(0 1 2)
IDX=${SLURM_ARRAY_TASK_ID:-0}
L=${DEPTHS[$(( IDX / ${#SEEDS[@]} ))]}
S=${SEEDS[$(( IDX % ${#SEEDS[@]} ))]}

echo "array task $IDX -> subgraph_layers=$L model_seed=$S"
DEVICE=cuda JOBS=1 THREADS="${SLURM_CPUS_PER_TASK:-8}" \
  OUT="$SWEEP_DIR/runs" LOGS="$SWEEP_DIR/logs" \
  DEPTHS="$L" SEEDS="$S" "$SWEEP_DIR/run_depth_sweep.sh"
