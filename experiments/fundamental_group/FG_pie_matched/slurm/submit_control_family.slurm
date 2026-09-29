#!/bin/bash -l
# =============================================================================
# Slurm launcher for the pi_1-matched fundamental-group controls.
#
# Default: both control families, SMCN binary + SMCN SPD + ESAN,
#          model seeds 0,1,2,3 as four CPU workers.
#
# Submit from FG_pie_matched (or its slurm directory):
#   sbatch submit_control_family.slurm
#
# Optional overrides:
#   sbatch --export=ALL,CONTROL_FAMILY=surface_vs_surface submit_control_family.slurm
#   sbatch --export=ALL,LADDER=1 submit_control_family.slurm
# =============================================================================
#SBATCH --job-name=fg_pi1_control
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=30G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%j-%x.out
#SBATCH --error=slurm-%j-%x.err

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -f "${SUBMIT_DIR}/run_control_family.sh" ]]; then
    EXPERIMENT_DIR="$SUBMIT_DIR"
elif [[ -f "${SUBMIT_DIR}/../run_control_family.sh" ]]; then
    EXPERIMENT_DIR="$(cd -- "${SUBMIT_DIR}/.." && pwd)"
else
    echo "Error: run_control_family.sh was not found." >&2
    echo "Submit from FG_pie_matched with: sbatch slurm/submit_control_family.slurm" >&2
    exit 2
fi

export CONTROL_FAMILY="${CONTROL_FAMILY:-both}"
export SEEDS="${SEEDS:-0 1 2 3}"
export NUM_GPUS="${NUM_GPUS:-4}"
export DEVICE=cpu
export SAMPLES="${SAMPLES:-16}"
export COPIES="${COPIES:-32}"
export EPOCHS="${EPOCHS:-20}"
export BATCH="${BATCH:-8}"
export LADDER="${LADDER:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export PYTHONDONTWRITEBYTECODE=1

cd "$EXPERIMENT_DIR"

echo "job=${SLURM_JOB_ID:-local}"
echo "control_family=${CONTROL_FAMILY} seeds=${SEEDS} num_gpus=${NUM_GPUS} ladder=${LADDER}"

bash run_control_family.sh \
    "control_family=${CONTROL_FAMILY}" \
    "seeds=[${SEEDS// /,}]" \
    "num_gpus=${NUM_GPUS}" \
    "ladder=${LADDER}"

echo "Control-family experiment completed."
