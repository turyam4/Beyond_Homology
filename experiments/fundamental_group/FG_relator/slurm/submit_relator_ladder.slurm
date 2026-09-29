#!/bin/bash -l
#SBATCH --job-name=fg_relator
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
if [[ -f "${SUBMIT_DIR}/run_relator_experiment.sh" ]]; then
    EXPERIMENT_DIR="$SUBMIT_DIR"
elif [[ -f "${SUBMIT_DIR}/../run_relator_experiment.sh" ]]; then
    EXPERIMENT_DIR="$(cd -- "${SUBMIT_DIR}/.." && pwd)"
else
    echo "Error: run_relator_experiment.sh was not found." >&2
    echo "Submit from FG_relator with: sbatch slurm/submit_relator_ladder.slurm" >&2
    exit 2
fi

export PAIRING="${PAIRING:-deep_vs_free}"
export MODELS="${MODELS:-spec_mlp baselines esan smcn}"
export SEEDS="${SEEDS:-0 1 2 3}"
export NUM_GPUS="${NUM_GPUS:-4}"
export DEVICE=cpu
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export PYTHONDONTWRITEBYTECODE=1

cd "$EXPERIMENT_DIR"

echo "job=${SLURM_JOB_ID:-local} pairing=${PAIRING} seeds=${SEEDS}"
bash run_relator_experiment.sh
