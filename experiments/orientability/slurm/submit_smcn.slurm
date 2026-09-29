#!/bin/bash -l
#SBATCH --job-name=fg_relator_smcn
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail

# Paths are derived from the checkout and may be overridden by environment.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${BEYOND_HOMOLOGY_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"
CODE="${RELATOR_DIR:-$REPO_ROOT/experiments/fundamental_group/FG_relator}"
WORK="${OUTPUT_ROOT:-$CODE/outputs/fg_relator_smcn}"
if [[ -n "${PYTHON:-}" ]]; then
  PY="$PYTHON"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python3" ]]; then
  PY="$VIRTUAL_ENV/bin/python3"
elif [[ -x "$REPO_ROOT/../smcn-env/bin/python3" ]]; then
  PY="$REPO_ROOT/../smcn-env/bin/python3"
else
  PY="$(command -v python3)"
fi
[[ -f "$CODE/run_relator.py" ]] || { echo "Missing $CODE/run_relator.py" >&2; exit 2; }
[[ -x "$PY" ]] || { echo "Missing Python executable: $PY" >&2; exit 2; }
mkdir -p "$WORK"

export FG_ROOT="$REPO_ROOT/experiments/fundamental_group"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/experiments:$CODE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1        # never drop __pycache__ into Beyond_Homology
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=$OMP_NUM_THREADS

PAIRING=deep_vs_free
COMMON="samples_per_g=16 number_of_permuted_copies=32 batch_size=8 eval_batch_size=8 num_epochs=20"

echo "JOB=${SLURM_JOB_ID:-local}"
"$PY" -c "import torch;print('cuda available:',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# ---- 1. build the two fixed caches (binary + distance marking) ----
for MARK in true false; do
  echo "=== CACHE marking=$MARK ==="
  "$PY" "$CODE/run_relator.py" pairing=$PAIRING cache_only=true \
      binary_marking=$MARK $COMMON \
      dataset_cache_dir=$WORK/cache_$MARK
done

# ---- 2. SMCN, both markings, 4 seeds ----
for MARK in true false; do
  TAG=$([ "$MARK" = "true" ] && echo binary || echo spd)
  for SEED in 0 1 2 3; do
    echo "=== SMCN marking=$TAG seed=$SEED ==="
    "$PY" "$CODE/run_relator.py" pairing=$PAIRING \
        'model_names=[smcn]' binary_marking=$MARK model_seed=$SEED \
        $COMMON device=auto \
        dataset_cache_dir=$WORK/cache_$MARK \
        output_dir=$WORK/runs/smcn_$TAG/seed$SEED
  done
done
echo "ALL DONE"
