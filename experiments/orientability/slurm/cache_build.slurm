#!/bin/bash -l
#SBATCH --job-name=fg_rel_cache
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail

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
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
COMMON="samples_per_g=16 number_of_permuted_copies=32 batch_size=8 eval_batch_size=8 num_epochs=20"
for MARK in true false; do
  echo "=== CACHE marking=$MARK ==="
  "$PY" "$CODE/run_relator.py" pairing=deep_vs_free cache_only=true \
      binary_marking=$MARK $COMMON dataset_cache_dir=$WORK/cache_$MARK
done
echo "CACHES DONE"
