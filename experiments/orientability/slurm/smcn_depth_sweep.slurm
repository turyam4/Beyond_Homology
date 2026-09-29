#!/bin/bash -l
#SBATCH --job-name=fg_rel_smcn_depth
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:1
#SBATCH --time=05:30:00
#SBATCH --array=0-9
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
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

# depth grid: (cin_layers, subgraph_layers) -> total depth 2,3,4,6,8
DEPTHS=("1 1" "1 2" "2 2" "2 4" "2 6")
MARKS=(true false)
D=$(( SLURM_ARRAY_TASK_ID / 2 ))
M=$(( SLURM_ARRAY_TASK_ID % 2 ))
read CIN SUB <<< "${DEPTHS[$D]}"
MARK=${MARKS[$M]}
TAG=$([ "$MARK" = "true" ] && echo binary || echo spd)
TOT=$(( CIN + SUB ))

echo "TASK=$SLURM_ARRAY_TASK_ID  cin=$CIN subgraph=$SUB total_depth=$TOT marking=$TAG"
"$PY" -c "import torch;print('cuda:',torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

COMMON="samples_per_g=16 number_of_permuted_copies=32 batch_size=8 eval_batch_size=8 num_epochs=20"
for SEED in 0 1 2 3; do
  echo "=== SMCN depth=${CIN}x${SUB} marking=$TAG seed=$SEED ==="
  "$PY" "$CODE/run_relator.py" pairing=deep_vs_free 'model_names=[smcn]' \
      binary_marking=$MARK model_seed=$SEED device=auto $COMMON \
      number_cin_layers_for_subcomplex=$CIN number_subgraph_layers=$SUB \
      dataset_cache_dir=$WORK/cache_$MARK \
      output_dir=$WORK/runs/depth_${CIN}x${SUB}/smcn_${TAG}/seed$SEED
done
echo "TASK $SLURM_ARRAY_TASK_ID DONE"
