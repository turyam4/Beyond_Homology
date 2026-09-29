#!/bin/bash -l
# Run GIN, HOMP, and SMCN concurrently on the metric-matched punctured family.
# Each model receives one node with two GPUs; seeds 0 and 1 run concurrently,
# one seed per GPU. All inputs, code, and outputs stay inside Beyond_Homology.
#
# Submit with:
#   sbatch submit_metric_punc_models.slurm

#SBATCH --job-name=punc_metric_models
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --time=0-06:00:00
#SBATCH --output=slurm-%j-%x.out
#SBATCH --error=slurm-%j-%x.err

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${BEYOND_HOMOLOGY_ROOT:-$(cd -- "$script_dir/../../../../.." && pwd)}"
orient_dir="$repo_root/experiments/orientability"
metric_root="$orient_dir/punc_torus_vs_punc_klein/metric_matched"
data_dir="$repo_root/data/punc_torus_vs_punc_klein_metric_matched"
if [[ -n "${PYTHON:-}" ]]; then
    python_bin="$PYTHON"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python3" ]]; then
    python_bin="$VIRTUAL_ENV/bin/python3"
elif [[ -x "$repo_root/../smcn-env/bin/python3" ]]; then
    python_bin="$repo_root/../smcn-env/bin/python3"
else
    python_bin="$(command -v python3)"
fi
run_tag="${RUN_TAG:-job${SLURM_JOB_ID:-local}}"

[[ -f "$orient_dir/run.py" ]] || { echo "Missing $orient_dir/run.py" >&2; exit 2; }
[[ -x "$python_bin" ]] || { echo "Missing $python_bin" >&2; exit 2; }
[[ -d "$data_dir" ]] || { echo "Missing data directory: $data_dir" >&2; exit 2; }
[[ $(find "$data_dir" -maxdepth 1 -name 'h*_p*_first.pt' | wc -l) -eq 50 ]] || {
    echo "Expected 50 metric-matched pairs in $data_dir" >&2
    exit 2
}

mkdir -p "$metric_root/logs" "$metric_root/runs" "$metric_root/slurm"
export PYTHONPATH="$repo_root:$repo_root/experiments:$orient_dir${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export PYTHON_BIN="$python_bin"
export ORIENT_DIR="$orient_dir"
export DATA_DIR="$data_dir"
export METRIC_ROOT="$metric_root"
export RUN_TAG="$run_tag"

echo "job=${SLURM_JOB_ID:-local} nodes=${SLURM_JOB_NUM_NODES:-3}"
echo "data=$data_dir"
echo "outputs=$metric_root"
echo "mapping: one model per node; seeds 0 and 1 use two GPUs"

run_step='seed=$SLURM_PROCID
exec "$PYTHON_BIN" "$ORIENT_DIR/run.py" \
    family=punc_torus_vs_punc_klein \
    device=cuda \
    max_pairs=null \
    "model_seed=$seed" \
    "seed=$seed" \
    "families.punc_torus_vs_punc_klein.base_pair_dir=$DATA_DIR" \
    families.punc_torus_vs_punc_klein.dataset.metric_matched=true \
    "model_names=[$MODEL_NAME]" \
    "batch_size=$MODEL_BATCH" \
    "models.smcn.batch_size=$MODEL_BATCH" \
    models.smcn.free_memory_each_batch=true \
    "output_dir=$METRIC_ROOT/runs/$MODEL_DIR/seed${seed}_${RUN_TAG}"'

MODEL_NAME=gin MODEL_DIR=gin MODEL_BATCH=16 \
srun --nodes=1 --ntasks=2 --ntasks-per-node=2 --exclusive --gpus-per-task=1 \
    --output="$metric_root/logs/gin_seed%t_${run_tag}.log" \
    --error="$metric_root/logs/gin_seed%t_${run_tag}.err" \
    bash -c "$run_step" &
gin_pid=$!

MODEL_NAME=homp_cin MODEL_DIR=homp MODEL_BATCH=16 \
srun --nodes=1 --ntasks=2 --ntasks-per-node=2 --exclusive --gpus-per-task=1 \
    --output="$metric_root/logs/homp_seed%t_${run_tag}.log" \
    --error="$metric_root/logs/homp_seed%t_${run_tag}.err" \
    bash -c "$run_step" &
homp_pid=$!

MODEL_NAME=smcn MODEL_DIR=smcn MODEL_BATCH=4 \
srun --nodes=1 --ntasks=2 --ntasks-per-node=2 --exclusive --gpus-per-task=1 \
    --output="$metric_root/logs/smcn_seed%t_${run_tag}.log" \
    --error="$metric_root/logs/smcn_seed%t_${run_tag}.err" \
    bash -c "$run_step" &
smcn_pid=$!

failed=0
wait "$gin_pid" || { echo "GIN failed; inspect its model log" >&2; failed=1; }
wait "$homp_pid" || { echo "HOMP failed; inspect its model log" >&2; failed=1; }
wait "$smcn_pid" || { echo "SMCN failed; inspect its model log" >&2; failed=1; }

if (( failed != 0 )); then
    exit 1
fi
echo "GIN, HOMP, and SMCN completed successfully for seeds 0 and 1."
