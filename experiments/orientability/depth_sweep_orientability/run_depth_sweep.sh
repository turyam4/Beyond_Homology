#!/usr/bin/env bash
# Full SMCN depth sweep on the old punctured torus/Klein benchmark.
set -uo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORIENT_DIR="$(cd -- "$HERE/.." && pwd)"
REPO_ROOT="$(cd -- "$ORIENT_DIR/../.." && pwd)"
RUNNER="$ORIENT_DIR/run.py"
PAIR_DIR="$REPO_ROOT/data/punc_torus_vs_punc_klein"
PY="${PYTHON:-$REPO_ROOT/../smcn-env/bin/python3}"
OUT="${OUT:-$HERE/runs}"
LOGS="${LOGS:-$HERE/logs}"

# Eight depths x three seeds x all 50 old-grid pairs.
DEPTHS="${DEPTHS:-1 2 3 4 5 6 7 8}"
SEEDS="${SEEDS:-0 1 2}"
DEVICE="${DEVICE:-cuda}"
JOBS="${JOBS:-1}"
THREADS="${THREADS:-8}"
GPU_CARDS="${GPU_CARDS:-${CUDA_VISIBLE_DEVICES:-0}}"

[[ -f "$RUNNER" ]] || { echo "Missing local evaluator: $RUNNER" >&2; exit 2; }
[[ -x "$PY" ]] || { echo "Missing local Python environment: $PY" >&2; exit 2; }
[[ $(find "$PAIR_DIR" -maxdepth 1 -name 'h*_p*_first.pt' | wc -l) -eq 50 ]] || {
    echo "Expected 50 old punctured pairs in $PAIR_DIR" >&2
    exit 2
}
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "JOBS must be a positive integer" >&2; exit 2; }
IFS=',' read -r -a gpu_cards <<< "$GPU_CARDS"
if [[ "$DEVICE" == cuda* && ${#gpu_cards[@]} -lt "$JOBS" ]]; then
    echo "JOBS=$JOBS requires at least $JOBS entries in GPU_CARDS=$GPU_CARDS" >&2
    exit 2
fi

mkdir -p "$OUT" "$LOGS"
echo "Evaluator: $RUNNER"
echo "Pair data: $PAIR_DIR"
echo "Output:    $OUT"
echo "Depths:    $DEPTHS"
echo "Seeds:     $SEEDS"
echo "Device:    $DEVICE (concurrent runs: $JOBS)"

run_one() {
    local depth=$1 seed=$2 gpu=$3 tag="L${1}_seed${2}"
    local run_dir="$OUT/$tag"
    if [[ -f "$run_dir/pair_results.csv" && -f "$run_dir/metrics.json" ]]; then
        echo "skip $tag (complete)"
        return 0
    fi

    echo "start $tag"
    CUDA_VISIBLE_DEVICES="$gpu" \
    OMP_NUM_THREADS="$THREADS" \
    MKL_NUM_THREADS="$THREADS" \
    OPENBLAS_NUM_THREADS="$THREADS" \
    "$PY" "$RUNNER" \
        family=punc_torus_vs_punc_klein \
        'model_names=[smcn]' \
        "families.punc_torus_vs_punc_klein.base_pair_dir=$PAIR_DIR" \
        families.punc_torus_vs_punc_klein.dataset.metric_matched=false \
        "models.smcn.number_subgraph_layers=$depth" \
        models.smcn.number_cin_layers=0 \
        models.smcn.binary_marking=true \
        models.smcn.free_memory_each_batch=true \
        fresh_model_per_pair=true \
        max_pairs=null \
        "model_seed=$seed" \
        "seed=$seed" \
        "device=$DEVICE" \
        "output_dir=$run_dir" >"$LOGS/$tag.log" 2>&1
    local status=$?
    if (( status == 0 )); then
        echo "done $tag"
    else
        echo "FAILED $tag (status=$status, log=$LOGS/$tag.log)" >&2
    fi
    return "$status"
}

failures=0
active=0
launch_index=0
pids=()
for depth in $DEPTHS; do
    for seed in $SEEDS; do
        gpu="${gpu_cards[$((launch_index % ${#gpu_cards[@]}))]}"
        if (( JOBS == 1 )); then
            run_one "$depth" "$seed" "$gpu" || failures=1
        else
            run_one "$depth" "$seed" "$gpu" &
            pids+=("$!")
            active=$((active + 1))
            if (( active >= JOBS )); then
                for pid in "${pids[@]}"; do wait "$pid" || failures=1; done
                active=0
                pids=()
            fi
        fi
        launch_index=$((launch_index + 1))
    done
done
for pid in "${pids[@]}"; do wait "$pid" || failures=1; done

if (( failures != 0 )); then
    echo "Depth sweep finished with failed runs; inspect $LOGS" >&2
    exit 1
fi
echo "All depth-sweep runs complete. Aggregate with:"
echo "  $PY $HERE/aggregate_depth_sweep.py"
