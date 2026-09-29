#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash fundamental_group.sh [family=fundamental_group] [model_name=<model|model,...>] [key=value ...]

Defaults:
  family=fundamental_group model_name=gin num_gpus=1
  GPU_CARDS sets the GPU count when num_gpus is omitted.
  Without seeds=..., seeds are 0..GPU_COUNT-1.

Friendly aliases:
  epochs=N       -> num_epochs=N
  copies=N       -> number_of_permuted_copies=N
  lr=VALUE       -> learning_rate=VALUE
  model=NAME     -> model_name=NAME
  model_seeds=.. -> seeds=..
  n_gpus=N, gpus=N, num_cuda_gpus=N, cuda_gpus=N -> num_gpus=N
  GPU_CARDS="0,2" selects physical cards and, when num_gpus is omitted,
  determines the GPU count. Explicit seeds override GPU-derived seeds.
  GPU arguments schedule visible cards; they do not allocate cluster GPUs.

Models:
  smcn, mlp, count_mlp, dist_mlp, boundary_mlp, spec_mlp, gin, esan, homp, hasse, baselines, all

Example:
  bash fundamental_group.sh model_name=gin num_gpus=4
EOF
}

family_spec="fundamental_group"
model_spec="gin"
seed_spec=""
num_gpus=1
num_gpus_was_set=0
hydra_overrides=()

for argument in "$@"; do
    case "$argument" in
        -h|--help)
            usage
            exit 0 ;;
    esac
    [[ "$argument" == *=* ]] || {
        echo "Invalid argument '$argument'; use key=value with no spaces." >&2
        exit 2
    }
    case "$argument" in
        family=*) family_spec="${argument#*=}" ;;
        model_name=*|model=*) model_spec="${argument#*=}" ;;
        seeds=*|model_seeds=*) seed_spec="${argument#*=}" ;;
        num_gpus=*|n_gpus=*|gpus=*|num_cuda_gpus=*|cuda_gpus=*)
            num_gpus="${argument#*=}"
            num_gpus_was_set=1 ;;
        epochs=*) hydra_overrides+=("num_epochs=${argument#*=}") ;;
        copies=*) hydra_overrides+=("number_of_permuted_copies=${argument#*=}") ;;
        lr=*|learning_rate=*) hydra_overrides+=("learning_rate=${argument#*=}") ;;
        *) hydra_overrides+=("$argument") ;;
    esac
done

case "${family_spec,,}" in
    fundamental_group|fg|fg_genus_2|all) ;;
    *) echo "Unknown family: $family_spec; use family=fundamental_group." >&2; exit 2 ;;
esac

if [[ -n "${GPU_CARDS:-}" && "$num_gpus_was_set" == 0 ]]; then
    gpu_card_spec="${GPU_CARDS//,/ }"
    read -r -a configured_gpu_cards <<< "$gpu_card_spec"
    num_gpus="${#configured_gpu_cards[@]}"
fi
[[ "$num_gpus" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid num_gpus=${num_gpus}; use a positive integer." >&2; exit 2; }

model_seeds=()
if [[ -n "$seed_spec" ]]; then
    seed_spec="${seed_spec#[}"
    seed_spec="${seed_spec%]}"
    seed_spec="${seed_spec//,/ }"
    read -r -a model_seeds <<< "$seed_spec"
else
    for ((seed = 0; seed < num_gpus; seed++)); do
        model_seeds+=("$seed")
    done
fi
[[ ${#model_seeds[@]} -gt 0 ]] || { echo "seeds cannot be empty." >&2; exit 2; }
for seed in "${model_seeds[@]}"; do
    [[ "$seed" =~ ^[0-9]+$ ]] || { echo "Invalid model seed: $seed" >&2; exit 2; }
done

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
smcn_root="$(cd -- "${script_dir}/../../.." && pwd)"

python_has_hydra() {
    "$1" -c "import hydra" >/dev/null 2>&1
}

python_candidates=()
if [[ -n "${PYTHON:-}" ]]; then
    python_candidates+=("$PYTHON")
fi
if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python3" ]]; then
    python_candidates+=("${VIRTUAL_ENV}/bin/python3")
fi
if [[ -x "${smcn_root}/../smcn-env/bin/python3" ]]; then
    python_candidates+=("${smcn_root}/../smcn-env/bin/python3")
fi
python_candidates+=("$(command -v python3)")

python_bin=""
for candidate in "${python_candidates[@]}"; do
    if [[ -x "$candidate" ]] && python_has_hydra "$candidate"; then
        python_bin="$candidate"
        break
    fi
done
if [[ -z "$python_bin" ]]; then
    echo "Could not find a Python with hydra installed." >&2
    echo "Activate the SMCN environment or pass PYTHON=/path/to/python." >&2
    exit 2
fi
if ! "$python_bin" -c \
    'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)'; then
    echo "CUDA was requested, but PyTorch cannot access an NVIDIA GPU on this node." >&2
    echo "num_gpus schedules visible GPUs; it does not allocate them." >&2
    echo "Run fundamental_group.sh inside a GPU allocation." >&2
    exit 2
fi
output_root="${OUTPUT_ROOT:-runs}"
log_root="${LOG_ROOT:-logs}"
run_tag="${RUN_TAG:-$(date +%d%m_%H:%M)}"

model_spec="${model_spec#[}"
model_spec="${model_spec%]}"
model_spec="${model_spec//,/ }"
read -r -a requested_models <<< "$model_spec"
if [[ ${#requested_models[@]} -eq 0 ]]; then
    echo "model_name cannot be empty." >&2
    exit 2
fi

display_models=()
config_models=()
for requested in "${requested_models[@]}"; do
    normalized="${requested,,}"
    case "$normalized" in
        all)
            if [[ ${#requested_models[@]} -ne 1 ]]; then
                echo "Use model_name=all by itself." >&2
                exit 2
            fi
            display_models=(smcn mlp dist_mlp boundary_mlp spec_mlp gin homp hasse esan)
            config_models=(smcn mlp dist_mlp boundary_mlp spec_mlp gin homp_cin hasse_gin esan)
            break
            ;;
        baselines)
            if [[ ${#requested_models[@]} -ne 1 ]]; then
                echo "Use model_name=baselines by itself." >&2
                exit 2
            fi
            display_models=(mlp dist_mlp boundary_mlp spec_mlp gin esan homp hasse)
            config_models=(mlp dist_mlp boundary_mlp spec_mlp gin esan homp_cin hasse_gin)
            break
            ;;
        smcn|subcomplex)
            display_models+=(smcn)
            config_models+=(smcn)
            ;;
        mlp)
            display_models+=(mlp dist_mlp boundary_mlp)
            config_models+=(mlp dist_mlp boundary_mlp)
            ;;
        count|count_mlp)
            display_models+=(mlp)
            config_models+=(mlp)
            ;;
        dist|distance|distance_mlp|dist_mlp)
            display_models+=(dist_mlp)
            config_models+=(dist_mlp)
            ;;
        boundary|boundary_loop|boundary_loop_mlp|boundary_mlp)
            display_models+=(boundary_mlp)
            config_models+=(boundary_mlp)
            ;;
        spec|spectral|spectral_mlp|spec_mlp)
            display_models+=(spec_mlp)
            config_models+=(spec_mlp)
            ;;
        gin|esan)
            display_models+=("$normalized")
            config_models+=("$normalized")
            ;;
        homp|homp_cin)
            display_models+=(homp)
            config_models+=(homp_cin)
            ;;
        hasse|hasse_gin)
            display_models+=(hasse)
            config_models+=(hasse_gin)
            ;;
        *)
            echo "Unknown model: $requested" >&2
            echo "Choose from: smcn, mlp, count_mlp, dist_mlp, boundary_mlp, spec_mlp, gin, homp, hasse, esan, baselines, or all." >&2
            exit 2
            ;;
    esac
done

if [[ -n "${GPU_CARDS:-}" ]]; then
    gpu_card_spec="${GPU_CARDS//,/ }"
    read -r -a gpu_cards <<< "$gpu_card_spec"
    [[ ${#gpu_cards[@]} -ge "$num_gpus" ]] || {
        echo "GPU_CARDS provides ${#gpu_cards[@]} cards but num_gpus=${num_gpus}." >&2
        exit 2
    }
    gpu_cards=("${gpu_cards[@]:0:$num_gpus}")
else
    gpu_cards=()
    for ((gpu_id = 0; gpu_id < num_gpus; gpu_id++)); do
        gpu_cards+=("$gpu_id")
    done
fi
if [[ ${#gpu_cards[@]} -eq 0 ]]; then
    echo "No CUDA GPU cards configured." >&2
    exit 2
fi

export SMCN_ROOT="${SMCN_ROOT:-$smcn_root}"
export PYTHONPATH="$(dirname "$smcn_root"):$smcn_root:${script_dir}/..:$script_dir${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$script_dir"
mkdir -p "$output_root" "$log_root"

wait_for_seed_batch() {
    local display_model="$1"
    shift
    local -n pids_ref="$1"
    local -n seeds_ref="$2"
    local failed=0

    for seed_index in "${!pids_ref[@]}"; do
        local seed="${seeds_ref[$seed_index]}"
        if ! wait "${pids_ref[$seed_index]}"; then
            echo "${display_model} seed=${seed} failed; see logs." >&2
            failed=1
        fi
    done
    pids_ref=()
    seeds_ref=()
    return "$failed"
}

run_parallel_seeds() {
    local display_model="$1"
    local config_model="$2"
    local evaluator="$3"
    shift 3
    local evaluator_overrides=("$@")

    mkdir -p "${output_root}/${display_model}" "${log_root}/${display_model}"
    echo "Launching ${display_model}: model seeds ${model_seeds[*]}"

    local pids=()
    local launched_seeds=()
    for seed_index in "${!model_seeds[@]}"; do
        local seed="${model_seeds[$seed_index]}"
        local gpu="${gpu_cards[$((seed_index % ${#gpu_cards[@]}))]}"
        local run_name="seed${seed}_${run_tag}"
        local log_file="${log_root}/${display_model}/${run_name}.log"

        echo "Starting model=${display_model}, seed=${seed}, GPU=${gpu}"
        (
            CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" "$evaluator" \
                "${hydra_overrides[@]}" \
                "${evaluator_overrides[@]}" \
                "model_seed=${seed}" \
                "device=cuda" \
                2>&1 |
                sed -u "s/^/[${display_model} seed=${seed} gpu=${gpu}] /" |
                tee "$log_file"
        ) &
        pids+=("$!")
        launched_seeds+=("$seed")

        if (( ${#pids[@]} == ${#gpu_cards[@]} )); then
            if ! wait_for_seed_batch "$display_model" pids launched_seeds; then
                exit 1
            fi
        fi
    done
    if (( ${#pids[@]} > 0 )) && ! wait_for_seed_batch "$display_model" pids launched_seeds; then
        exit 1
    fi

    "$python_bin" ../../aggregate_runs.py "$output_root" "$display_model" --tag "$run_tag" |
        tee "${log_root}/${display_model}/summary_${run_tag}.log"
}

for model_index in "${!display_models[@]}"; do
    display_model="${display_models[$model_index]}"
    config_model="${config_models[$model_index]}"

    if [[ "$display_model" == "smcn" ]]; then
        sync_root="${SYNC_DIR:-${TMPDIR:-/tmp}/smcn_fg_${SLURM_JOB_ID:-local}_${run_tag}}"
        mkdir -p "$sync_root"
        echo "Building/validating SMCN cache if needed"
        CUDA_VISIBLE_DEVICES="" "$python_bin" run.py \
            "${hydra_overrides[@]}" \
            model_names=[smcn] \
            model_type=subcomplex \
            cache_only=true \
            model_seed=0

        mkdir -p "${output_root}/smcn" "${log_root}/smcn"
        for ((batch_start = 0; batch_start < ${#model_seeds[@]}; batch_start += ${#gpu_cards[@]})); do
            pids=()
            launched_seeds=()
            remaining=$((${#model_seeds[@]} - batch_start))
            batch_size="${#gpu_cards[@]}"
            if (( remaining < batch_size )); then
                batch_size="$remaining"
            fi
            sync_dir="${sync_root}/batch_${batch_start}"
            mkdir -p "$sync_dir"

            for ((batch_rank = 0; batch_rank < batch_size; batch_rank++)); do
                seed_index=$((batch_start + batch_rank))
                seed="${model_seeds[$seed_index]}"
                gpu="${gpu_cards[$batch_rank]}"
                run_name="seed${seed}_${run_tag}"
                output_dir="${output_root}/smcn/${run_name}"
                log_file="${log_root}/smcn/${run_name}.log"
                (
                    CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" run.py \
                        "${hydra_overrides[@]}" \
                        model_names=[smcn] \
                        model_type=subcomplex \
                        cache_only=false \
                        model_seed="$seed" \
                        sync_dir="$sync_dir" \
                        sync_world_size="$batch_size" \
                        sync_rank="$batch_rank" \
                        sync_timeout_seconds=1800 \
                        device=cuda \
                        output_dir="$output_dir" \
                        2>&1 |
                        sed -u "s/^/[smcn seed=${seed} gpu=${gpu}] /" |
                        tee "$log_file"
                ) &
                pids+=("$!")
                launched_seeds+=("$seed")
            done

            if ! wait_for_seed_batch smcn pids launched_seeds; then
                exit 1
            fi
        done
        "$python_bin" ../../aggregate_runs.py "$output_root" smcn --tag "$run_tag" |
            tee "${log_root}/smcn/summary_${run_tag}.log"
    else
        echo "Building/validating baseline cache if needed for ${display_model}"
        CUDA_VISIBLE_DEVICES="" "$python_bin" run.py \
            "${hydra_overrides[@]}" \
            "model_names=[${config_model}]" \
            include_subcomplex_features=false \
            prune_to_baseline_fields=true \
            compact_subcomplex_cache=true \
            pin_memory=false \
            cache_only=true

        run_parallel_seeds \
            "$display_model" \
            "$config_model" \
            run.py \
            "model_names=[${config_model}]" \
            "output_dir=${output_root}" \
            "run_model_name=${display_model}" \
            "run_tag=${run_tag}" \
            include_subcomplex_features=false \
            prune_to_baseline_fields=true \
            compact_subcomplex_cache=true \
            pin_memory=false \
            cache_only=false
    fi
done

echo "All requested fundamental-group model runs completed successfully."
