#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash orientability.sh [family=<cylinder_vs_mobius|torus_vs_klein|punc_torus_vs_punc_klein|all>] [model_name=<model|model,...>] [key=value ...]

Defaults:
  family=cylinder_vs_mobius model_name=gin num_gpus=1
  GPU_CARDS sets the GPU count when num_gpus is omitted.
  Without seeds=..., seeds are 0..GPU_COUNT-1.

Friendly aliases:
  epochs=N       -> num_epochs=N
  copies=N       -> number_of_permuted_copies=N
  lr=VALUE       -> learning_rate=VALUE
  model=NAME     -> model_name=NAME
  model_seeds=.. -> seeds=..
  n_gpus=N, gpus=N, num_cuda_gpus=N -> num_gpus=N
  GPU_CARDS="0,2" selects physical cards and, when num_gpus is omitted,
  determines the GPU count. Explicit seeds override GPU-derived seeds.
  GPU arguments schedule visible cards; they do not allocate cluster GPUs.

Models:
  smcn, mlp, count_mlp, dist_mlp, boundary_mlp, gin, esan, homp, hasse, baselines, all

Example:
  bash orientability.sh family=torus_vs_klein model_name=gin num_gpus=4
EOF
}

family_spec="cylinder_vs_mobius"
model_spec="gin"
overrides=()
seed_spec=""
num_gpus=1
num_gpus_was_set=0
device_was_set=0
requested_device="cuda"
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
        num_gpus=*|n_gpus=*|gpus=*|num_cuda_gpus=*)
            num_gpus="${argument#*=}"
            num_gpus_was_set=1 ;;
        epochs=*) overrides+=("num_epochs=${argument#*=}") ;;
        copies=*) overrides+=("number_of_permuted_copies=${argument#*=}") ;;
        lr=*|learning_rate=*)
            value="${argument#*=}"
            overrides+=("learning_rate=${value}") ;;
        device=*)
            device_was_set=1
            requested_device="${argument#*=}"
            overrides+=("$argument") ;;
        *) overrides+=("$argument") ;;
    esac
done
if [[ -n "${GPU_CARDS:-}" && "$num_gpus_was_set" == 0 ]]; then
    gpu_card_spec="${GPU_CARDS//,/ }"
    read -r -a configured_gpu_cards <<< "$gpu_card_spec"
    num_gpus="${#configured_gpu_cards[@]}"
fi
[[ "$num_gpus" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid num_gpus=${num_gpus}; use a positive integer." >&2; exit 2; }
if [[ "$device_was_set" == 0 ]]; then
    overrides+=(device=cuda)
fi

seeds=()
if [[ -n "$seed_spec" ]]; then
    seed_spec="${seed_spec#[}"
    seed_spec="${seed_spec%]}"
    seed_spec="${seed_spec//,/ }"
    read -r -a seeds <<< "$seed_spec"
else
    for ((seed = 0; seed < num_gpus; seed++)); do
        seeds+=("$seed")
    done
fi
[[ ${#seeds[@]} -gt 0 ]] || { echo "seeds cannot be empty." >&2; exit 2; }
for seed in "${seeds[@]}"; do
    [[ "$seed" =~ ^[0-9]+$ ]] || { echo "Invalid model seed: $seed" >&2; exit 2; }
done

family_spec="${family_spec#[}"
family_spec="${family_spec%]}"
family_spec="${family_spec//,/ }"
read -r -a requested_families <<< "$family_spec"
families=()
for family in "${requested_families[@]}"; do
    case "${family,,}" in
        all)
            [[ ${#requested_families[@]} -eq 1 ]] || { echo "Use family=all alone." >&2; exit 2; }
            families=(cylinder_vs_mobius torus_vs_klein punc_torus_vs_punc_klein)
            break ;;
        cylinder|cyl|mobius|cylinder_vs_mobius) families+=(cylinder_vs_mobius) ;;
        torus|klein|closed|closed_torus_klein|torus_vs_klein) families+=(torus_vs_klein) ;;
        punc|punctured|punc_torus|punc_torus_vs_punc_klein|punctured_torus_vs_klein) families+=(punc_torus_vs_punc_klein) ;;
        *) echo "Unknown family: $family" >&2; exit 2 ;;
    esac
done

model_spec="${model_spec#[}"
model_spec="${model_spec%]}"
model_spec="${model_spec//,/ }"
read -r -a requested_models <<< "$model_spec"
display_models=()
config_models=()
for item in "${requested_models[@]}"; do
    case "${item,,}" in
        all)
            [[ ${#requested_models[@]} -eq 1 ]] || { echo "Use model_name=${item} alone." >&2; exit 2; }
            display_models=(smcn mlp dist_mlp boundary_mlp gin esan homp hasse)
            config_models=(smcn mlp dist_mlp boundary_mlp gin esan homp_cin hasse_gin)
            break ;;
        baselines)
            [[ ${#requested_models[@]} -eq 1 ]] || { echo "Use model_name=${item} alone." >&2; exit 2; }
            display_models=(mlp dist_mlp boundary_mlp gin esan homp hasse)
            config_models=(mlp dist_mlp boundary_mlp gin esan homp_cin hasse_gin)
            break ;;
        mlp)
            display_models+=(mlp dist_mlp boundary_mlp)
            config_models+=(mlp dist_mlp boundary_mlp) ;;
        count|count_mlp) display_models+=(mlp); config_models+=(mlp) ;;
        gin|esan|smcn|dist_mlp|boundary_mlp) display_models+=("${item,,}"); config_models+=("${item,,}") ;;
        subcomplex) display_models+=(smcn); config_models+=(smcn) ;;
        dist|distance|distance_mlp) display_models+=(dist_mlp); config_models+=(dist_mlp) ;;
        boundary|boundary_loop|boundary_loop_mlp) display_models+=(boundary_mlp); config_models+=(boundary_mlp) ;;
        homp|homp_cin) display_models+=(homp); config_models+=(homp_cin) ;;
        hasse|hasse_gin) display_models+=(hasse); config_models+=(hasse_gin) ;;
        *) echo "Unknown model: $item" >&2; exit 2 ;;
    esac
done
[[ ${#families[@]} -gt 0 ]] || { echo "No family selected." >&2; exit 2; }
[[ ${#display_models[@]} -gt 0 ]] || { echo "No model selected." >&2; exit 2; }

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
new_experiments_root="$(cd -- "$script_dir/.." && pwd)"
project_root="$(cd -- "$script_dir/../.." && pwd)"
aggregate_script="$new_experiments_root/aggregate_runs.py"
generate_script="$project_root/data/get_base_pairs.py"
python_bin="${PYTHON:-python3}"
run_tag="${RUN_TAG:-$(date +%d%m_%H:%M)}"
if [[ "$requested_device" == cuda* ]] && ! "$python_bin" -c \
    'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)'; then
    echo "CUDA was requested, but PyTorch cannot access an NVIDIA GPU on this node." >&2
    echo "num_gpus schedules visible GPUs; it does not allocate them." >&2
    echo "Run inside a GPU allocation, or use device=cpu num_gpus=1." >&2
    exit 2
fi
if [[ -n "${GPU_CARDS:-}" ]]; then
    gpu_card_spec="${GPU_CARDS//,/ }"
    read -r -a gpus <<< "$gpu_card_spec"
    [[ ${#gpus[@]} -ge "$num_gpus" ]] || {
        echo "GPU_CARDS provides ${#gpus[@]} cards but num_gpus=${num_gpus}." >&2
        exit 2
    }
    gpus=("${gpus[@]:0:$num_gpus}")
else
    gpus=()
    for ((gpu_id = 0; gpu_id < num_gpus; gpu_id++)); do
        gpus+=("$gpu_id")
    done
fi
export PYTHONPATH="$project_root:$script_dir:$script_dir/..${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
cd "$script_dir"
[[ -f "$aggregate_script" ]] || { echo "Missing aggregate script: $aggregate_script" >&2; exit 2; }

successful_runs=()
failed_runs=()

for family in "${families[@]}"; do
    echo "Validating/generating base pairs for ${family}"
    "$python_bin" "$generate_script" --family "$family"
    output_root="${family}/runs"
    log_root="${family}/logs"
    for index in "${!display_models[@]}"; do
        model="${display_models[$index]}"
        config_model="${config_models[$index]}"
        mkdir -p "$output_root/$model" "$log_root/$model"
        echo "Launching family=${family} model=${model} seeds ${seeds[*]} with num_gpus=${num_gpus} cards: ${gpus[*]}"
        failed=0
        pids=()
        for seed_index in "${!seeds[@]}"; do
            seed="${seeds[$seed_index]}"
            gpu="${gpus[$((seed_index % num_gpus))]}"
            run_dir="$output_root/$model/seed${seed}_${run_tag}"
            run_dir_abs="$script_dir/$run_dir"
            log_file="$log_root/$model/seed${seed}_${run_tag}.log"
            (
                CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" run.py \
                    "family=${family}" \
                    "${overrides[@]}" \
                    "model_names=[${config_model}]" \
                    "model_seed=${seed}" \
                    "output_dir=${run_dir_abs}" 2>&1 |
                sed -u "s/^/[${family} ${model} seed=${seed} gpu=${gpu}] /" |
                tee "$log_file"
            ) &
            pids+=("$!")
            if (( ${#pids[@]} >= num_gpus )); then
                for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
                pids=()
            fi
        done
        for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
        if (( failed != 0 )); then
            echo "family=${family} model=${model} failed; skipping to next requested model." >&2
            failed_runs+=("${family}/${model}")
            continue
        fi
        if "$python_bin" "$aggregate_script" "$output_root" "$model" \
            --tag "$run_tag" | tee "$log_root/$model/summary_${run_tag}.log"; then
            successful_runs+=("${family}/${model}")
        else
            echo "family=${family} model=${model} finished seeds but aggregation failed; skipping to next requested model." >&2
            failed_runs+=("${family}/${model}:aggregation")
        fi
    done
done

echo
echo "Orientability baseline launcher summary"
echo "Run tag: ${run_tag}"
if ((${#successful_runs[@]})); then
    echo "Completed:"
    for item in "${successful_runs[@]}"; do echo "  - $item"; done
else
    echo "Completed: none"
fi
if ((${#failed_runs[@]})); then
    echo "Failed:"
    for item in "${failed_runs[@]}"; do echo "  - $item"; done
    exit 1
fi
echo "Failed: none"
echo "All requested orientability baseline groups completed successfully."
