#!/usr/bin/env bash
set -euo pipefail

SELF_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${PYTHON:-}" ]]; then
    python_executable="$PYTHON"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python3" ]]; then
    python_executable="${VIRTUAL_ENV}/bin/python3"
elif [[ -x "${SELF_DIR}/../../../../smcn-env/bin/python3" ]]; then
    python_executable="${SELF_DIR}/../../../../smcn-env/bin/python3"
elif [[ -x "${SELF_DIR}/../../../../gpu-env/bin/python3" ]]; then
    python_executable="${SELF_DIR}/../../../../gpu-env/bin/python3"
else
    echo "No compatible project Python found. Activate smcn-env or set PYTHON." >&2
    exit 2
fi

num_gpus=1
num_gpus_was_set=0
seed_spec=""
control_family=""
hydra_overrides=()

for argument in "$@"; do
    [[ "$argument" == *=* ]] || {
        echo "Invalid argument '$argument'; use key=value." >&2
        exit 2
    }
    case "$argument" in
        num_gpus=*|n_gpus=*|gpus=*)
            num_gpus="${argument#*=}"
            num_gpus_was_set=1 ;;
        seeds=*|model_seeds=*) seed_spec="${argument#*=}" ;;
        model_seed=*) seed_spec="${argument#*=}" ;;
        control_family=*|+control_family=*)
            control_family="${argument#*=}"
            hydra_overrides+=("control_family=${argument#*=}") ;;
        device=*|output_dir=*|run_model_name=*|cache_only=*)
            echo "$argument is managed by run_spec_mlp.sh." >&2
            exit 2 ;;
        *) hydra_overrides+=("$argument") ;;
    esac
done

[[ -n "$control_family" ]] || {
    echo "Missing control_family (surface_vs_surface or free_vs_free)." >&2
    exit 2
}

if [[ -n "${GPU_CARDS:-}" && "$num_gpus_was_set" == 0 ]]; then
    gpu_spec="${GPU_CARDS//,/ }"
    read -r -a configured_cards <<< "$gpu_spec"
    num_gpus="${#configured_cards[@]}"
fi
[[ "$num_gpus" =~ ^[1-9][0-9]*$ ]] || {
    echo "num_gpus must be a positive integer." >&2
    exit 2
}

if [[ -n "${GPU_CARDS:-}" ]]; then
    gpu_spec="${GPU_CARDS//,/ }"
    read -r -a gpu_cards <<< "$gpu_spec"
    [[ ${#gpu_cards[@]} -ge "$num_gpus" ]] || {
        echo "GPU_CARDS supplies ${#gpu_cards[@]} cards, but num_gpus=${num_gpus}." >&2
        exit 2
    }
    gpu_cards=("${gpu_cards[@]:0:$num_gpus}")
else
    gpu_cards=()
    for ((gpu=0; gpu<num_gpus; gpu++)); do
        gpu_cards+=("$gpu")
    done
fi

model_seeds=()
if [[ -n "$seed_spec" ]]; then
    seed_spec="${seed_spec#[}"
    seed_spec="${seed_spec%]}"
    seed_spec="${seed_spec//,/ }"
    read -r -a model_seeds <<< "$seed_spec"
else
    for ((seed=0; seed<num_gpus; seed++)); do
        model_seeds+=("$seed")
    done
fi
[[ ${#model_seeds[@]} -gt 0 ]] || {
    echo "seeds cannot be empty." >&2
    exit 2
}
for seed in "${model_seeds[@]}"; do
    [[ "$seed" =~ ^[0-9]+$ ]] || {
        echo "Invalid model seed: $seed" >&2
        exit 2
    }
done

family_tag="${control_family:-standard}"
output_root="${OUTPUT_ROOT:-${SELF_DIR}/runs_control_${family_tag}_spec}"
log_root="${LOG_ROOT:-${SELF_DIR}/logs_control_${family_tag}_spec}"
mkdir -p "$output_root" "$log_root"

echo "Preparing cache for ${family_tag}"
CUDA_VISIBLE_DEVICES="" "$python_executable" "${SELF_DIR}/run_spec_mlp.py" \
    "${hydra_overrides[@]}" \
    cache_only=true device=cpu \
    "output_dir=${output_root}"

visible_cards="$(IFS=,; echo "${gpu_cards[*]}")"
if ! CUDA_VISIBLE_DEVICES="$visible_cards" "$python_executable" -c \
    'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
    echo "CUDA is unavailable to the selected Python process or GPU allocation." >&2
    exit 2
fi

echo "Launching SPEC_MLP seeds ${model_seeds[*]} on GPUs ${gpu_cards[*]}"
for ((wave_start=0; wave_start<${#model_seeds[@]}; wave_start+=num_gpus)); do
    pids=()
    wave_seeds=()
    for ((rank=0; rank<num_gpus && wave_start+rank<${#model_seeds[@]}; rank++)); do
        seed="${model_seeds[$((wave_start+rank))]}"
        gpu="${gpu_cards[$rank]}"
        log_file="${log_root}/seed${seed}.log"
        echo "Starting seed=${seed} on GPU=${gpu}"
        (
            CUDA_VISIBLE_DEVICES="$gpu" "$python_executable" \
                "${SELF_DIR}/run_spec_mlp.py" \
                "${hydra_overrides[@]}" \
                cache_only=false device=cuda "model_seed=${seed}" \
                "output_dir=${output_root}" \
                2>&1 | sed -u "s/^/[spec_mlp seed=${seed} gpu=${gpu}] /" | tee "$log_file"
        ) &
        pids+=("$!")
        wave_seeds+=("$seed")
    done
    failed=0
    for index in "${!pids[@]}"; do
        if ! wait "${pids[$index]}"; then
            echo "SPEC_MLP seed=${wave_seeds[$index]} failed; see its log." >&2
            failed=1
        fi
    done
    (( failed == 0 )) || exit 1
done

echo "SPEC_MLP runs complete: ${output_root}"
