#!/usr/bin/env bash
# =============================================================================
# E-G driver: run pi_1-matched control families with the FG memory-managed engine.
#
# Fills the three rows of the R3 table:
#     surface vs. free   (cross-class)  <- you already have this
#     surface vs. surface (pi_1-matched)  <- this script
#     free vs. free       (pi_1-matched)  <- this script
#
# Runs, in this order:
#     smcn binary_marking=true   over all requested families
#     smcn binary_marking=false  over all requested families
#     esan                       over all requested families
#   + optionally the full baseline ladder with LADDER=1
#
# over the SAME 4 model seeds as the cross-class runs.
#
# Usage:
#   bash run_control_family.sh                       # both control families, one seed
#   bash run_control_family.sh num_gpus=4            # seeds 0..3 in parallel
#   bash run_control_family.sh seeds=[0,1,2,3] num_gpus=2
#   bash run_control_family.sh control_family=surface_vs_surface
#   bash run_control_family.sh control_family=free_vs_free
#   bash run_control_family.sh ladder=1              # + gin/homp/hasse/mlps
#   DRY=1 bash run_control_family.sh                 # print commands only
#
# Env:
#   PYTHON     python with torch/torch_geometric/toponetx/sympy
#   SEEDS      model seeds            (default 0..num_gpus-1)
#   CONTROL_FAMILY families to run     (default both)
#   NUM_GPUS   CUDA GPUs available    (default 1)
#   SAMPLES    samples_per_g          (default 16, matches cross-class)
#   COPIES     permuted copies        (default 32)
#   EPOCHS     num_epochs             (default 20)
#   BATCH      batch size             (default 8)
# =============================================================================
set -euo pipefail

SELF_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
seed_spec="${SEEDS:-}"
CONTROL_FAMILY="${CONTROL_FAMILY:-both}"
SAMPLES="${SAMPLES:-16}"
COPIES="${COPIES:-32}"
EPOCHS="${EPOCHS:-20}"
BATCH="${BATCH:-8}"
LADDER="${LADDER:-0}"
DRY="${DRY:-0}"
DEVICE="${DEVICE:-cuda}"
num_gpus="${NUM_GPUS:-1}"
num_gpus_was_set=0
hydra_extra=()

for argument in "$@"; do
    [[ "$argument" == *=* ]] || {
        echo "Invalid argument '$argument'; use key=value with no spaces." >&2
        exit 2
    }
    case "$argument" in
        seeds=*|model_seeds=*) seed_spec="${argument#*=}" ;;
        families=*|control_families=*)
            echo "Use control_family=both, control_family=surface_vs_surface, or control_family=free_vs_free." >&2
            exit 2 ;;
        control_family=*) CONTROL_FAMILY="${argument#*=}" ;;
        samples=*|samples_per_g=*) SAMPLES="${argument#*=}" ;;
        copies=*|number_of_permuted_copies=*) COPIES="${argument#*=}" ;;
        epochs=*|num_epochs=*) EPOCHS="${argument#*=}" ;;
        batch=*|batch_size=*) BATCH="${argument#*=}" ;;
        ladder=*|LADDER=*) LADDER="${argument#*=}" ;;
        dry=*|DRY=*) DRY="${argument#*=}" ;;
        num_gpus=*|n_gpus=*|gpus=*|num_cuda_gpus=*|cuda_gpus=*)
            num_gpus="${argument#*=}"
            num_gpus_was_set=1 ;;
        *)
            hydra_extra+=("$argument") ;;
    esac
done

case "$CONTROL_FAMILY" in
    both)
        FAMILIES="surface_vs_surface free_vs_free" ;;
    surface_vs_surface|free_vs_free)
        FAMILIES="$CONTROL_FAMILY" ;;
    *)
        echo "control_family must be one of: both, surface_vs_surface, free_vs_free." >&2
        exit 2 ;;
esac

if [[ -n "${GPU_CARDS:-}" && "$num_gpus_was_set" == 0 ]]; then
    gpu_card_spec="${GPU_CARDS//,/ }"
    read -r -a configured_gpu_cards <<< "$gpu_card_spec"
    num_gpus="${#configured_gpu_cards[@]}"
fi
[[ "$num_gpus" =~ ^[1-9][0-9]*$ ]] || {
    echo "num_gpus must be a positive integer." >&2
    exit 2
}

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

if [[ -z "${PYTHON:-}" ]]; then
    if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python3" ]]; then
        PYTHON="${VIRTUAL_ENV}/bin/python3"
    elif [[ -x "${SELF_DIR}/../../../../smcn-env/bin/python3" ]]; then
        PYTHON="${SELF_DIR}/../../../../smcn-env/bin/python3"
    elif [[ -x "${SELF_DIR}/../../../../gpu-env/bin/python3" ]]; then
        PYTHON="${SELF_DIR}/../../../../gpu-env/bin/python3"
    else
        PYTHON="$(command -v python3)"
    fi
fi

run() {
    if [[ "$DRY" == "1" ]]; then
        echo "DRY: $1"
        return 0
    fi
    shift
    "$@"
}

COMMON=(
    "samples_per_g=${SAMPLES}"
    "number_of_permuted_copies=${COPIES}"
    "batch_size=${BATCH}"
    "eval_batch_size=${BATCH}"
    "num_epochs=${EPOCHS}"
    "cache_as_minibatches=true"
    "construct_cache_by_phase=true"
    "compact_subcomplex_cache=true"
    "pin_memory=false"
    "${hydra_extra[@]}"
)

BASELINE_MEMORY=(
    "include_subcomplex_features=false"
    "prune_to_baseline_fields=true"
)

cd "$SELF_DIR"

wait_for_seed_batch() {
    local label="$1"
    shift
    local -n pids_ref="$1"
    local -n seeds_ref="$2"
    local failed=0

    for seed_index in "${!pids_ref[@]}"; do
        local seed="${seeds_ref[$seed_index]}"
        if ! wait "${pids_ref[$seed_index]}"; then
            echo "${label} seed=${seed} failed." >&2
            failed=1
        fi
    done
    pids_ref=()
    seeds_ref=()
    return "$failed"
}

run_seed_wave() {
    local family="$1"
    local label="$2"
    local model_names="$3"
    local output_prefix="$4"
    local sync_prefix="$5"
    shift 5
    local extra_overrides=("$@")

    echo "Launching ${family}/${label}: seeds ${model_seeds[*]} on GPUs ${gpu_cards[*]}"
    for ((batch_start = 0; batch_start < ${#model_seeds[@]}; batch_start += ${#gpu_cards[@]})); do
        local pids=()
        local launched_seeds=()
        local remaining=$((${#model_seeds[@]} - batch_start))
        local wave_size="${#gpu_cards[@]}"
        if (( remaining < wave_size )); then
            wave_size="$remaining"
        fi
        local sync_dir=""
        if [[ -n "$sync_prefix" ]]; then
            sync_dir="${SYNC_DIR:-${TMPDIR:-/tmp}/fg_control_${SLURM_JOB_ID:-local}}/${sync_prefix}_batch_${batch_start}"
            mkdir -p "$sync_dir"
        fi

        for ((batch_rank = 0; batch_rank < wave_size; batch_rank++)); do
            local seed_index=$((batch_start + batch_rank))
            local seed="${model_seeds[$seed_index]}"
            local gpu="${gpu_cards[$batch_rank]}"
            local output_dir="${output_prefix}/seed${seed}"

            echo "Starting ${family}/${label}: seed=${seed}, GPU=${gpu}"
            (
                overrides=(
                    "control_family=${family}"
                    "model_names=${model_names}"
                    "model_seed=${seed}"
                    "device=${DEVICE}"
                    "${COMMON[@]}"
                    "${extra_overrides[@]}"
                    "output_dir=${output_dir}"
                )
                if [[ -n "$sync_dir" ]]; then
                    overrides+=(
                        "sync_dir=${sync_dir}"
                        "sync_world_size=${wave_size}"
                        "sync_rank=${batch_rank}"
                        "sync_timeout_seconds=1800"
                    )
                fi
                if [[ "$DRY" == "1" ]]; then
                    echo "DRY: ${family}/${label} seed=${seed} gpu=${gpu}"
                else
                    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" run_control.py "${overrides[@]}"
                fi
            ) &
            pids+=("$!")
            launched_seeds+=("$seed")
        done

        if ! wait_for_seed_batch "${family}/${label}" pids launched_seeds; then
            exit 1
        fi
    done
}

build_smcn_cache() {
    local family="$1"
    local marking="$2"
    run "build ${family} smcn binary_marking=${marking}" \
        "$PYTHON" run_control.py \
        "control_family=${family}" cache_only=true \
        'model_names=[smcn]' \
        "binary_marking=${marking}" "${COMMON[@]}"
}

build_baseline_cache() {
    local family="$1"
    run "build ${family} baseline cache" \
        "$PYTHON" run_control.py \
        "control_family=${family}" cache_only=true \
        'model_names=[esan]' \
        "${COMMON[@]}" \
        "${BASELINE_MEMORY[@]}"
}

run_smcn_marking_for_families() {
    local marking="$1"
    local tag
    tag=$([[ "$marking" == "true" ]] && echo "binary" || echo "spd")
    for FAMILY in $FAMILIES; do
        echo
        echo "############################################################"
        echo "### CONTROL FAMILY: ${FAMILY} | SMCN ${tag}"
        echo "############################################################"
        build_smcn_cache "$FAMILY" "$marking"
        run_seed_wave \
            "$FAMILY" \
            "smcn_${tag}" \
            "[smcn]" \
            "runs_control_${FAMILY}/smcn_${tag}" \
            "${FAMILY}_smcn_${tag}" \
            "binary_marking=${marking}"
    done
}

run_esan_for_families() {
    for FAMILY in $FAMILIES; do
        echo
        echo "############################################################"
        echo "### CONTROL FAMILY: ${FAMILY} | ESAN"
        echo "############################################################"
        build_baseline_cache "$FAMILY"
        run_seed_wave \
            "$FAMILY" \
            "esan" \
            "[esan]" \
            "runs_control_${FAMILY}/esan" \
            "" \
            "${BASELINE_MEMORY[@]}"
    done
}

run_baseline_ladder_for_families() {
    for FAMILY in $FAMILIES; do
        echo
        echo "############################################################"
        echo "### CONTROL FAMILY: ${FAMILY} | BASELINE LADDER"
        echo "############################################################"
        build_baseline_cache "$FAMILY"
        run_seed_wave \
            "$FAMILY" \
            "baselines" \
            "[gin,homp_cin,hasse_gin,mlp,dist_mlp,boundary_mlp,spec_mlp]" \
            "runs_control_${FAMILY}/baselines" \
            "" \
            "${BASELINE_MEMORY[@]}"
    done
}

run_smcn_marking_for_families true
run_smcn_marking_for_families false
run_esan_for_families

if [[ "$LADDER" == "1" ]]; then
    run_baseline_ladder_for_families
fi

echo
echo "Done. Aggregate with:"
echo "  python aggregate_control.py"
