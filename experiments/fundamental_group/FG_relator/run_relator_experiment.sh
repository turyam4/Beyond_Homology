#!/usr/bin/env bash
# =============================================================================
# PHASE 5 driver -- held-out relator family (deep_commutator vs free F_4)
#
#   bash run_relator_experiment.sh                 # full: caches + all models
#   MODELS="smcn" SEEDS="0 1 2 3" NUM_GPUS=4 bash run_relator_experiment.sh
#   PAIRING=surface_vs_deep bash run_relator_experiment.sh
#   DRY=1 bash run_relator_experiment.sh
# =============================================================================
set -euo pipefail
SELF="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python3" ]]; then
    PYTHON="${VIRTUAL_ENV}/bin/python3"
  elif [[ -x "${SELF}/../../../../smcn-env/bin/python3" ]]; then
    PYTHON="${SELF}/../../../../smcn-env/bin/python3"
  elif [[ -x "${SELF}/../../../../gpu-env/bin/python3" ]]; then
    PYTHON="${SELF}/../../../../gpu-env/bin/python3"
  else
    PYTHON="$(command -v python3)"
  fi
fi
PAIRING="${PAIRING:-deep_vs_free}"
SEEDS="${SEEDS:-0 1 2 3}"
SAMPLES="${SAMPLES:-16}"
COPIES="${COPIES:-32}"
EPOCHS="${EPOCHS:-20}"
BATCH="${BATCH:-8}"
MODELS="${MODELS:-spec_mlp baselines smcn}"
DRY="${DRY:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
DEVICE="${DEVICE:-auto}"
CACHE="${CACHE:-$SELF/fg_pairs_rel}"
OUT="${OUT:-$SELF/runs_rel}"

[[ -n "$PYTHON" && -x "$PYTHON" ]] || {
  echo "Error: python3 was not found. Activate the intended environment or set PYTHON." >&2
  exit 1
}
[[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]] || {
  echo "Error: NUM_GPUS must be a positive integer." >&2
  exit 1
}

run(){ echo "+ $*"; [[ "$DRY" == "1" ]] || "$@"; }
COMMON=( "samples_per_g=$SAMPLES" "number_of_permuted_copies=$COPIES"
         "batch_size=$BATCH" "eval_batch_size=$BATCH" "num_epochs=$EPOCHS"
         "device=$DEVICE" )

cd "$SELF"
echo "### pairing=$PAIRING seeds=$SEEDS"

# ---- caches (binary + distance marking) ----
for MARK in true false; do
  run "$PYTHON" run_relator.py "pairing=$PAIRING" cache_only=true \
      "binary_marking=$MARK" "${COMMON[@]}" "dataset_cache_dir=${CACHE}_${MARK}"
done

run_model_seed() {
  local M="$1" SEED="$2" GPU="$3"
  export CUDA_VISIBLE_DEVICES="$GPU"
    case "$M" in
      spec_mlp)
        run "$PYTHON" run_relator.py "pairing=$PAIRING" 'model_names=[spec_mlp]' \
            "model_seed=$SEED" +spec_k=48 +spec_moment_L=12 "${COMMON[@]}" \
            "dataset_cache_dir=${CACHE}_true" "output_dir=$OUT/$PAIRING/spec_mlp/seed$SEED" ;;
      baselines)
        run "$PYTHON" run_relator.py "pairing=$PAIRING" \
            'model_names=[gin,hasse_gin,homp_cin,mlp,dist_mlp,boundary_mlp]' \
            "model_seed=$SEED" "${COMMON[@]}" \
            "dataset_cache_dir=${CACHE}_true" "output_dir=$OUT/$PAIRING/baselines/seed$SEED" ;;
      smcn)
        for MARK in true false; do
          TAG=$([[ "$MARK" == "true" ]] && echo binary || echo spd)
          run "$PYTHON" run_relator.py "pairing=$PAIRING" 'model_names=[smcn]' \
              "binary_marking=$MARK" "model_seed=$SEED" "${COMMON[@]}" \
              "dataset_cache_dir=${CACHE}_${MARK}" \
              "output_dir=$OUT/$PAIRING/smcn_$TAG/seed$SEED"
        done ;;
      *) run "$PYTHON" run_relator.py "pairing=$PAIRING" "model_names=[$M]" \
            "model_seed=$SEED" "${COMMON[@]}" \
            "dataset_cache_dir=${CACHE}_true" "output_dir=$OUT/$PAIRING/$M/seed$SEED" ;;
    esac
}

for M in $MODELS; do
  seed_index=0
  wave_pids=()
  for SEED in $SEEDS; do
    gpu=$((seed_index % NUM_GPUS))
    run_model_seed "$M" "$SEED" "$gpu" &
    wave_pids+=("$!")
    seed_index=$((seed_index + 1))

    if (( ${#wave_pids[@]} == NUM_GPUS )); then
      for pid in "${wave_pids[@]}"; do wait "$pid"; done
      wave_pids=()
    fi
  done
  for pid in "${wave_pids[@]}"; do wait "$pid"; done
done
echo "Done. Aggregate: find $OUT -name metrics.json"
