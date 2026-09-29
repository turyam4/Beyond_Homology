# SMCN depth sweep: original punctured orientability pairs

This directory contains the authoritative depth sweep for the original
punctured torus versus punctured Klein bottle benchmark. It reads code and data
only from this `Beyond_Homology` repository.

## Protocol

- Data: `data/punc_torus_vs_punc_klein`
- Pairs: all 50 combinations of `h=4,...,13` and `p in {4,6,8,10,12}`
- Construction: original girth-matched paper grid, not the later fixed-`p`
  distance-histogram-matched family
- Model: repository-local SMCN
- Subgraph depths: `L=1,2,3,4,5,6,7,8`
- Seeds: `0,1,2`
- Permutation copies: 32
- Epochs: 20
- Minibatch size: 16
- Binary subcomplex marking: enabled
- Fresh model for every pair: enabled
- CIN layers: 0

All other model and BREC settings come from
`experiments/orientability/config.yaml`.

## Run

Run inside a GPU allocation:

```bash
cd "$BEYOND_HOMOLOGY_ROOT/experiments/orientability/depth_sweep_orientability"
./run_depth_sweep.sh
```

Submit the complete 24-task Slurm array with:

```bash
sbatch slurm/submit_depth_sweep.slurm
```

One run is launched at a time by default. To use multiple allocated GPUs:

```bash
GPU_CARDS=0,1 JOBS=2 ./run_depth_sweep.sh
```

The launcher is resumable: a depth/seed combination with both
`pair_results.csv` and `metrics.json` is skipped.

## Outputs

```text
runs/L<depth>_seed<seed>/
logs/L<depth>_seed<seed>.log
slurm/slurm-<job>_<task>-depth_orient.out
slurm/slurm-<job>_<task>-depth_orient.err
```

After all runs finish:

```bash
"${PYTHON:-python3}" aggregate_depth_sweep.py
```

Historical outputs under `depth_sweeps/` and `depth_papergrid_out/` are
retained for provenance, but the top-level launcher does not read or extend
them.
