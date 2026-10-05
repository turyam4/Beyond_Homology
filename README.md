# Topological Benchmark Experiments

This repository contains benchmark code for testing whether graph and cellular
neural networks can distinguish topological objects that are intentionally
matched on simple local statistics.

Run the commands below from the `Beyond_Homology` repository root unless a
section explicitly changes directory.

## Repository Structure

```text
Beyond_Homology/
├── baselines/
│   ├── boundary_mlp/
│   ├── count_mlp/
│   ├── dist_mlp/
│   ├── esan/
│   ├── gin/
│   ├── hasse/
│   ├── cin/
│   └── smcn/
├── data/
│   ├── complex_data.py
│   ├── complex_dataset.py
│   ├── get_base_pairs.py
│   └── utils.py
└── experiments/
    ├── aggregate_runs.py
    ├── orientability/
    │   ├── cylinder_vs_mobius/
    │   ├── torus_vs_klein/
    │   └── punc_torus_vs_punc_klein/
    ├── fundamental_group/
    │   └── FG_genus_2/
    └── torus_vs_sphere/
```

The main experiments are:

```text
experiments/orientability/
experiments/fundamental_group/
experiments/torus_vs_sphere/
```

The evaluation protocol is BREC-style: each base pair is expanded into
permuted copies, the model is trained to separate the target pair, and the run
is accepted only when the target separates while the reliability pair does not
falsely separate.

## Requirements
The experiments were developed and run with Python 3.13.Use Python 3.10 or newer.  A typical setup also
needs PyTorch, PyTorch Geometric, Hydra/OmegaConf, NumPy, SciPy, TopoNetX, and
Matplotlib for plotting.

## Benchmarks

### Orientability

The orientability benchmark has three family members:

```text
cylinder_vs_mobius
torus_vs_klein
punc_torus_vs_punc_klein
```

These are designed to test whether a model can distinguish orientable from
non-orientable complexes after controlling for simple local cues.

The punctured family uses a certified metric-matched sweep of 50 pairs:
`p=4` and `h=6,8,...,104`. Generation rejects any pair whose rank counts,
degree profile, boundary loops, vertical-cycle periods, or exact node-to-face
distance histogram differ between classes. Consequently the implemented
count, boundary, and distance probes receive identical inputs on every pair;
the benchmark measures whether stronger models detect structure beyond those
controls. Certificates record the matched signatures for each evaluated pair.

The shared orientability config is:

```text
experiments/orientability/config.yaml
```

This single config contains construction/evaluation parameters for all three
orientability families.  Family-specific data settings live under:

```yaml
families:
  cylinder_vs_mobius:
  torus_vs_klein:
  punc_torus_vs_punc_klein:
```

The launcher is:

```bash
cd experiments/orientability
bash orientability.sh
```

Default behavior is intentionally small:

```text
family=cylinder_vs_mobius
model_name=gin
seeds=0
num_gpus=1
```

Examples:

```bash
# One model, one family, four GPUs and automatic seeds 0,1,2,3
bash orientability.sh family=torus_vs_klein model_name=gin num_gpus=4

# SMCN on all orientability families
bash orientability.sh family=all model_name=smcn seeds=0,1,2,3 num_gpus=4

# Run the available baseline set
bash orientability.sh family=all model_name=baselines seeds=0,1,2,3 num_gpus=4

# Override training hyperparameters
bash orientability.sh family=punc_torus_vs_punc_klein model_name=esan epochs=40 copies=32 batch_size=8
```

Supported model names:

```text
mlp
count_mlp
dist_mlp
boundary_mlp
gin
esan
homp
hasse
smcn
baselines
all
```

`model_name=mlp` expands to the three MLP-style controls:

```text
count_mlp
dist_mlp
boundary_mlp
```

Each probe has one canonical implementation:

```text
baselines/count_mlp/
baselines/dist_mlp/
baselines/boundary_mlp/
```

In configuration and result files, the count probe retains the legacy model
label `mlp`. At the launcher level, `model_name=mlp` runs all three probes;
use `model_name=count_mlp` to run only the count probe.

`model_name=smcn` uses the SMCN model implementation from:

```text
baselines/smcn/
```

### Fundamental Group

The fundamental-group benchmark tests whether models can separate cellular
complexes built from surface-group presentations from matched free-group
controls.  The active experiment lives in:

```text
experiments/fundamental_group/
```

The genus-2 benchmark entry point is:

```bash
cd experiments/fundamental_group/FG_genus_2
bash fundamental_group.sh
```

The shared config is:

```text
experiments/fundamental_group/config.yaml
```

Examples:

```bash
# Default: fundamental_group, GIN, GPU 0, seed 0
bash fundamental_group.sh

# SMCN
bash fundamental_group.sh model_name=smcn

# One baseline
bash fundamental_group.sh model_name=gin

# Main baseline set
bash fundamental_group.sh model_name=baselines

# Everything
bash fundamental_group.sh model_name=all

# Four GPUs and automatic seeds 0,1,2,3
bash fundamental_group.sh model_name=esan num_gpus=4
```

Friendly launcher aliases:

```text
epochs=N   -> num_epochs=N
copies=N   -> number_of_permuted_copies=N
lr=VALUE   -> learning_rate=VALUE
seeds=...  -> explicit model seeds
num_gpus=N -> number of concurrent CUDA jobs
```

Run either launcher with `-h` or `--help` for its compact usage summary.

### Torus vs Sphere Control

The torus-vs-sphere experiment is a positive control for the baseline
pipeline. Run it from the repository root:

```bash
python experiments/torus_vs_sphere/test_torus_vs_sphere.py
python experiments/torus_vs_sphere/run_baselines.py
```

For a smaller smoke run:

```bash
python experiments/torus_vs_sphere/run_baselines.py \
  --copies 8 --epochs 10 --mlp-epochs 40
```

Its dataset cache, summary logs, and figures are written below
`experiments/torus_vs_sphere/` and are ignored by Git.

## Data Generation

Base-pair generation is handled by:

```text
data/get_base_pairs.py
```

Generated base pairs are stored under:

```text
data/<family>/
```

For orientability families, `get_base_pairs.py` reads the common config:

```text
experiments/orientability/config.yaml
```

For the free-vs-surface fundamental-group data, it reads:

```text
experiments/fundamental_group/config.yaml
```

Generate all orientability base pairs:

```bash
python data/get_base_pairs.py --family orientability
```

Generate every supported generated-data family:

```bash
python data/get_base_pairs.py --family all
```

Generate one family:

```bash
python data/get_base_pairs.py --family cylinder_vs_mobius
python data/get_base_pairs.py --family torus_vs_klein
python data/get_base_pairs.py --family punc_torus_vs_punc_klein
python data/get_base_pairs.py --family free_vs_surface
```

The generator writes only base pairs.  It does not write all permutation copies
to disk.  Evaluators construct permutation copies on demand during evaluation.

Each generated-data directory can contain a `manifest.json` recording the
construction parameters.  If an existing directory matches the current
manifest, generation is skipped.  Use `--overwrite` only when you intentionally
want to regenerate a family.

## Outputs

Orientability runs write:

```text
experiments/orientability/<family>/runs/<model>/seed<seed>_<tag>/
experiments/orientability/<family>/logs/<model>/seed<seed>_<tag>.log
```

Fundamental-group runs write:

```text
experiments/fundamental_group/FG_genus_2/runs/<model>/seed<seed>_<tag>/
experiments/fundamental_group/FG_genus_2/logs/<model>/seed<seed>_<tag>.log
```

Typical run artifacts:

```text
config.json
metrics.json
pair_results.csv
certificates.jsonl
```

`metrics.json` contains BREC counts, target/reliability T2 statistics, runtime,
and model parameter count when available.

`certificates.jsonl` records benchmark-construction metadata for auditability.

## Aggregation

The shared aggregation helper is:

```text
experiments/aggregate_runs.py
```

Launchers call it automatically after successful seed runs.  It can also be
run manually:

```bash
python experiments/aggregate_runs.py experiments/orientability/cylinder_vs_mobius/runs gin
python experiments/aggregate_runs.py experiments/fundamental_group/FG_genus_2/runs smcn
```

The aggregator reports per-seed BREC counts and final mean ± standard deviation
over the selected seed runs.

## GPU Scheduling

Both main launchers accept `num_gpus=N`, `n_gpus=N`, `gpus=N`, and
`num_cuda_gpus=N`. By default they use one GPU and run seed `0`. When seeds
are omitted, `N` GPUs automatically produce seeds `0..N-1`:

```bash
bash orientability.sh family=all model_name=gin n_gpus=4
```

Use `GPU_CARDS` to select physical cards. If no GPU-count argument is given,
the number of listed cards determines the GPU count and automatic seed count:

```bash
GPU_CARDS="0,2" bash orientability.sh family=torus_vs_klein model_name=smcn
```

An explicit `seeds=...` overrides automatic seeds. Seeds are assigned to the
selected GPUs round-robin and run in GPU-sized waves.

## Memory Policy

The evaluators are written to avoid materializing the full
`pairs × copies × seeds` dataset in memory.

Important behavior:

- base pairs are loaded pair-wise;
- permutation copies are constructed for the current pair/trial;
- target and reliability phases are evaluated separately;
- tensors are moved to GPU batch-wise;
- CPU/GPU memory is released between pair phases where possible;
- gradient accumulation keeps one optimizer update per epoch even when copies
  are processed in mini-batches.

This matters for fair BREC comparison: reducing the mini-batch size should not
increase the number of optimizer steps per epoch.
