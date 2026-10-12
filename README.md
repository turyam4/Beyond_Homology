# Beyond Homology

Code for controlled experiments on cellular complexes with graph and
cell-complex neural networks. The primary experiments measure whether a model
separates matched pairs under a BREC-style target/reliability protocol.

These are representation-level experiments. A successful run shows separation
of the emitted tensors under the tested model and budget; it does **not** by
itself prove that the model computes orientability, the fundamental group, or
any other topological invariant.

## Repository layout

```text
baselines/                         Model implementations
data/get_base_pairs.py             Shared base-pair generator
experiments/orientability/         Three orientability-labelled families
experiments/fundamental_group/     Genus-2 and auxiliary presentation families
experiments/torus_vs_sphere/       Positive-control benchmark
experiments/aggregate_runs.py      Multi-seed result aggregation
```

Generated datasets, caches, logs, and run directories are intentionally
ignored by Git.

## Environment

- Python 3.10 or newer; the checked environment used Python 3.13.7.
- PyTorch and PyTorch Geometric compatible with the target CPU/CUDA platform.
- The main launchers require Bash 4.3 or newer. macOS `/bin/bash` 3.2 is too old.
- The fundamental-group launcher requires CUDA. The orientability launcher can
  use CPU with `device=cpu num_gpus=1`.

On macOS, install a current Bash with `brew install bash` and invoke a launcher
as `"$(brew --prefix)/bin/bash" orientability.sh`. The examples below assume
that `bash --version` reports version 4.3 or newer.

`requirements.txt` records the package versions used, but it is not a complete
environment bootstrap file: its first line is a Python version requirement,
and the repository also needs `hydra-core`, `matplotlib`, and `pytest`.
Install a platform-appropriate PyTorch/PyG stack first, then the remaining
packages. Activate that environment before running the commands below, or set
`PYTHON=/path/to/python`.

## Validate the checkout

Run from the repository root:

```bash
python -m pytest -q
python experiments/orientability/test_orient.py
python experiments/torus_vs_sphere/test_torus_vs_sphere.py
```

In the validated environment, pytest reports 13 passing tests and both direct
topology checks finish with all checks passed.

## Generate base pairs

The repository does not track generated tensor datasets. Preview the resolved
families and destinations with:

```bash
python data/get_base_pairs.py --family all --dry-run
```

Generate all orientability families:

```bash
python data/get_base_pairs.py --family orientability
```

Individual orientability families are `cylinder_vs_mobius`,
`torus_vs_klein`, and `punc_torus_vs_punc_klein`. The current configurations
produce 64, 64, and 50 base pairs, respectively. Existing data is reused;
`--overwrite` regenerates it deliberately.

The generator also supports `--family free_vs_surface`, but those standalone
files are not inputs to the genus-2 launcher below. That launcher builds its
own target/reliability cache automatically.

## Orientability-labelled experiments

Configuration: `experiments/orientability/config.yaml`

The three families are:

- `cylinder_vs_mobius`
- `torus_vs_klein`
- `punc_torus_vs_punc_klein` (the configured metric-matched punctured family)

The launcher validates or generates the requested base pairs, runs the selected
seeds, and aggregates successful runs:

```bash
cd experiments/orientability

# Default: cylinder_vs_mobius, GIN, seed 0, CUDA device 0
bash orientability.sh

# One CPU smoke run
bash orientability.sh family=cylinder_vs_mobius model_name=gin \
  device=cpu num_gpus=1 seeds=0 epochs=1 copies=8 max_pairs=1

# SMCN on every family with four allocated GPUs
bash orientability.sh family=all model_name=smcn \
  num_gpus=4 seeds=0,1,2,3
```

Supported names are `smcn`, `count_mlp`, `dist_mlp`, `boundary_mlp`,
`gin`, `esan`, `homp`, and `hasse`, plus `baselines` and `all`. The alias
`model_name=mlp` expands to the count, distance, and boundary MLP controls;
use `model_name=count_mlp` for the count control alone.

## Genus-2 free-vs-surface experiment

Configuration: `experiments/fundamental_group/config.yaml`

The current configuration evaluates 16 matched genus-2 pairs with 32 permuted
copies per example. Target trials compare a genus-2 surface-group presentation
with a free-group presentation whose relator freely reduces to the identity.
Reliability trials compare independently permuted copies of the same surface
example. A fresh model is trained for each pair.

Interpret results relative to the stored complex representation and reported
shortcut controls. Separation is not evidence that a model generally computes
the fundamental group.

Run inside a CUDA allocation:

```bash
cd experiments/fundamental_group/FG_genus_2

# Default: GIN, seed 0, CUDA device 0
bash fundamental_group.sh

# SMCN or all configured baselines
bash fundamental_group.sh model_name=smcn
bash fundamental_group.sh model_name=baselines num_gpus=4 seeds=0,1,2,3
```

Supported names are `smcn`, `count_mlp`, `dist_mlp`, `boundary_mlp`,
`spec_mlp`, `gin`, `esan`, `homp`, and `hasse`, plus `baselines` and `all`.
The launcher builds or validates the required model-specific target/reliability
cache before evaluation and aggregates completed seeds. Do not pre-generate
`data/free_vs_surface` for this workflow.

Both main launchers accept aliases such as `epochs=N`, `copies=N`, `lr=X`,
`seeds=...`, and `num_gpus=N`. `GPU_CARDS="0,2"` selects visible physical
cards. These options schedule processes; they do not request cluster resources.
Use `--help` for the full launcher syntax.

## Positive control: torus vs sphere

This experiment checks the non-SMCN baseline pipeline on a deliberately easier
pair:

```bash
python experiments/torus_vs_sphere/run_baselines.py

# Reduced run
python experiments/torus_vs_sphere/run_baselines.py \
  --copies 8 --epochs 10 --mlp-epochs 40
```

## Auxiliary experiments

- `experiments/fundamental_group/FG_pie_matched/` contains same-class control
  families and its own README and launchers.
- `experiments/fundamental_group/FG_relator/` contains held-out relator-family
  experiments. Its independent validator imports `hom_certificate.py`, which
  is absent from this snapshot; do not claim that validation is reproducible
  until that file is restored.
- `experiments/orientability/depth_sweep_orientability/` contains the SMCN
  depth sweep and a dedicated README.

## Outputs and interpretation

The main launchers write under the corresponding experiment directory:

```text
<family>/runs/<model>/seed<seed>_<tag>/
<family>/logs/<model>/seed<seed>_<tag>.log
```

Typical run artifacts:

- `config.json`: resolved run configuration;
- `metrics.json`: aggregate counts, statistics, runtime, and parameter count;
- `pair_results.csv`: per-pair measurements;
- `certificates.jsonl`: construction and matching metadata.

Report the three BREC quantities separately:

- **target found**: the target pair separates;
- **reliability passed**: the reliability pair does not falsely separate;
- **BREC solved**: both conditions hold for the same pair.

The launchers call `experiments/aggregate_runs.py` automatically. Generated
artifacts may contain machine paths, so re-run an anonymity scan before sharing
raw result directories. Also treat `.pt` files as trusted inputs: parts of the
codebase use `torch.load(..., weights_only=False)`, which is unsafe for files
from untrusted sources.
