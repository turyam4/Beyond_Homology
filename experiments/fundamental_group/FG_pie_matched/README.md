# E-G · π₁-matched control families (fixes R3)

Falsification test for *"separation may come from another global structural
feature created during realization."*

The cross-class family pairs a **surface-group** complex against a **free-group**
complex. These controls pair **two members of the same π₁ class** with
independent noise realizations, pushed through the **identical gate stack**:

| family | member A | member B | π₁ |
|---|---|---|---|
| `surface_vs_surface` | reduces to `[a₁,b₁][a₂,b₂]` | reduces to `[a₁,b₁][a₂,b₂]` | identical (surface group) |
| `free_vs_free` | reduces to identity | reduces to identity | identical (F₄) |

If a realization artifact drives separation, it appears here too. If separation
is class-level, these sit at the reliability floor.

## Why the gates are provably the same

`control_family_dataset.py` **imports** `local_incidence_profile`,
`vertex_link_wl_profile`, `stable_rank_coloured_hasse_wl`, `_rank_counts` and
`_edge_coface_counts` from `tight_fundamental_group_dataset.py` and calls them
on the control pairs. No gate is reimplemented, so none can drift.

Gates enforced (all of them): algebraic π₁ certification of **both** members ·
cell counts · Euler characteristic · exact integral homology, torsion-free
(1,4,1) · word length · slot multiset · per-generator counts · signed exponent
sums · node-token multiset · backtrack motifs · local incidence profile ·
vertex-link WL profile · rank-coloured Hasse 1-WL to stabilisation.

**Plus one gate the cross-class family does not need:** *distinctness*. Two
same-class members can coincide as words up to the construction's own
rotation/reversal symmetry; such a pair would silently degenerate into the
existing same-word reliability control and sit at the floor for a trivial
reason. Those are rejected (`require_distinct=True`) and the certificate
records `words_distinct_up_to_symmetry`. **State this in the paper** — without
it a skeptic can argue the controls were rigged to pass.

## Verified on this machine

- `validate_control.py --pairs 8`: **16/16 gates pass 8/8** for both families.
- Acceptance: `surface_vs_surface` ≈ 9–14 attempts/pair, `free_vs_free` ≈ 49–62
  (cross-class ≈ 57), so cost is comparable.
- Certified pairs land in the same three size buckets (48/72/120) and the same
  Hasse-WL stabilisation round (6) as the cross-class family.
- End-to-end smoke run of the full harness on a control cache: works.

## Run

```bash
export PYTHON=/path/to/smcn-env/bin/python3
export FG_GENUS2_DIR=/path/to/FG_genus_2      # only if not the default location

"$PYTHON" validate_control.py --pairs 16       # audit first, always
bash run_control_family.sh                     # both families, 4 seeds
"$PYTHON" aggregate_control.py                 # fills the R3 table (+ LaTeX)
```

Single runs (note the **quotes** — zsh globs bare brackets):

```bash
"$PYTHON" run_control.py control_family=surface_vs_surface cache_only=true
"$PYTHON" run_control.py control_family=surface_vs_surface 'model_names=[smcn]' binary_marking=true model_seed=0
```

Run the native spectral MLP baseline through the environment-selecting wrapper:

```bash
./run_spec_mlp.sh control_family=surface_vs_surface
./run_spec_mlp.sh control_family=surface_vs_surface num_gpus=4
./run_spec_mlp.sh control_family=surface_vs_surface \
  num_gpus=4 seeds=[4,5,6,7]
```

The wrapper defaults to CUDA with one GPU and one model seed (`0`). With
`num_gpus=N` and no explicit `seeds`, it launches seeds `0..N-1`, one per GPU.
More explicit seeds are processed in waves of at most `N`. Results are written
under `runs_control_<family>_spec/spec_mlp/seedN`, matching the existing result
tree, and console logs under `logs_control_<family>_spec/seedN.log`. Use
`GPU_CARDS=...` to select physical GPU IDs.

It can also be selected with the shared FG launcher as
`model_names=[spec_mlp]`, or included beside the other MLP baselines.

`LADDER=1 bash run_control_family.sh` adds gin / homp_cin / hasse_gin / the MLPs.

## Cache safety

`cache_version` is suffixed with the family (flows into both the cache
directory name and the manifest) and `control_family` is injected into the
manifest. `validate_cache` compares the whole manifest, so a control cache can
never be silently cross-loaded as a cross-class cache.

## Also measured here: the C5 metric channel

`validate_control.py` reports the global-metric channel at two granularities,
because probe-level and model-level granularity differ:

- **coarse** — node→node SPD histogram, eccentricity, diameter, aggregated
  node→cell marking histogram. *This is what `DIST_MLP` sees.*
- **fine** — the per-vertex profile `m(v,e) = min_{u≺e} d(v,u)`. *This is what
  SMCN's shortest-path marking actually consumes.*

On the cross-class family the coarse channel matched 16/16 while the fine
channel matched only 6/16. Report the control families' fine-openness rates
alongside it: if the controls carry comparable metric openness yet sit at the
floor, metric openness and separation vary independently — which is the
strongest available defence of the SPD configuration.

## Noise stress sweep (the second, weaker test)

Magnitude is a config key; placement is already randomised per pair by the
generator. Sweep magnitude on either family:

```bash
for n in 1 2 4 8; do
  "$PYTHON" run_control.py control_family=surface_vs_surface 'model_names=[smcn]' \
      noise_pairs_per_g=$n binary_marking=true model_seed=0 \
      output_dir=runs_noise/n$n
done
```

Separation tracking noise magnitude indicates an artifact; separation flat in
noise indicates a class-level signal.

## Report either way

A high separation rate on the π₁-matched controls is a **real negative finding
about the benchmark**. It is far better to publish it than to have it found
later. The aggregator prints both readings so the outcome is not left to
interpretation after the fact.
