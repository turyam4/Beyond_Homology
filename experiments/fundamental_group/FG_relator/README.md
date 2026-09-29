# Held-out relator families — expanding the FG experiment (R3 comment 1)

Answers *"additional presentation families ... or a held-out family that was not
used in designing the generator."*

## The enabling fact

Every relator here lies in the commutator subgroup **[F,F]**, so it abelianises
to 0 and therefore gives **H = (ℤ, ℤ^{2g}, ℤ), torsion-free — automatically.**
Homology matching across families is free; the existing gates pass by
construction and are still checked numerically per pair.

`relator_library.py` (all verified in [F,F], all counts even):

| family | π₁ | length @ g=2 | counts |
|---|---|---|---|
| `trivial` | F₄ | 0 | (0,0,0,0) |
| `surface` | genus-2 surface group | 8 | (2,2,2,2) |
| `deep_commutator` `[[a,b],[c,d]]` | second-derived | 16 | (4,4,4,4) |
| `commutator_square` `[a,b]²` | proper power | 8 | (4,4,0,0) |
| `surface_cubed_first` | unequal handle powers | 16 | (6,6,2,2) |

## Non-isomorphism is CERTIFIED, not assumed

Homology cannot separate these (all identical). `hom_certificate.py` counts
|Hom(G,H)| for H ∈ {S3, D4, Q8, A4, S4} — an isomorphism invariant, so
**differing counts prove non-isomorphism**. All 10 pairs certified distinct.
Equal counts are reported as "not separated by this panel", never as
"isomorphic".

The counting code reproduces two known closed forms exactly — |Hom(F₄,H)| =
|H|⁴, and the Frobenius/Mednykh value |H|³·Σ_χ(1/χ(1))² for genus-2 surface
groups (S3→486, A4→5376, S4→34176). It is not an unchecked hand-rolled
invariant.

## Length / count budgeting (the one subtle piece)

Different relators have different lengths and counts, which would break the
word-length and letter-count gates. Members are budgeted to a common target

    t_k = max(c_A[k], c_B[k]);   target_k = t_k + 2·N_k   (N = SHARED noise)

This is **always solvable**: a relator in [F,F] has exponent sum 0, so
#pos = #neg, so every generator occurs an *even* number of times, so every
deficit is even and fillable with x·x⁻¹ pairs.

Member construction mirrors the original benchmark: nontrivial relator = base
+ Tietze insertions; trivial relator = random non-crossing (Dyck) pairing (not
adjacent insertions, which backtrack-counting would detect).

Every gate is **imported** from `FG_genus_2/fundamental_group_dataset_genus2.py`,
never reimplemented.

## Verified on this machine (primary pairing `deep_vs_free`)

- **All 16 gates pass 8/8**, including `A_reduces_to_relator_A` /
  `B_reduces_to_relator_B` recomputed independently.
- Groups **certified distinct via S4** (172,800 vs 331,776).
- Acceptance ≈ 15 attempts/pair (min 3, max 50) — cheaper than surface-vs-free (≈57).
- Homology (1,4,1) torsion-free, χ = −2, Hasse-WL stabilises at round 6.

## Files

`relator_library.py` · `hom_certificate.py` · `relator_pair_dataset.py` (Phase 2)
· `validate_relator_pairs.py` (Phase 3) · `run_relator.py` (Phase 4 shim)
· `run_relator_experiment.sh` (Phase 5 driver)

## Run

```bash
python hom_certificate.py                                   # non-isomorphism table
python validate_relator_pairs.py --pairing deep_vs_free --pairs 16
bash run_relator_experiment.sh                              # caches + all models
PAIRING=surface_vs_deep bash run_relator_experiment.sh      # neither class free
```

Cache safety: `cache_version` is suffixed with the pairing and `pairing` is
injected into the manifest, so a relator-family cache can never be
cross-loaded as the surface-vs-free cache.

## Honest limitation

**This addresses generality, not MC4.** Measured on the held-out family:
spectral channel matched **0/8** (fully open) and fine metric **3/8** — i.e.
the same leak profile as surface-vs-free. SPEC_MLP scores **25/32 (78%)** here,
matching its 87.5% on the original family. Expanding the families does not
close the spectral channel, and should not be presented as if it does.
