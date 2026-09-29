"""
relator_pair_dataset.py -- PHASE 2: certified pairs from ARBITRARY relator
families with identical homology.
==========================================================================
Generalises the surface-vs-free generator to any two relators r_A, r_B drawn
from the commutator subgroup [F,F] (see relator_library.py). Because every
such relator abelianises to 0, all families share homology (1, 2g, 1),
torsion-free -- so the existing homology gates pass by construction and are
still checked numerically per pair.

PRIMARY PAIRING (the reviewer's requested HELD-OUT family):
    deep_commutator  [[a1,b1],[a2,b2]]   vs   trivial (free F_4)
The generator was designed and tuned on surface-vs-free; this family was not
used in that design.

THE ONE NON-OBVIOUS PIECE: LENGTH / COUNT BUDGETING
Different relators have different lengths and per-generator counts, e.g. at
g=2: surface (2,2,2,2) len 8; deep_commutator (4,4,4,4) len 16; trivial
(0,0,0,0) len 0. Pairing them naively breaks the word-length and
letter-count gates. We therefore budget to a common per-generator target

    t_k     = max(c_A[k], c_B[k])            (structural floor)
    N       = a SHARED random noise multiset (diversity; identical for both)
    target_k = t_k + 2 * N_k

and fill each member up to target_k. This is always solvable because a
relator in [F,F] has exponent sum 0 for every generator, hence
#positive = #negative, hence EVERY generator occurs an EVEN number of times,
so every deficit t_k - c[k] is even and fillable with x x^-1 pairs.

MEMBER CONSTRUCTION (mirrors the original benchmark exactly)
  * nontrivial relator: base relator + Tietze insertions of x x^-1 at random
    positions (free reduction returns the base -- a theorem, and verified).
  * trivial relator: a random non-crossing (Dyck) pairing over the letter
    pool -- NOT adjacent x x^-1 insertions, which would be trivially
    detectable by backtrack counting.

Every gate is IMPORTED from the genus-2 tight generator, never reimplemented,
so "passes the same gate stack" is structural rather than asserted.
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

Slot = Tuple[int, bool]
Edge = Tuple[int, int]
Face = List[int]


# ---------------------------------------------------------------------------
# locate the Beyond_Homology fundamental_group tree
# ---------------------------------------------------------------------------
def _fg_dirs() -> Tuple[Path, Path]:
    """(fg_root with fg_core.py, genus2_dir with the tight gates)."""
    env = os.environ.get("FG_ROOT")
    module_dir = Path(__file__).resolve().parent
    cands = ([Path(env)] if env else []) + [
        module_dir.parent,
        module_dir.parent.parent / "fundamental_group",
    ]
    for c in cands:
        if (c / "fg_core.py").is_file() and \
           (c / "FG_genus_2" / "fundamental_group_dataset_genus2.py").is_file():
            return c, c / "FG_genus_2"
    raise RuntimeError(
        "Could not locate Beyond_Homology/experiments/fundamental_group "
        "(needs fg_core.py and FG_genus_2/fundamental_group_dataset_genus2.py). "
        "Set FG_ROOT."
    )


FG_ROOT, FG_G2 = _fg_dirs()
for _p in (str(Path(__file__).resolve().parent), str(FG_ROOT), str(FG_G2)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fg_core import (  # noqa: E402
    create_complex_edges_faces,
    free_reduce,
    integral_homology,
    is_rotation,
    random_trivial_slots,
    signed_word,
    slot_letter_counts,
    slot_signed_exponent_sums,
    word_backtracks,
)
from fg_core import betti_numbers as _betti  # noqa: E402

# GATES -- imported, never reimplemented
from fundamental_group_dataset_genus2 import (  # noqa: E402
    _edge_coface_counts,
    _rank_counts,
    local_incidence_profile,
    stable_rank_coloured_hasse_wl,
    vertex_link_wl_profile,
)

from relator_library import (  # noqa: E402
    RELATORS,
    is_in_commutator_subgroup,
    letter_counts as rel_letter_counts,
)

# named pairings
PAIRINGS: Dict[str, Tuple[str, str]] = {
    "deep_vs_free": ("deep_commutator", "trivial"),          # PRIMARY, held-out
    "surface_vs_deep": ("surface", "deep_commutator"),       # neither class free
    "surface_vs_free": ("surface", "trivial"),               # original, for parity
    "sq_vs_free": ("commutator_square", "trivial"),
    "cubed_vs_free": ("surface_cubed_first", "trivial"),
}


# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RelatorPairCertificate:
    # field names mirror TightPairCertificate for downstream compatibility
    genus: int
    subdivision: int
    configured_noise_pairs_per_g: int
    attempts: int
    rank_counts: Tuple[int, int, int]
    euler_characteristic: int
    betti_numbers: Tuple[int, int, int]
    integral_betti: Tuple[int, int, int]
    torsion_H1: Tuple[int, ...]
    integral_homology_torsion_free: bool
    word_path_length: int
    abstract_word_length: int
    boundary_edges: int
    hasse_wl_rounds_to_stability: int
    hasse_wl_equivalent: bool
    vertex_link_wl_profiles_equal: bool
    local_incidence_profiles_equal: bool
    # relator-family specific
    pairing: str
    relator_a: str
    relator_b: str
    relator_a_reduced_len: int
    relator_b_reduced_len: int
    both_in_commutator_subgroup: bool
    words_distinct_up_to_symmetry: bool
    backtrack_count: int


@dataclass
class RelatorPair:
    surface_word_path: List[int]      # member A  (names kept for drop-in use)
    free_word_path: List[int]         # member B
    surface_slots: List[Slot]
    free_slots: List[Slot]
    surface_edges: List[Edge]
    free_edges: List[Edge]
    surface_faces: List[Face]
    free_faces: List[Face]
    certificate: RelatorPairCertificate


# ---------------------------------------------------------------------------
def verify_reduces_to(slots: List[Slot], target: List[Slot]) -> bool:
    """True iff the slot word free-reduces EXACTLY to `target`.
    Generalises verify_class_A_reduces_to_commutator / ..._to_identity."""
    return free_reduce(signed_word(slots)) == signed_word(target)


def _words_distinct(a: List[int], b: List[int]) -> bool:
    if len(a) != len(b):
        return True
    if is_rotation(a, b):
        return False
    return not is_rotation(a, list(reversed(b)))


# ---------------------------------------------------------------------------
def _generate_relator_pair_words(
    g: int, rel_a: str, rel_b: str, extra_noise_pairs: int,
    rng: np.random.Generator, subdivision: int,
    match_backtracks: bool = True, max_tries: int = 100,
    max_expansions: int = 10,
):
    """Two members realising relators rel_a / rel_b, budgeted to identical
    per-generator counts and identical word length."""
    p = int(subdivision)
    r_a = RELATORS[rel_a](g)
    r_b = RELATORS[rel_b](g)
    if not (is_in_commutator_subgroup(r_a, g) and is_in_commutator_subgroup(r_b, g)):
        raise ValueError("both relators must lie in [F,F] for homology matching")

    gen_map = list(range(1, 2 * g + 1))
    rng.shuffle(gen_map)
    flip_map = rng.choice([True, False], size=2 * g + 1)

    def node_id(mg: int, t: int) -> int:
        return 1 + (mg - 1) * (p - 1) + (t - 1)

    def get_path(gi: int, is_inv: bool) -> List[int]:
        mg = gen_map[gi - 1]
        minv = is_inv ^ flip_map[mg]
        inner = [node_id(mg, t) for t in range(1, p)]
        if minv:
            inner = inner[::-1]
        return [0] + inner

    def slots_to_path(slots: List[Slot]) -> List[int]:
        out: List[int] = []
        for gi, iv in slots:
            out += get_path(gi, iv)
        return out

    c_a = rel_letter_counts(r_a, g)
    c_b = rel_letter_counts(r_b, g)
    floor = {k: max(c_a[k], c_b[k]) for k in range(1, 2 * g + 1)}

    def build_member(relator: List[Slot], counts: Dict[int, int],
                     target: Dict[int, int]) -> List[Slot]:
        if len(relator) == 0:
            # trivial relator -> random non-crossing (Dyck) pairing
            letters: List[int] = []
            for k in range(1, 2 * g + 1):
                letters += [k] * (target[k] // 2)
            rng.shuffle(letters)
            return random_trivial_slots(letters, rng)
        slots = list(relator)
        to_insert: List[int] = []
        for k in range(1, 2 * g + 1):
            deficit = target[k] - counts[k]
            assert deficit >= 0 and deficit % 2 == 0, "budget must be even/non-negative"
            to_insert += [k] * (deficit // 2)
        rng.shuffle(to_insert)
        for letter in to_insert:
            at = int(rng.integers(0, len(slots) + 1))
            slots = slots[:at] + [(letter, False), (letter, True)] + slots[at:]
        return slots

    extra = int(extra_noise_pairs)
    a_slots = b_slots = None
    best_gap = None

    for expansion in range(max_expansions):
        noise = [int(rng.integers(1, 2 * g + 1)) for _ in range(extra)]
        n_k = Counter(noise)
        target = {k: floor[k] + 2 * n_k.get(k, 0) for k in range(1, 2 * g + 1)}

        a_slots = build_member(r_a, c_a, target)

        def draw_b():
            return build_member(r_b, c_b, target)

        tgt_bt = word_backtracks(slots_to_path(a_slots))
        best, best_gap = None, None
        tries = max_tries if match_backtracks else 1
        for _ in range(tries):
            cand = draw_b()
            gap = (abs(word_backtracks(slots_to_path(cand)) - tgt_bt)
                   if match_backtracks else 0)
            if best_gap is None or gap < best_gap:
                best, best_gap = cand, gap
            if gap == 0:
                break
        b_slots = best
        if (not match_backtracks) or best_gap == 0 or expansion == max_expansions - 1:
            break
        extra += 1  # widen the SHARED budget -- both members stay matched

    WA, WB = slots_to_path(a_slots), slots_to_path(b_slots)

    def randomize(W: List[int]) -> List[int]:
        shift = int(rng.integers(0, len(W) // p)) * p
        out = W[shift:] + W[:shift]
        if rng.random() > 0.5:
            out = out[::-1]
        return out

    return randomize(WA), randomize(WB), p, (a_slots, b_slots), extra


# ---------------------------------------------------------------------------
def _certify(g: int, pairing: str, rel_a: str, rel_b: str, configured_noise: int,
             attempts: int, subdivision: int, word_a, word_b, slots_a, slots_b):
    r_a, r_b = RELATORS[rel_a](g), RELATORS[rel_b](g)
    ea, fa = create_complex_edges_faces(word_a)
    eb, fb = create_complex_edges_faces(word_b)

    # --- algebraic pi_1 certification of BOTH members ---
    if not verify_reduces_to(slots_a, r_a):
        raise RuntimeError(f"member A does not reduce to relator {rel_a!r}")
    if not verify_reduces_to(slots_b, r_b):
        raise RuntimeError(f"member B does not reduce to relator {rel_b!r}")

    ca, cb = _rank_counts(ea, fa), _rank_counts(eb, fb)
    if ca != cb:
        return None
    chi_a = ca[0] - ca[1] + ca[2]
    chi_b = cb[0] - cb[1] + cb[2]
    if chi_a != chi_b:
        return None

    expected = (1, 2 * g, 1)
    ba = tuple(int(v) for v in _betti(ea, fa))
    bb = tuple(int(v) for v in _betti(eb, fb))
    if ba != expected or bb != expected:
        return None

    if len(word_a) != len(word_b) or len(slots_a) != len(slots_b):
        return None
    if slot_letter_counts(slots_a) != slot_letter_counts(slots_b):
        return None
    if Counter(slots_a) != Counter(slots_b):
        return None
    if slot_signed_exponent_sums(slots_a, g) != slot_signed_exponent_sums(slots_b, g):
        return None
    if Counter(word_a) != Counter(word_b):
        return None
    bt_a, bt_b = word_backtracks(word_a), word_backtracks(word_b)
    if bt_a != bt_b:
        return None

    if local_incidence_profile(ea, fa) != local_incidence_profile(eb, fb):
        return None
    if vertex_link_wl_profile(ea, fa) != vertex_link_wl_profile(eb, fb):
        return None

    wl_eq, wl_rounds = stable_rank_coloured_hasse_wl((ea, fa), (eb, fb))
    if not wl_eq:
        return None

    ha, hb = integral_homology(ea, fa), integral_homology(eb, fb)
    if ha["betti"] != expected or hb["betti"] != expected:
        return None
    if not (ha["torsion_free"] and hb["torsion_free"]):
        return None
    if ha["torsion_H1"] != hb["torsion_H1"]:
        return None

    boundary = sum(v == 1 for v in _edge_coface_counts(ea, fa).values())
    cert = RelatorPairCertificate(
        genus=g, subdivision=subdivision,
        configured_noise_pairs_per_g=configured_noise, attempts=attempts,
        rank_counts=ca, euler_characteristic=chi_a, betti_numbers=ba,
        integral_betti=tuple(int(v) for v in ha["betti"]),
        torsion_H1=tuple(ha["torsion_H1"]),
        integral_homology_torsion_free=True,
        word_path_length=len(word_a), abstract_word_length=len(slots_a),
        boundary_edges=boundary,
        hasse_wl_rounds_to_stability=wl_rounds, hasse_wl_equivalent=True,
        vertex_link_wl_profiles_equal=True, local_incidence_profiles_equal=True,
        pairing=pairing, relator_a=rel_a, relator_b=rel_b,
        relator_a_reduced_len=len(free_reduce(signed_word(r_a))),
        relator_b_reduced_len=len(free_reduce(signed_word(r_b))),
        both_in_commutator_subgroup=True,
        words_distinct_up_to_symmetry=_words_distinct(word_a, word_b),
        backtrack_count=int(bt_a),
    )
    return cert, ea, fa, eb, fb


# ---------------------------------------------------------------------------
def generate_relator_pair(*, seed: int, pairing: str = "deep_vs_free",
                          genus: int = 2,
                          noise_candidates: Sequence[int] = (2, 4, 8),
                          subdivision: int = 3, max_attempts: int = 5_000,
                          max_tries: int = 100, max_expansions: int = 10
                          ) -> RelatorPair:
    if pairing not in PAIRINGS:
        raise ValueError(f"pairing must be one of {list(PAIRINGS)}")
    if genus != 2:
        raise ValueError("genus-2 only until a separate leakage audit exists")
    rel_a, rel_b = PAIRINGS[pairing]

    rng = np.random.default_rng(seed)
    for attempt in range(1, max_attempts + 1):
        noise = int(noise_candidates[(attempt - 1) % len(noise_candidates)])
        wa, wb, p, (sa, sb), used = _generate_relator_pair_words(
            genus, rel_a, rel_b, noise, rng, subdivision,
            match_backtracks=True, max_tries=max_tries,
            max_expansions=max_expansions)
        got = _certify(genus, pairing, rel_a, rel_b, noise, attempt, p, wa, wb, sa, sb)
        if got is None:
            continue
        cert, ea, fa, eb, fb = got
        return RelatorPair(surface_word_path=wa, free_word_path=wb,
                           surface_slots=sa, free_slots=sb,
                           surface_edges=ea, free_edges=eb,
                           surface_faces=fa, free_faces=fb, certificate=cert)
    raise RuntimeError(
        f"No certified {pairing} pair in {max_attempts} attempts. Increase "
        "max_attempts or change seed; do NOT relax the gates.")


def generate_relator_pair_descriptor(*, seed: int, pairing: str = "deep_vs_free",
                                     noise_candidates: Sequence[int] = (2, 4, 8),
                                     subdivision: int = 3,
                                     max_attempts: int = 5_000):
    """Drop-in for generate_tight_pair_descriptor."""
    pair = generate_relator_pair(seed=seed, pairing=pairing,
                                 noise_candidates=noise_candidates,
                                 subdivision=subdivision,
                                 max_attempts=max_attempts)
    c = pair.certificate
    csv_metadata = {
        "g": int(c.genus), "subdivision": int(c.subdivision),
        "num_cells_rank_0": int(c.rank_counts[0]),
        "num_cells_rank_1": int(c.rank_counts[1]),
        "num_cells_rank_2": int(c.rank_counts[2]),
        "word_length": int(c.word_path_length),
    }
    full = asdict(c)
    full["is_relator_family"] = True
    return pair, csv_metadata, full


def build_relator_pair_copies_compatible(
    g, rng, noise_pairs_per_g, min_subdiv, max_subdiv, match_backtracks,
    max_tries, max_expansions, number_of_permuted_copies, low_rk, high_rk,
    *, pairing: str = "deep_vs_free",
    include_subcomplex_features: bool = True, binary_marking: bool = False,
):
    """Drop-in for build_tight_pair_copies_compatible."""
    if g != 2:
        raise ValueError("genus-2 only; set min_g=max_g=2")
    if min_subdiv != max_subdiv:
        raise ValueError("use a fixed subdivision for one evaluation")
    from fundamental_group_dataset import build_fundamental_group_copies_from_word

    derived = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
    nc = tuple(sorted({max(1, int(noise_pairs_per_g)),
                       max(1, int(noise_pairs_per_g) * 2),
                       max(1, int(noise_pairs_per_g) * 4)}))
    pair, meta, cert = generate_relator_pair_descriptor(
        seed=derived, pairing=pairing, noise_candidates=nc,
        subdivision=int(min_subdiv),
        max_attempts=max(5_000, int(max_tries) * int(max_expansions) * 5))

    def copies(word):
        return build_fundamental_group_copies_from_word(
            word, number_of_permuted_copies, low_rk, high_rk,
            include_subcomplex_features=include_subcomplex_features,
            binary_marking=binary_marking)

    return (copies(pair.surface_word_path), copies(pair.free_word_path),
            copies(pair.surface_word_path), copies(pair.surface_word_path),
            meta, cert)


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairing", default="deep_vs_free", choices=list(PAIRINGS))
    ap.add_argument("--pairs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(f"=== {a.pairing}: {PAIRINGS[a.pairing]} ===")
    print(f"{'seed':>5} {'attempts':>9} {'|V,E,F|':>18} {'chi':>4} {'H_*':>10} "
          f"{'wl':>3} {'len':>4} {'bt':>4}")
    for i in range(a.pairs):
        pr = generate_relator_pair(seed=a.seed + i, pairing=a.pairing)
        c = pr.certificate
        print(f"{a.seed+i:>5} {c.attempts:>9} {str(tuple(c.rank_counts)):>18} "
              f"{c.euler_characteristic:>4} {str(tuple(c.integral_betti)):>10} "
              f"{c.hasse_wl_rounds_to_stability:>3} {c.word_path_length:>4} "
              f"{c.backtrack_count:>4}")
