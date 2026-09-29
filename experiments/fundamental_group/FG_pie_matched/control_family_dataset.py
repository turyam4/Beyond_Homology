"""
control_family_dataset.py  --  E-G: pi_1-MATCHED CONTROL FAMILIES
==================================================================
Falsification test for reviewer concern R3: "separation may come from another
global structural feature created during realization."

The cross-class family pairs a SURFACE-group complex against a FREE-group
complex. If a model separates them, that could be pi_1 sensitivity -- or it
could be an artifact of how the two words get realized as complexes. This
module builds the control that tells those apart:

    surface_vs_surface : BOTH members freely reduce to [a1,b1][a2,b2]
                         (identical pi_1 = genus-2 surface group),
                         INDEPENDENT noise realizations.
    free_vs_free       : BOTH members freely reduce to the identity
                         (identical pi_1 = F_4),
                         INDEPENDENT noise realizations.

Both control families are pushed through the IDENTICAL gate stack as the
cross-class family. That is enforced structurally, not by convention: the
gate functions (local_incidence_profile, vertex_link_wl_profile,
stable_rank_coloured_hasse_wl, _rank_counts, _edge_coface_counts) are
IMPORTED from tight_fundamental_group_dataset.py and called on the control
pairs. They are not reimplemented here, so they cannot silently drift.

Gates applied (all of them, same code path as cross-class):
  * algebraic class certification of BOTH members (family-appropriate)
  * rank-wise cell counts, Euler characteristic
  * exact integral homology: torsion-free (1, 2g, 1), matching torsion
  * word length, abstract slot length
  * per-generator occurrence counts, signed exponent sums, slot multiset
  * node-token multiset of the realized path
  * immediate-inverse (backtrack) motif count
  * local incidence profile (vertex-edge / vertex-face / edge-coface / face size)
  * vertex-link WL profile
  * rank-coloured Hasse 1-WL equivalence to stabilisation

PLUS one gate the cross-class family does not need, and must not be omitted
here: DISTINCTNESS. Two same-class members could coincide as words (up to the
construction's own rotation/reversal symmetry), in which case the "control"
would degenerate into the existing surface-vs-surface RELIABILITY control
(same word, different permutation copies) and would sit at the floor for a
trivial reason. Such pairs are rejected, and the certificate records the fact.
See `require_distinct`.

INTERPRETATION (state this in the paper regardless of outcome):
  * Controls at the reliability floor  -> separation on the cross-class family
    is class-level, not a realization artifact.
  * Controls separating at a high rate -> the family measures realization
    structure. That is a real negative finding about the benchmark and should
    be reported as such.

CLI audit:
    python control_family_dataset.py --family surface_vs_surface --pairs 8 --seed 0
    python control_family_dataset.py --family free_vs_free       --pairs 8 --seed 0
"""

from __future__ import annotations

import argparse
import json
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

FAMILIES = ("surface_vs_surface", "free_vs_free")


# ---------------------------------------------------------------------------
# locate the FG_genus_2 checkout and import BOTH the generator core and the
# EXISTING gate functions (never reimplement a gate here)
# ---------------------------------------------------------------------------
def _fg_dir() -> Path:
    """Directory holding the active FG_genus_2 evaluator files."""
    env = os.environ.get("FG_GENUS2_DIR")
    candidates = []
    if env:
        candidates.append(Path(env))
    here = Path(__file__).resolve()
    candidates += [
        here.parent.parent / "FG_genus_2",
        here.parent.parent.parent / "fundamental_group" / "FG_genus_2",
    ]
    for c in candidates:
        if (c / "fundamental_group_dataset_genus2.py").is_file():
            return c
    raise RuntimeError(
        "Could not locate FG_genus_2 (needs fundamental_group_dataset_genus2.py). "
        "Set FG_GENUS2_DIR to that directory."
    )


FG_DIR = _fg_dir()
for _path in (FG_DIR, FG_DIR.parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from fg_core import (  # noqa: E402
    build_base_commutator_slots,
    create_complex_edges_faces,
    integral_homology,
    is_rotation,
    random_trivial_slots,
    slot_letter_counts,
    slot_signed_exponent_sums,
    verify_class_A_reduces_to_commutator,
    verify_class_B_reduces_to_identity,
    word_backtracks,
)
from fg_core import betti_numbers as _betti_numbers  # noqa: E402

# THE GATES -- imported, not reimplemented. This is what makes "passes every
# gate the cross-class pairs pass" a structural fact rather than a claim.
from fundamental_group_dataset_genus2 import (  # noqa: E402
    _edge_coface_counts,
    _rank_counts,
    local_incidence_profile,
    stable_rank_coloured_hasse_wl,
    vertex_link_wl_profile,
)


# ---------------------------------------------------------------------------
# certificate
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ControlPairCertificate:
    # --- identical field names to TightPairCertificate (drop-in downstream) ---
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
    # --- control-family specific ---
    control_family: str
    pi1_class: str
    members_pi1_identical: bool
    words_distinct_up_to_symmetry: bool
    backtracks_equal: bool
    backtrack_count: int


@dataclass
class ControlFundamentalGroupPair:
    """Field names mirror TightFundamentalGroupPair so this object is a
    drop-in for the caching/eval ladder. `surface_*` = member A,
    `free_*` = member B; in a control family BOTH members belong to the SAME
    pi_1 class, so the names carry no class meaning here -- use the
    `member_a_*` / `member_b_*` aliases when writing new code."""
    surface_word_path: List[int]
    free_word_path: List[int]
    surface_slots: List[Slot]
    free_slots: List[Slot]
    surface_edges: List[Edge]
    free_edges: List[Edge]
    surface_faces: List[Face]
    free_faces: List[Face]
    certificate: ControlPairCertificate

    # readable aliases
    @property
    def member_a_word_path(self) -> List[int]:
        return self.surface_word_path

    @property
    def member_b_word_path(self) -> List[int]:
        return self.free_word_path


# ---------------------------------------------------------------------------
# generation: two members of the SAME pi_1 class, shared realization params
# ---------------------------------------------------------------------------
def _generate_control_words(
    g: int,
    family: str,
    noise_pairs_per_g: int,
    rng: np.random.Generator,
    subdivision: int,
    match_backtracks: bool = True,
    max_tries: int = 100,
    max_expansions: int = 10,
):
    """Mirror of fg_core.generate_words, but emitting two SAME-class members.

    Both members share the subdivision level p, the generator relabelling
    (gen_map), the orientation flips (flip_map) and the noise-letter MULTISET
    -- exactly as the cross-class pair does. Only the arrangement differs:

      surface_vs_surface : same commutator base, same noise letters,
                           INDEPENDENT insertion positions.
      free_vs_free       : same letter pool, INDEPENDENT non-crossing
                           (Dyck) pairings.

    Sharing the letter multiset is what makes word length, per-generator
    occurrence counts, signed exponent sums and the node-token multiset match
    by construction, unconditionally -- the same mechanism the cross-class
    generator relies on. Backtrack counts are then matched by rejection
    sampling member B against member A, widening the SHARED pool by 2 letters
    and rebuilding BOTH members on failure (never padding one side, which
    would break length matching).
    """
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}, got {family!r}")
    if subdivision < 3:
        raise ValueError("subdivision must be >= 3 (p=2 is degenerate)")

    p = int(subdivision)
    gen_map = list(range(1, 2 * g + 1))
    rng.shuffle(gen_map)
    flip_map = rng.choice([True, False], size=2 * g + 1)

    def node_id(mapped_gen: int, t: int) -> int:
        return 1 + (mapped_gen - 1) * (p - 1) + (t - 1)

    def get_path(gen_idx: int, is_inverse: bool) -> List[int]:
        mapped_gen = gen_map[gen_idx - 1]
        mapped_inv = is_inverse ^ flip_map[mapped_gen]
        inner = [node_id(mapped_gen, t) for t in range(1, p)]
        if mapped_inv:
            inner = inner[::-1]
        return [0] + inner

    def slots_to_path(slots: List[Slot]) -> List[int]:
        path: List[int] = []
        for gen_idx, is_inv in slots:
            path += get_path(gen_idx, is_inv)
        return path

    def make_surface_member(noise_letters: List[int]) -> List[Slot]:
        """Commutator base + Tietze insertions of xx^-1 at random positions.
        Free-reduces to the base commutator for ANY insertion positions."""
        slots = build_base_commutator_slots(g)
        for letter in noise_letters:
            at = int(rng.integers(0, len(slots) + 1))
            slots = slots[:at] + [(letter, False), (letter, True)] + slots[at:]
        return slots

    def draw_noise(extra_budget: int) -> List[int]:
        n = g * noise_pairs_per_g + extra_budget
        return [int(rng.integers(1, 2 * g + 1)) for _ in range(n)]

    extra_budget = 0
    a_slots = b_slots = None
    best_gap = None

    for expansion in range(max_expansions):
        noise_letters = draw_noise(extra_budget)
        base_and_noise = list(range(1, 2 * g + 1)) + noise_letters

        if family == "surface_vs_surface":
            a_slots = make_surface_member(noise_letters)

            def draw_member():
                return make_surface_member(noise_letters)
        else:  # free_vs_free
            a_slots = random_trivial_slots(base_and_noise, rng)

            def draw_member():
                return random_trivial_slots(base_and_noise, rng)

        target = word_backtracks(slots_to_path(a_slots))
        best_slots, best_gap = None, None
        tries = max_tries if match_backtracks else 1
        for _ in range(tries):
            cand = draw_member()
            gap = (abs(word_backtracks(slots_to_path(cand)) - target)
                   if match_backtracks else 0)
            if best_gap is None or gap < best_gap:
                best_slots, best_gap = cand, gap
            if gap == 0:
                break
        b_slots = best_slots

        if (not match_backtracks) or best_gap == 0 or expansion == max_expansions - 1:
            break
        extra_budget += 2  # widen SHARED pool -- both members stay matched

    WA, WB = slots_to_path(a_slots), slots_to_path(b_slots)

    def randomize_word(W: List[int]) -> List[int]:
        period = p
        shift = int(rng.integers(0, len(W) // period)) * period
        shifted = W[shift:] + W[:shift]
        if rng.random() > 0.5:
            shifted = shifted[::-1]
        return shifted

    return randomize_word(WA), randomize_word(WB), p, (a_slots, b_slots)


def _words_distinct(word_a: List[int], word_b: List[int]) -> bool:
    """False iff the two realized paths coincide up to the construction's own
    symmetries (cyclic rotation and reversal). Such a pair would silently
    reduce the control to the existing same-word reliability control."""
    if len(word_a) != len(word_b):
        return True
    if is_rotation(word_a, word_b):
        return False
    if is_rotation(word_a, list(reversed(word_b))):
        return False
    return True


# ---------------------------------------------------------------------------
# certification -- same gate stack as _certify_candidate, family-aware
# ---------------------------------------------------------------------------
def _certify_control_candidate(
    genus: int,
    family: str,
    configured_noise: int,
    attempts: int,
    subdivision: int,
    word_a: List[int],
    word_b: List[int],
    slots_a: List[Slot],
    slots_b: List[Slot],
    require_distinct: bool = True,
):
    edges_a, faces_a = create_complex_edges_faces(word_a)
    edges_b, faces_b = create_complex_edges_faces(word_b)

    # --- algebraic class certification: BOTH members, family-appropriate ---
    if family == "surface_vs_surface":
        ok_a = verify_class_A_reduces_to_commutator(slots_a, genus)
        ok_b = verify_class_A_reduces_to_commutator(slots_b, genus)
        pi1_class = "surface group <a1,b1,a2,b2 | [a1,b1][a2,b2]>"
    else:
        ok_a = verify_class_B_reduces_to_identity(slots_a)
        ok_b = verify_class_B_reduces_to_identity(slots_b)
        pi1_class = "free group F_{2g}"
    if not (ok_a and ok_b):
        raise RuntimeError(
            f"{family}: a member failed algebraic pi_1 certification "
            "(generator bug -- both members must lie in the same class)"
        )

    # --- distinctness (control-family specific; see module docstring) ---
    distinct = _words_distinct(word_a, word_b)
    if require_distinct and not distinct:
        return None

    # --- cell counts / Euler ---
    counts_a = _rank_counts(edges_a, faces_a)
    counts_b = _rank_counts(edges_b, faces_b)
    if counts_a != counts_b:
        return None
    chi_a = counts_a[0] - counts_a[1] + counts_a[2]
    chi_b = counts_b[0] - counts_b[1] + counts_b[2]
    if chi_a != chi_b:
        return None

    # --- cheap rational Betti pre-filter ---
    expected_betti = (1, 2 * genus, 1)
    betti_a = tuple(int(v) for v in _betti_numbers(edges_a, faces_a))
    betti_b = tuple(int(v) for v in _betti_numbers(edges_b, faces_b))
    if betti_a != expected_betti or betti_b != expected_betti:
        return None

    # --- word-level matching ---
    if len(word_a) != len(word_b):
        return None
    if len(slots_a) != len(slots_b):
        return None
    if slot_letter_counts(slots_a) != slot_letter_counts(slots_b):
        return None
    if Counter(slots_a) != Counter(slots_b):
        return None
    if slot_signed_exponent_sums(slots_a, genus) != slot_signed_exponent_sums(slots_b, genus):
        return None
    if Counter(word_a) != Counter(word_b):
        return None
    bt_a, bt_b = word_backtracks(word_a), word_backtracks(word_b)
    if bt_a != bt_b:
        return None

    # --- local structure gates (imported from the tight generator) ---
    if local_incidence_profile(edges_a, faces_a) != local_incidence_profile(edges_b, faces_b):
        return None
    if vertex_link_wl_profile(edges_a, faces_a) != vertex_link_wl_profile(edges_b, faces_b):
        return None

    # --- rank-coloured Hasse 1-WL to stabilisation ---
    wl_equivalent, wl_rounds = stable_rank_coloured_hasse_wl(
        (edges_a, faces_a), (edges_b, faces_b)
    )
    if not wl_equivalent:
        return None

    # --- exact integral homology (last: most expensive) ---
    hom_a = integral_homology(edges_a, faces_a)
    hom_b = integral_homology(edges_b, faces_b)
    if hom_a["betti"] != expected_betti or hom_b["betti"] != expected_betti:
        return None
    if not (hom_a["torsion_free"] and hom_b["torsion_free"]):
        return None
    if hom_a["torsion_H1"] != hom_b["torsion_H1"]:
        return None

    boundary_edges = sum(v == 1 for v in _edge_coface_counts(edges_a, faces_a).values())
    certificate = ControlPairCertificate(
        genus=genus,
        subdivision=subdivision,
        configured_noise_pairs_per_g=configured_noise,
        attempts=attempts,
        rank_counts=counts_a,
        euler_characteristic=chi_a,
        betti_numbers=betti_a,
        integral_betti=tuple(int(v) for v in hom_a["betti"]),
        torsion_H1=tuple(hom_a["torsion_H1"]),
        integral_homology_torsion_free=True,
        word_path_length=len(word_a),
        abstract_word_length=len(slots_a),
        boundary_edges=boundary_edges,
        hasse_wl_rounds_to_stability=wl_rounds,
        hasse_wl_equivalent=True,
        vertex_link_wl_profiles_equal=True,
        local_incidence_profiles_equal=True,
        control_family=family,
        pi1_class=pi1_class,
        members_pi1_identical=True,
        words_distinct_up_to_symmetry=distinct,
        backtracks_equal=True,
        backtrack_count=int(bt_a),
    )
    return certificate, edges_a, faces_a, edges_b, faces_b


# ---------------------------------------------------------------------------
# public API -- mirrors generate_tight_pair / *_descriptor
# ---------------------------------------------------------------------------
def generate_control_pair(
    *,
    seed: int,
    family: str,
    genus: int = 2,
    noise_candidates: Sequence[int] = (2, 4, 8),
    subdivision: int = 3,
    max_attempts: int = 5_000,
    match_backtracks: bool = True,
    backtrack_max_tries: int = 100,
    backtrack_max_expansions: int = 10,
    require_distinct: bool = True,
) -> ControlFundamentalGroupPair:
    """Generate one certified pi_1-MATCHED control pair.

    Genus 2 only, matching the cross-class benchmark it controls for.
    """
    if genus != 2:
        raise ValueError(
            "The control families mirror the genus-2 cross-class benchmark; "
            "genus != 2 is not audited. Do not broaden without a new audit."
        )
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}, got {family!r}")
    if not noise_candidates or any(v < 1 for v in noise_candidates):
        raise ValueError("noise_candidates must contain positive integers")

    rng = np.random.default_rng(seed)
    for attempt in range(1, max_attempts + 1):
        noise = int(noise_candidates[(attempt - 1) % len(noise_candidates)])
        word_a, word_b, p, (slots_a, slots_b) = _generate_control_words(
            g=genus,
            family=family,
            noise_pairs_per_g=noise,
            rng=rng,
            subdivision=subdivision,
            match_backtracks=match_backtracks,
            max_tries=backtrack_max_tries,
            max_expansions=backtrack_max_expansions,
        )
        certified = _certify_control_candidate(
            genus, family, noise, attempt, p,
            word_a, word_b, slots_a, slots_b,
            require_distinct=require_distinct,
        )
        if certified is None:
            continue
        cert, edges_a, faces_a, edges_b, faces_b = certified
        return ControlFundamentalGroupPair(
            surface_word_path=word_a,
            free_word_path=word_b,
            surface_slots=slots_a,
            free_slots=slots_b,
            surface_edges=edges_a,
            free_edges=edges_b,
            surface_faces=faces_a,
            free_faces=faces_b,
            certificate=cert,
        )

    raise RuntimeError(
        f"No certified {family} control pair found in {max_attempts} attempts. "
        "Increase max_attempts or change the seed; do NOT relax the filters -- "
        "the control is only meaningful if it passes the same gates."
    )


def generate_control_pair_descriptor(
    *,
    seed: int,
    family: str,
    noise_candidates: Sequence[int] = (2, 4, 8),
    subdivision: int = 3,
    max_attempts: int = 5_000,
):
    """Drop-in replacement for `generate_tight_pair_descriptor`.

    Returns (pair, csv_metadata, full_certificate) with the identical shape,
    so the phased cache builder consumes it unchanged.
    """
    pair = generate_control_pair(
        seed=seed,
        family=family,
        noise_candidates=noise_candidates,
        subdivision=subdivision,
        max_attempts=max_attempts,
    )
    cert = pair.certificate
    csv_metadata = {
        "g": int(cert.genus),
        "subdivision": int(cert.subdivision),
        "num_cells_rank_0": int(cert.rank_counts[0]),
        "num_cells_rank_1": int(cert.rank_counts[1]),
        "num_cells_rank_2": int(cert.rank_counts[2]),
        "word_length": int(cert.word_path_length),
    }
    full_certificate = asdict(cert)
    full_certificate["control_family"] = family
    full_certificate["is_pi1_matched_control"] = True
    return pair, csv_metadata, full_certificate


def build_control_pair_copies_compatible(
    g: int,
    rng: np.random.Generator,
    noise_pairs_per_g: int,
    min_subdiv: int,
    max_subdiv: int,
    match_backtracks: bool,
    max_tries: int,
    max_expansions: int,
    number_of_permuted_copies: int,
    low_rk: int,
    high_rk: int,
    *,
    family: str,
    include_subcomplex_features: bool = True,
    binary_marking: bool = False,
):
    """Drop-in replacement for `build_tight_pair_copies_compatible`.

    Returns the same 6-tuple: (a_target, b_target, rel_a, rel_b,
    csv_metadata, full_certificate). The reliability copy-sets are two
    independent permutation sets of MEMBER A, exactly as the cross-class
    builder uses two independent sets of the surface word -- so the
    reliability floor is measured identically in both families.
    """
    if g != 2:
        raise ValueError("control families are genus-2 only; set min_g=max_g=2")
    if min_subdiv != max_subdiv:
        raise ValueError("use a fixed subdivision for one evaluation")
    if not match_backtracks:
        raise ValueError("control families require match_backtracks=True")

    from fundamental_group_dataset import build_fundamental_group_copies_from_word

    derived_seed = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
    noise_candidates = tuple(sorted({
        max(1, int(noise_pairs_per_g)),
        max(1, int(noise_pairs_per_g) * 2),
        max(1, int(noise_pairs_per_g) * 4),
    }))
    pair, csv_metadata, full_certificate = generate_control_pair_descriptor(
        seed=derived_seed,
        family=family,
        noise_candidates=noise_candidates,
        subdivision=int(min_subdiv),
        max_attempts=max(5_000, int(max_tries) * int(max_expansions) * 5),
    )

    def copies(word):
        return build_fundamental_group_copies_from_word(
            word, number_of_permuted_copies, low_rk, high_rk,
            include_subcomplex_features=include_subcomplex_features,
            binary_marking=binary_marking,
        )

    return (
        copies(pair.surface_word_path),   # member A  (target, first)
        copies(pair.free_word_path),      # member B  (target, second)
        copies(pair.surface_word_path),   # reliability A
        copies(pair.surface_word_path),   # reliability B
        csv_metadata,
        full_certificate,
    )


# ---------------------------------------------------------------------------
# CLI audit
# ---------------------------------------------------------------------------
def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=FAMILIES + ("both",), default="both")
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subdivision", type=int, default=3)
    ap.add_argument("--max-attempts", type=int, default=5_000)
    ap.add_argument("--json", action="store_true", help="dump full certificates")
    args = ap.parse_args()

    families = FAMILIES if args.family == "both" else (args.family,)
    all_certs = []
    for fam in families:
        print(f"\n=== {fam} ===")
        print(f"{'seed':>4} {'attempts':>9} {'|V,E,F|':>18} {'chi':>4} "
              f"{'H_*':>10} {'tors':>5} {'wl':>3} {'len':>4} {'distinct':>9}")
        attempts_seen = []
        for i in range(args.pairs):
            pair = generate_control_pair(
                seed=args.seed + i, family=fam,
                subdivision=args.subdivision, max_attempts=args.max_attempts,
            )
            c = pair.certificate
            attempts_seen.append(c.attempts)
            all_certs.append(asdict(c))
            print(f"{args.seed+i:>4} {c.attempts:>9} {str(tuple(c.rank_counts)):>18} "
                  f"{c.euler_characteristic:>4} {str(tuple(c.integral_betti)):>10} "
                  f"{str(list(c.torsion_H1)):>5} {c.hasse_wl_rounds_to_stability:>3} "
                  f"{c.word_path_length:>4} {str(c.words_distinct_up_to_symmetry):>9}")
        print(f"  attempts/pair: min={min(attempts_seen)} "
              f"mean={sum(attempts_seen)/len(attempts_seen):.1f} max={max(attempts_seen)}")
    if args.json:
        print(json.dumps(all_certs, indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()
