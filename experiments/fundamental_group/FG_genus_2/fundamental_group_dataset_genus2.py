"""
Tight genus-2 surface-group versus free-group benchmark generator.

This file is intentionally separate from the upstream SMCN checkout.  It wraps
the existing, algebraically verified generator and adds rejection filters for
the two leakage channels found in the original default dataset:

1. rank-coloured Hasse 1-WL separation; and
2. differing local vertex-link statistics.

Accepted pairs have:
  * surface group  <a1,b1,a2,b2 | [a1,b1][a2,b2]>;
  * free group F4 (the other relator freely reduces to the identity);
  * equal rank-wise cell counts and Euler characteristic;
  * equal rational Betti vector (1, 4, 1);
  * equal word length, signed token counts, and generator frequencies;
  * equal 1-skeleton, coface, face-size, and vertex-link WL profiles;
  * rank-coloured Hasse graphs indistinguishable by 1-WL to stabilization.

This makes the data substantially tighter and harder for local/WL-level
baselines.  It does NOT prove that every possible vanilla SMCN must fail:
SMCN is stronger than ordinary 1-WL on some inputs.  Any paper must still
report empirical SMCN results and must call this a pi_1-sensitive benchmark,
not a proof of full fundamental-group recognition or non-recognition.

Typical use after copying this file into
SMCN/new_experiments/fundamental_group/:

    from tight_fundamental_group_dataset import (
        generate_tight_pair,
        build_tight_fundamental_group_pair_copies,
    )

    pair = generate_tight_pair(seed=0)
    print(pair.certificate)

Command-line audit from this workspace:

    python tight_fundamental_group_dataset.py --pairs 8 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, MutableMapping, Sequence, Tuple

import networkx as nx
import numpy as np

warnings.filterwarnings(
    "ignore",
    message=r"The hashes produced for graphs without node or edge attributes changed in v3\.5.*",
    category=UserWarning,
    module=r"networkx\.algorithms\.graph_hashing",
)


Slot = Tuple[int, bool]
Edge = Tuple[int, int]
Face = List[int]


def _smcn_root() -> Path | None:
    """Locate the read-only SMCN checkout (needed only for the heavy
    data.complex_data / models machinery imported lazily by the streamed
    copy builder -- NOT for the pure-python generator itself)."""
    env_root = os.environ.get("SMCN_ROOT")
    candidates = []
    if env_root:
        candidates.append(Path(env_root))
    candidates.append(Path(__file__).resolve().parents[3])
    for root in candidates:
        if (root / "data" / "complex_data.py").is_file():
            return root
    return None


def _load_upstream_modules():
    """Import the pure-python generator core.

    This file ships ALONGSIDE a local copy of fg_core.py (which adds the
    exact integral-homology certificate). We import that local copy, not the
    upstream one, so the leakage filters and the homology certificate use a
    single, self-contained math implementation. The SMCN root is added to
    sys.path only so the lazily-imported streamed copy builder can reach
    data.complex_data / models; the generator here does not need it.
    """
    self_dir = Path(__file__).resolve().parent
    if str(self_dir) not in sys.path:
        sys.path.insert(0, str(self_dir))
    root = _smcn_root()
    if root is not None:
        for path in (str(root.parent), str(root)):
            if path not in sys.path:
                sys.path.insert(0, path)

    try:
        from fg_core import (
            betti_numbers,
            create_complex_edges_faces,
            free_reduce,
            generate_words,
            integral_homology,
            signed_word,
            slot_letter_counts,
            slot_signed_exponent_sums,
            verify_class_A_reduces_to_commutator,
            verify_class_B_reduces_to_identity,
            word_backtracks,
        )
    except ImportError as exc:
        raise ImportError(
            "Could not import the local fg_core.py that must sit next to this "
            f"file (looked in {self_dir})."
        ) from exc

    return {
        "betti_numbers": betti_numbers,
        "create_complex_edges_faces": create_complex_edges_faces,
        "free_reduce": free_reduce,
        "generate_words": generate_words,
        "integral_homology": integral_homology,
        "signed_word": signed_word,
        "slot_letter_counts": slot_letter_counts,
        "slot_signed_exponent_sums": slot_signed_exponent_sums,
        "verify_A": verify_class_A_reduces_to_commutator,
        "verify_B": verify_class_B_reduces_to_identity,
        "word_backtracks": word_backtracks,
    }


UPSTREAM = _load_upstream_modules()


@dataclass(frozen=True)
class TightPairCertificate:
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


@dataclass
class TightFundamentalGroupPair:
    surface_word_path: List[int]
    free_word_path: List[int]
    surface_slots: List[Slot]
    free_slots: List[Slot]
    surface_edges: List[Edge]
    free_edges: List[Edge]
    surface_faces: List[Face]
    free_faces: List[Face]
    certificate: TightPairCertificate


def _vertices(edges: Sequence[Edge]) -> List[int]:
    return sorted({v for edge in edges for v in edge})


def _edge_coface_counts(edges: Sequence[Edge], faces: Sequence[Face]) -> Counter:
    counts = Counter({tuple(sorted(edge)): 0 for edge in edges})
    for face in faces:
        a, b, c = sorted(face)
        for edge in ((a, b), (a, c), (b, c)):
            counts[edge] += 1
    return counts


def local_incidence_profile(
    edges: Sequence[Edge], faces: Sequence[Face]
) -> Tuple:
    """Marginal rank-conditioned incidence statistics used as leakage checks."""
    vertex_edge_degree = Counter()
    vertex_face_degree = Counter()
    for u, v in edges:
        vertex_edge_degree[u] += 1
        vertex_edge_degree[v] += 1
    for face in faces:
        for v in face:
            vertex_face_degree[v] += 1

    edge_cofaces = _edge_coface_counts(edges, faces)
    return (
        tuple(sorted(Counter(vertex_edge_degree.values()).items())),
        tuple(sorted(Counter(vertex_face_degree.values()).items())),
        tuple(sorted(Counter(edge_cofaces.values()).items())),
        tuple(sorted(Counter(len(face) for face in faces).items())),
        sum(value == 1 for value in edge_cofaces.values()),
        sum(value > 2 for value in edge_cofaces.values()),
    )


def _vertex_link_graphs(
    edges: Sequence[Edge], faces: Sequence[Face]
) -> List[nx.Graph]:
    """Return the simplicial link graph of every vertex."""
    links: List[nx.Graph] = []
    for vertex in _vertices(edges):
        graph = nx.Graph()
        for face in faces:
            if vertex not in face:
                continue
            other = [v for v in face if v != vertex]
            if len(other) != 2:
                raise ValueError(f"non-triangular or degenerate face encountered: {face}")
            graph.add_edge(other[0], other[1])
        links.append(graph)
    return links


def _link_wl_key(graph: nx.Graph) -> Tuple:
    """Strong local fingerprint; exact enough for filtering, not a proof of isomorphism."""
    degree_histogram = tuple(sorted(Counter(dict(graph.degree()).values()).items()))
    components = nx.number_connected_components(graph) if graph.number_of_nodes() else 0
    cycle_rank = graph.number_of_edges() - graph.number_of_nodes() + components
    iterations = max(8, min(32, graph.number_of_nodes()))
    wl_hash = nx.weisfeiler_lehman_graph_hash(graph, iterations=iterations)
    return (
        graph.number_of_nodes(),
        graph.number_of_edges(),
        components,
        cycle_rank,
        degree_histogram,
        wl_hash,
    )


def vertex_link_wl_profile(
    edges: Sequence[Edge], faces: Sequence[Face]
) -> Counter:
    """Multiset of stable-ish WL fingerprints of all vertex links."""
    return Counter(_link_wl_key(graph) for graph in _vertex_link_graphs(edges, faces))


def _hasse_adjacency(
    edges: Sequence[Edge], faces: Sequence[Face]
) -> Dict[Tuple[str, int], List[Tuple[str, int]]]:
    adjacency: MutableMapping[Tuple[str, int], List[Tuple[str, int]]] = defaultdict(list)
    edge_index = {tuple(sorted(edge)): i for i, edge in enumerate(edges)}

    for vertex in _vertices(edges):
        adjacency[("v", vertex)]

    for edge, index in edge_index.items():
        edge_node = ("e", index)
        adjacency[edge_node]
        for vertex in edge:
            vertex_node = ("v", vertex)
            adjacency[edge_node].append(vertex_node)
            adjacency[vertex_node].append(edge_node)

    for face_index, face in enumerate(faces):
        face_node = ("f", face_index)
        adjacency[face_node]
        a, b, c = sorted(face)
        for edge in ((a, b), (a, c), (b, c)):
            edge_node = ("e", edge_index[edge])
            adjacency[face_node].append(edge_node)
            adjacency[edge_node].append(face_node)

    return dict(adjacency)


def stable_rank_coloured_hasse_wl(
    first: Tuple[Sequence[Edge], Sequence[Face]],
    second: Tuple[Sequence[Edge], Sequence[Face]],
) -> Tuple[bool, int]:
    """Joint rank-coloured 1-WL, continued until the joint partition stabilizes."""
    adjacency_a = _hasse_adjacency(*first)
    adjacency_b = _hasse_adjacency(*second)
    rank_colour = {"v": 0, "e": 1, "f": 2}
    colours_a = {node: rank_colour[node[0]] for node in adjacency_a}
    colours_b = {node: rank_colour[node[0]] for node in adjacency_b}

    if Counter(colours_a.values()) != Counter(colours_b.values()):
        return False, 0

    previous_joint_classes = len(set(colours_a.values()) | set(colours_b.values()))
    max_rounds = len(adjacency_a) + len(adjacency_b) + 1

    for round_index in range(1, max_rounds + 1):
        signatures_a = {
            node: (
                colours_a[node],
                tuple(sorted(colours_a[nbr] for nbr in adjacency_a[node])),
            )
            for node in adjacency_a
        }
        signatures_b = {
            node: (
                colours_b[node],
                tuple(sorted(colours_b[nbr] for nbr in adjacency_b[node])),
            )
            for node in adjacency_b
        }

        palette = {
            signature: index
            for index, signature in enumerate(
                sorted(
                    set(signatures_a.values()) | set(signatures_b.values()),
                    key=repr,
                )
            )
        }
        colours_a = {node: palette[sig] for node, sig in signatures_a.items()}
        colours_b = {node: palette[sig] for node, sig in signatures_b.items()}

        if Counter(colours_a.values()) != Counter(colours_b.values()):
            return False, round_index

        joint_classes = len(set(colours_a.values()) | set(colours_b.values()))
        if joint_classes == previous_joint_classes:
            return True, round_index
        previous_joint_classes = joint_classes

    raise RuntimeError("1-WL did not stabilize within the finite-graph bound")


def _rank_counts(edges: Sequence[Edge], faces: Sequence[Face]) -> Tuple[int, int, int]:
    return len(_vertices(edges)), len(edges), len(faces)


def _certify_candidate(
    genus: int,
    configured_noise: int,
    attempts: int,
    subdivision: int,
    surface_word: List[int],
    free_word: List[int],
    surface_slots: List[Slot],
    free_slots: List[Slot],
) -> Tuple[
    TightPairCertificate,
    List[Edge],
    List[Face],
    List[Edge],
    List[Face],
] | None:
    create = UPSTREAM["create_complex_edges_faces"]
    surface_edges, surface_faces = create(surface_word)
    free_edges, free_faces = create(free_word)

    if not UPSTREAM["verify_A"](surface_slots, genus):
        raise RuntimeError("surface relator failed algebraic certification")
    if not UPSTREAM["verify_B"](free_slots):
        raise RuntimeError("free relator failed algebraic certification")

    surface_counts = _rank_counts(surface_edges, surface_faces)
    free_counts = _rank_counts(free_edges, free_faces)
    if surface_counts != free_counts:
        return None

    chi_surface = surface_counts[0] - surface_counts[1] + surface_counts[2]
    chi_free = free_counts[0] - free_counts[1] + free_counts[2]
    if chi_surface != chi_free:
        return None

    expected_betti = (1, 2 * genus, 1)
    surface_betti = tuple(int(v) for v in UPSTREAM["betti_numbers"](surface_edges, surface_faces))
    free_betti = tuple(int(v) for v in UPSTREAM["betti_numbers"](free_edges, free_faces))
    if surface_betti != expected_betti or free_betti != expected_betti:
        return None

    if len(surface_word) != len(free_word):
        return None
    if len(surface_slots) != len(free_slots):
        return None
    if UPSTREAM["slot_letter_counts"](surface_slots) != UPSTREAM["slot_letter_counts"](free_slots):
        return None
    if Counter(surface_slots) != Counter(free_slots):
        return None
    if UPSTREAM["slot_signed_exponent_sums"](
        surface_slots, genus
    ) != UPSTREAM["slot_signed_exponent_sums"](free_slots, genus):
        return None
    if Counter(surface_word) != Counter(free_word):
        return None
    if UPSTREAM["word_backtracks"](surface_word) != UPSTREAM["word_backtracks"](free_word):
        return None

    surface_local = local_incidence_profile(surface_edges, surface_faces)
    free_local = local_incidence_profile(free_edges, free_faces)
    if surface_local != free_local:
        return None

    surface_links = vertex_link_wl_profile(surface_edges, surface_faces)
    free_links = vertex_link_wl_profile(free_edges, free_faces)
    if surface_links != free_links:
        return None

    wl_equivalent, wl_rounds = stable_rank_coloured_hasse_wl(
        (surface_edges, surface_faces),
        (free_edges, free_faces),
    )
    if not wl_equivalent:
        return None

    # Exact integral homology gate (runs at most once per accepted pair, after
    # all cheap filters + WL have passed). Requires BOTH classes to have
    # torsion-free H_* = (Z, Z^2g, Z). This is strictly stronger than the
    # float rational-Betti pre-filter above and closes the "matching rational
    # Betti could hide Z/2 torsion" objection.
    expected_betti = (1, 2 * genus, 1)
    hom_surface = UPSTREAM["integral_homology"](surface_edges, surface_faces)
    hom_free = UPSTREAM["integral_homology"](free_edges, free_faces)
    if hom_surface["betti"] != expected_betti or hom_free["betti"] != expected_betti:
        return None
    if not (hom_surface["torsion_free"] and hom_free["torsion_free"]):
        return None
    if hom_surface["torsion_H1"] != hom_free["torsion_H1"]:
        return None

    edge_cofaces = _edge_coface_counts(surface_edges, surface_faces)
    boundary_edges = sum(value == 1 for value in edge_cofaces.values())
    certificate = TightPairCertificate(
        genus=genus,
        subdivision=subdivision,
        configured_noise_pairs_per_g=configured_noise,
        attempts=attempts,
        rank_counts=surface_counts,
        euler_characteristic=chi_surface,
        betti_numbers=surface_betti,
        integral_betti=tuple(int(v) for v in hom_surface["betti"]),
        torsion_H1=tuple(hom_surface["torsion_H1"]),
        integral_homology_torsion_free=True,
        word_path_length=len(surface_word),
        abstract_word_length=len(surface_slots),
        boundary_edges=boundary_edges,
        hasse_wl_rounds_to_stability=wl_rounds,
        hasse_wl_equivalent=True,
        vertex_link_wl_profiles_equal=True,
        local_incidence_profiles_equal=True,
    )
    return (
        certificate,
        surface_edges,
        surface_faces,
        free_edges,
        free_faces,
    )


def generate_tight_pair(
    *,
    seed: int = 0,
    genus: int = 2,
    noise_candidates: Sequence[int] = (2, 4, 8),
    subdivision: int = 3,
    max_attempts: int = 5_000,
    backtrack_max_tries: int = 100,
    backtrack_max_expansions: int = 10,
) -> TightFundamentalGroupPair:
    """Generate one certified, leakage-filtered pair.

    Genus two is deliberate: it is the smallest non-abelian closed orientable
    surface group and has a practical acceptance rate under the strict filters.
    Higher genera should not be advertised until separately audited.
    """
    if genus != 2:
        raise ValueError(
            "This strict generator is currently certified/tuned only for genus=2. "
            "Do not broaden the research claim without a new leakage audit."
        )
    if subdivision < 3:
        raise ValueError("subdivision must be at least 3")
    if not noise_candidates or any(value < 1 for value in noise_candidates):
        raise ValueError("noise_candidates must contain positive integers")

    rng = np.random.default_rng(seed)
    generate_words = UPSTREAM["generate_words"]

    for attempt in range(1, max_attempts + 1):
        noise = int(noise_candidates[(attempt - 1) % len(noise_candidates)])
        surface_word, free_word, p, slots = generate_words(
            genus,
            noise,
            rng,
            min_subdiv=subdivision,
            max_subdiv=subdivision,
            match_backtracks=True,
            max_tries=backtrack_max_tries,
            max_expansions=backtrack_max_expansions,
            return_slots=True,
        )
        surface_slots, free_slots = slots
        certified = _certify_candidate(
            genus,
            noise,
            attempt,
            p,
            surface_word,
            free_word,
            surface_slots,
            free_slots,
        )
        if certified is None:
            continue

        certificate, surface_edges, surface_faces, free_edges, free_faces = certified
        return TightFundamentalGroupPair(
            surface_word_path=surface_word,
            free_word_path=free_word,
            surface_slots=surface_slots,
            free_slots=free_slots,
            surface_edges=surface_edges,
            free_edges=free_edges,
            surface_faces=surface_faces,
            free_faces=free_faces,
            certificate=certificate,
        )

    raise RuntimeError(
        f"No tight genus-{genus} pair found in {max_attempts} attempts. "
        "Increase max_attempts or change the seed; do not silently relax the filters."
    )


def build_tight_fundamental_group_pair_copies(
    *,
    seed: int,
    number_of_permuted_copies: int = 32,
    low_rk: int = 0,
    high_rk: int = 1,
    noise_candidates: Sequence[int] = (2, 4, 8),
    subdivision: int = 3,
    max_attempts: int = 5_000,
    return_pair: bool = False,
    include_subcomplex_features: bool = True,
    binary_marking: bool = False,
):
    """Return streamed SMCN copies for one accepted pair.

    Returns SIX values:
      (surface_target, free_target, surface_rel_a, surface_rel_b,
       csv_metadata, full_certificate)

    csv_metadata contains ONLY the keys the evaluator writes into its results
    CSV (g, subdivision, num_cells_rank_0/1/2, word_length) -- this is the fix
    for the schema mismatch that would otherwise crash csv.DictWriter after a
    full run. full_certificate is the complete, rich TightPairCertificate dict
    (all invariants + attempts + homology/torsion + WL rounds); the evaluator
    streams it to a certificates.jsonl sidecar instead of the CSV.

    With `return_pair=True`, additionally appends the accepted `pair`, which
    retains signed ordered slots for a future word-aware ablation. Vanilla SMCN
    receives only the triangulated ComplexData objects.
    """
    pair = generate_tight_pair(
        seed=seed,
        noise_candidates=noise_candidates,
        subdivision=subdivision,
        max_attempts=max_attempts,
    )

    try:
        from fundamental_group_dataset import (
            build_fundamental_group_copies_from_word,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import fundamental_group_dataset.py from the SMCN checkout."
        ) from exc

    surface_target = build_fundamental_group_copies_from_word(
        pair.surface_word_path,
        number_of_permuted_copies,
        low_rk,
        high_rk,
        include_subcomplex_features=include_subcomplex_features,
        binary_marking=binary_marking,
    )
    free_target = build_fundamental_group_copies_from_word(
        pair.free_word_path,
        number_of_permuted_copies,
        low_rk,
        high_rk,
        include_subcomplex_features=include_subcomplex_features,
        binary_marking=binary_marking,
    )
    surface_reliability_a = build_fundamental_group_copies_from_word(
        pair.surface_word_path,
        number_of_permuted_copies,
        low_rk,
        high_rk,
        include_subcomplex_features=include_subcomplex_features,
        binary_marking=binary_marking,
    )
    surface_reliability_b = build_fundamental_group_copies_from_word(
        pair.surface_word_path,
        number_of_permuted_copies,
        low_rk,
        high_rk,
        include_subcomplex_features=include_subcomplex_features,
        binary_marking=binary_marking,
    )

    cert = pair.certificate
    # CSV-safe metadata: EXACTLY the columns the evaluator tabulates. Anything
    # richer goes into full_certificate (JSONL sidecar), never into the CSV row.
    csv_metadata = {
        "g": int(cert.genus),
        "subdivision": int(cert.subdivision),
        "num_cells_rank_0": int(cert.rank_counts[0]),
        "num_cells_rank_1": int(cert.rank_counts[1]),
        "num_cells_rank_2": int(cert.rank_counts[2]),
        "word_length": int(cert.word_path_length),
    }
    full_certificate = asdict(cert)
    full_certificate["surface_group"] = "<a1,b1,a2,b2 | [a1,b1][a2,b2]>"
    full_certificate["free_group"] = "F4"

    result = (
        surface_target,
        free_target,
        surface_reliability_a,
        surface_reliability_b,
        csv_metadata,
        full_certificate,
    )
    return result + (pair,) if return_pair else result


def generate_tight_pair_descriptor(
    *,
    seed: int,
    noise_candidates: Sequence[int] = (2, 4, 8),
    subdivision: int = 3,
    max_attempts: int = 5_000,
):
    """Generate/certify a pair without materializing any permutation copies."""
    pair = generate_tight_pair(
        seed=seed,
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
    full_certificate["surface_group"] = "<a1,b1,a2,b2 | [a1,b1][a2,b2]>"
    full_certificate["free_group"] = "F4"
    return pair, csv_metadata, full_certificate


def build_tight_pair_copies_compatible(
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
    include_subcomplex_features: bool = True,
    binary_marking: bool = False,
):
    """Drop-in-ish replacement for the original streamed pair builder.

    NOTE: returns a SIX-tuple (extra trailing full_certificate) rather than the
    original five, so the patched evaluator (eval_fg_smcn.py in this subfolder)
    consumes it directly rather than via `as build_fundamental_group_pair_copies`
    aliasing. The strict dataset is genus two only and uses a fixed subdivision
    within a run. These restrictions are deliberate research controls.
    """
    if g != 2:
        raise ValueError(
            "The tight benchmark is genus-2 only. Configure min_g=max_g=2."
        )
    if min_subdiv != max_subdiv:
        raise ValueError(
            "Use a fixed subdivision for one evaluation. Run subdivision shift "
            "as a separate train/test protocol."
        )
    if not match_backtracks:
        raise ValueError("The tight benchmark requires match_backtracks=True.")

    derived_seed = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
    noise_candidates = tuple(
        sorted(
            {
                max(1, int(noise_pairs_per_g)),
                max(1, int(noise_pairs_per_g) * 2),
                max(1, int(noise_pairs_per_g) * 4),
            }
        )
    )
    return build_tight_fundamental_group_pair_copies(
        seed=derived_seed,
        number_of_permuted_copies=number_of_permuted_copies,
        low_rk=low_rk,
        high_rk=high_rk,
        noise_candidates=noise_candidates,
        subdivision=min_subdiv,
        max_attempts=max(5_000, max_tries * max_expansions * 5),
        return_pair=False,
        include_subcomplex_features=include_subcomplex_features,
        binary_marking=binary_marking,
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--subdivision", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=5_000)
    args = parser.parse_args()

    certificates = []
    for index in range(args.pairs):
        pair = generate_tight_pair(
            seed=args.seed + index,
            subdivision=args.subdivision,
            max_attempts=args.max_attempts,
        )
        certificates.append(asdict(pair.certificate))

    print(json.dumps(certificates, indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()
