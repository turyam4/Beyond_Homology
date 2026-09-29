from typing import Dict, List, Tuple
import torch
from torch_geometric.data import InMemoryDataset
from toponetx.classes import CombinatorialComplex as CCC
import numpy as np
from tqdm import tqdm

from data.complex_data import ComplexData
from fg_core import (
    betti_numbers,
    create_complex_edges_faces,
    degree_profile,
    generate_words,
    verify_class_A_reduces_to_commutator,
    verify_class_B_reduces_to_identity,
)


class FundamentalGroupDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        min_g: int = 1,
        max_g: int = 4,
        samples_per_g: int = 10,
        number_of_permuted_copies: int = 32,
        low_rk: int = 0,
        high_rk: int = 1,
        first_element_of_pair: bool = True,
        noise_pairs_per_g: int = 2,
        min_subdiv: int = 3,
        max_subdiv: int = 3,
        match_backtracks: bool = True,
        max_tries: int = 25,
        max_expansions: int = 6,
        seed: int = 0,
        binary_marking: bool = False,
    ):
        """
        Minimum-Leak Fundamental Group Benchmark (FINAL).

        min_subdiv / max_subdiv: each sample independently draws its
        generator-loop subdivision level p ~ Uniform[min_subdiv, max_subdiv]
        (both classes in a sample share the same p; p varies across
        samples). Leave both at 3 for fixed behavior. For a discretization-
        shift split, instantiate two datasets with disjoint ranges.

        match_backtracks (default True): closes the immediate-inverse-
        adjacency ("backtrack") gap between classes WITHOUT ever breaking
        word-length/degree/homology matching -- rejects-samples class B
        against class A's backtrack count, and if unsuccessful, widens the
        SHARED noise pool (both classes rebuilt from scratch together) up
        to max_expansions times. Measured: 100% exact backtrack match,
        exploit accuracy exactly 0.500 (chance), word length/degree/
        homology unaffected. IMPORTANT: this can inject shared noise even
        when noise_pairs_per_g=0 (measured: 200/200 samples at small g) --
        if you want the literal zero-noise, provably-1-WL-indistinguishable
        construction, set match_backtracks=False as well. See
        test_fundamental_group.py tests 10-12 for the numbers behind all of
        this, verified, not assumed.
        """
        self.min_g = min_g
        self.max_g = max_g
        self.samples_per_g = samples_per_g
        self.number_of_permuted_copies = number_of_permuted_copies
        self.first_element_of_pair = first_element_of_pair
        self.low_rk = low_rk
        self.high_rk = high_rk
        self.noise_pairs_per_g = noise_pairs_per_g
        self.min_subdiv = min_subdiv
        self.max_subdiv = max_subdiv
        self.match_backtracks = match_backtracks
        self.max_tries = max_tries
        self.max_expansions = max_expansions
        self.seed = seed
        self.binary_marking = binary_marking
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0], data_cls=ComplexData)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return (f"fundamental_group_dataset_seed{self.seed}"
                f"_subdiv{self.min_subdiv}-{self.max_subdiv}"
                f"_mb{int(self.match_backtracks)}"
                f"_bin{int(self.binary_marking)}.pt")

    def process(self):
        rng = np.random.default_rng(self.seed)
        data_list = []

        for g in tqdm(range(self.min_g, self.max_g + 1), desc="Generating Genus Pairs"):
            for _ in range(self.samples_per_g):
                W_A, W_B, p = generate_words(
                    g, self.noise_pairs_per_g, rng,
                    min_subdiv=self.min_subdiv, max_subdiv=self.max_subdiv,
                    match_backtracks=self.match_backtracks, max_tries=self.max_tries,
                    max_expansions=self.max_expansions)

                W_target = W_A if self.first_element_of_pair else W_B
                W_reliability = W_A

                cc_target = self._to_ccc(W_target)
                cc_reliability = self._to_ccc(W_reliability)

                data_target = ComplexData.create_from_cc(cc_target)
                data_reliability = ComplexData.create_from_cc(cc_reliability)

                target_copies = data_target.get_permuted_copies(num_copies=self.number_of_permuted_copies)
                reliability_copies = data_reliability.get_permuted_copies(num_copies=self.number_of_permuted_copies)

                for tgt, rel in zip(target_copies, reliability_copies):
                    tgt.compute_subcomplex_feature(
                        low_rk=self.low_rk,
                        high_rk=self.high_rk,
                        binary_marking=self.binary_marking,
                    )
                    rel.compute_subcomplex_feature(
                        low_rk=self.low_rk,
                        high_rk=self.high_rk,
                        binary_marking=self.binary_marking,
                    )
                    data_list.append(tgt)
                    data_list.append(rel)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])

    @staticmethod
    def _to_ccc(W_path: List[int]) -> CCC:
        edges, faces = create_complex_edges_faces(W_path)
        ccc = CCC()
        ccc.add_cells_from(edges, ranks=[1] * len(edges))
        ccc.add_cells_from(faces, ranks=[2] * len(faces))
        return ccc


def _build_base_complex_data(word_path: List[int]) -> ComplexData:
    """Build the single unpermuted ComplexData "shape" for a word.

    This is the expensive, non-O(n^2) part of the pipeline (CCC construction
    + incidence matrices) -- cheap to build once, wasteful to rebuild per
    copy-set when several copy-sets share the same underlying word.
    """
    cc_shape = FundamentalGroupDataset._to_ccc(word_path)
    return ComplexData.create_from_cc(cc_shape)


def _permuted_copies_with_subcomplex_features(
    data_shape: ComplexData,
    number_of_permuted_copies: int,
    low_rk: int,
    high_rk: int,
    include_subcomplex_features: bool = True,
    binary_marking: bool = False,
) -> List[ComplexData]:
    data_list = []
    for copy in data_shape.get_permuted_copies(num_copies=number_of_permuted_copies):
        if include_subcomplex_features:
            copy.compute_subcomplex_feature(
                low_rk=low_rk,
                high_rk=high_rk,
                binary_marking=binary_marking,
            )
        data_list.append(copy)
    return data_list


def build_fundamental_group_copies_from_word(
    word_path: List[int],
    number_of_permuted_copies: int,
    low_rk: int,
    high_rk: int,
    include_subcomplex_features: bool = True,
    binary_marking: bool = False,
) -> List[ComplexData]:
    """Build permuted ComplexData copies for exactly one FG word complex.

    The returned list has length `number_of_permuted_copies`; each item has
    subcomplex features for `(low_rk, high_rk)`. This is the same per-word
    construction used by FundamentalGroupDataset, but without PyG
    InMemoryDataset caching, so evaluators can stream one BREC trial at a time.
    """
    data_shape = _build_base_complex_data(word_path)
    return _permuted_copies_with_subcomplex_features(
        data_shape,
        number_of_permuted_copies,
        low_rk,
        high_rk,
        include_subcomplex_features=include_subcomplex_features,
        binary_marking=binary_marking,
    )


def build_fundamental_group_pair_copies(
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
    binary_marking: bool = False,
) -> Tuple[List[ComplexData], List[ComplexData], List[ComplexData], List[ComplexData], Dict]:
    """Create one streamed surface-vs-free FG BREC trial.

    Returns surface target copies, free target copies, and two independent
    surface reliability copy sets for exactly one generated sample. The
    function validates the construction's scientific invariants and raises on
    mismatch instead of emitting partial data.
    """
    word_surface, word_free, subdiv, slots = generate_words(
        g,
        noise_pairs_per_g,
        rng,
        min_subdiv=min_subdiv,
        max_subdiv=max_subdiv,
        match_backtracks=match_backtracks,
        max_tries=max_tries,
        max_expansions=max_expansions,
        return_slots=True,
    )
    surface_slots, free_slots = slots

    if not verify_class_A_reduces_to_commutator(surface_slots, g):
        raise RuntimeError(f"surface word failed commutator verification for g={g}")
    if not verify_class_B_reduces_to_identity(free_slots):
        raise RuntimeError(f"free word failed identity verification for g={g}")

    edges_surface, faces_surface = create_complex_edges_faces(word_surface)
    edges_free, faces_free = create_complex_edges_faces(word_free)
    vertex_count_surface = len({v for edge in edges_surface for v in edge})
    vertex_count_free = len({v for edge in edges_free for v in edge})
    cell_counts_surface = (vertex_count_surface, len(edges_surface), len(faces_surface))
    cell_counts_free = (vertex_count_free, len(edges_free), len(faces_free))
    if cell_counts_surface != cell_counts_free:
        raise RuntimeError(
            f"cell counts differ for g={g}: surface={cell_counts_surface}, free={cell_counts_free}"
        )

    expected_betti = (1, 2 * g, 1)
    betti_surface = betti_numbers(edges_surface, faces_surface)
    betti_free = betti_numbers(edges_free, faces_free)
    if betti_surface != expected_betti or betti_free != expected_betti:
        raise RuntimeError(
            f"unexpected Betti numbers for g={g}: surface={betti_surface}, free={betti_free}"
        )
    if degree_profile(edges_surface) != degree_profile(edges_free):
        raise RuntimeError(f"degree profiles differ for g={g}")

    metadata = {
        "g": int(g),
        "subdivision": int(subdiv),
        "num_cells_rank_0": int(cell_counts_surface[0]),
        "num_cells_rank_1": int(cell_counts_surface[1]),
        "num_cells_rank_2": int(cell_counts_surface[2]),
        "word_length": int(len(word_surface)),
    }

    # word_surface backs three copy-sets (target + two reliability sets) --
    # build its unpermuted base complex once and reuse it, instead of
    # rebuilding the CCC/incidence matrices from scratch three times.
    surface_base = _build_base_complex_data(word_surface)
    free_base = _build_base_complex_data(word_free)

    surface_target = _permuted_copies_with_subcomplex_features(
        surface_base, number_of_permuted_copies, low_rk, high_rk,
        binary_marking=binary_marking,
    )
    free_target = _permuted_copies_with_subcomplex_features(
        free_base, number_of_permuted_copies, low_rk, high_rk,
        binary_marking=binary_marking,
    )
    surface_rel_a = _permuted_copies_with_subcomplex_features(
        surface_base, number_of_permuted_copies, low_rk, high_rk,
        binary_marking=binary_marking,
    )
    surface_rel_b = _permuted_copies_with_subcomplex_features(
        surface_base, number_of_permuted_copies, low_rk, high_rk,
        binary_marking=binary_marking,
    )
    del surface_base, free_base
    return surface_target, free_target, surface_rel_a, surface_rel_b, metadata
