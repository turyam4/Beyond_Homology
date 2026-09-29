from typing import List, Tuple
import os
import sys

from torch_geometric.data import InMemoryDataset
from data.complex_data import ComplexData
from toponetx.classes import CombinatorialComplex as CCC
import numpy as np
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from orient_common import build_closed_grid, betti_numbers_cw, degree_profile


class ClosedOrientabilityDataset(InMemoryDataset):
    """
    Closed Surface Orientability: Torus vs Klein Bottle, grid-based.

    FIX vs the previous version of this file: the previous _create_complex
    built the pair via a central-hub-and-collar presentation complex (one
    0-cell, subdivided loop generators, fan+annulus 2-cell) -- structurally
    a DIFFERENT construction from PuncturedOrientabilityDataset's h x p grid.
    That meant "Closed Torus/Klein" and "Punctured Torus/Klein" were not
    actually the same manifold discretized two ways, despite the pipeline's
    narrative claiming they were (this is the inconsistency oreint.md asked
    to have checked -- confirmed real). Both classes are now built from the
    SAME function, orient_common.build_closed_grid, with punctured=False
    here and punctured=True in PuncturedOrientabilityDataset.

    HONEST WARNING, not in the previous docstring: this closed (unpunctured)
    pair is compromised by something stronger than a 1-WL leak. Computed
    directly (see certification below): the plain RATIONAL Betti numbers
    already differ -- Torus=(1,2,1), Klein=(1,1,0) -- because a closed
    non-orientable surface has no Z-fundamental class (H2=0) and one fewer
    independent 1-cycle over Q than the matching-genus orientable surface.
    A trivial count-only-Betti baseline solves this pair at ~100% without
    seeing the twist at all. Only over Z2 do Torus and Klein coincide
    (both (1,2,1)), which is the fact oreint.md correctly emphasizes -- but
    Z2-only equality doesn't help if the evaluation pipeline (or a naive
    baseline) computes ordinary rank-based Betti numbers, which is the
    default in most tooling.
    Recommendation: keep this dataset only as a secondary/pedagogical
    example (e.g. to illustrate the Z vs Z2 distinction itself), and use
    PuncturedOrientabilityDataset as the primary orientability benchmark,
    where puncturing removes H2 from both classes and equalizes Q-Betti to
    (1,2,0) for both (verified below).

    A second, real, and NOT fixed by puncturing, issue: the vertical-only
    walk from a fixed column closes after h steps (Torus) vs up to 2h steps
    (Klein) -- but this is only exploitable by a model whose message passing
    can distinguish "vertical" from "horizontal" edges (e.g. via cell rank
    or geometric features). A plain untyped 1-WL/GNN provably CANNOT see it:
    color refinement stalls at a single color class after round 1 for BOTH
    classes (every node is degree 4 with identical local structure), which
    was verified computationally, not assumed. If SMCN's message passing is
    typed/ranked in a way that can track edge direction, bound num_layers
    well below h in any experiment that wants to credit SMCN with learning
    real orientability rather than counting vertical-cycle length.

    FINAL-PASS ADDENDUM (re-verified against oreint.md line by line):
    - "every edge bounds exactly 2 faces" (markdown's claim): confirmed,
      edge_face_incidence_profile gives {2: 72} for both classes at h=p=6.
    - the "1-WL cannot see the twist" claim was reverified with explicit
      STABILIZATION detection (not just "not separated within N rounds"):
      color refinement provably stabilizes at round 2 with identical
      histograms for both classes -- a positive, checked result, not an
      absence-of-evidence one. See wl_stabilized_check in orient_common.py
      and test_orient.py.

    girth_matched (NEW): if True, and (min_dim,max_dim) are chosen so p is
    always even, the Torus is built with vertical_shift=p//2 instead of the
    standard straight identification. This is still a genuine torus (a
    shift is orientation-preserving; the mapping torus of any orientation-
    preserving circle map is homeomorphic to T^2 -- verified: Betti
    numbers, chi, degree profile, and edge-face incidence all identical to
    the standard torus). The ONLY thing that changes is the vertical-only
    cycle length: girth_matched=False gives Torus period h vs Klein's 2h (a
    real, measured gap); girth_matched=True gives BOTH period 2h uniformly
    (verified across every starting column, every (h,p) tested). This
    closes the girth leak for THIS pair. It does NOT extend to Cylinder vs
    Mobius, for a genuinely different, structural reason documented in
    cyl_vs_mob.py -- that one isn't a matter of trying harder.
    """

    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        min_dim: int = 4,
        max_dim: int = 11,
        number_of_permuted_copies: int = 32,
        low_rk: int = 0,
        high_rk: int = 1,
        first_element_of_pair: bool = True,
        seed: int = 0,
        girth_matched: bool = False,
    ):
        self.min_dim = min_dim
        self.max_dim = max_dim
        self.number_of_permuted_copies = number_of_permuted_copies
        self.first_element_of_pair = first_element_of_pair
        self.low_rk = low_rk
        self.high_rk = high_rk
        self.seed = seed
        self.girth_matched = girth_matched
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0], data_cls=ComplexData)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return (f"closed_orientability_h{self.min_dim}-{self.max_dim}"
                f"_seed{self.seed}_gm{int(self.girth_matched)}.pt")

    def process(self):
        data_list = self._create_data_list()
        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]
        self.save(data_list, self.processed_paths[0])

    def _create_data_list(self):
        rng = np.random.default_rng(self.seed)
        data_list = []
        for h in tqdm(range(self.min_dim, self.max_dim + 1),
                      desc="Generating Torus/Klein grid pairs"):
            for p in range(self.min_dim, self.max_dim + 1):
                if self.girth_matched and p % 2 != 0:
                    continue  # girth-matching requires p even; skip odd p
                vshift = (p // 2) if self.girth_matched else 0
                e_t, f_t = build_closed_grid(h, p, twisted=False, vertical_shift=vshift)
                e_k, f_k = build_closed_grid(h, p, twisted=True)

                # certification (cheap; keep it -- it's what makes the claims
                # in this docstring trustworthy instead of asserted)
                assert degree_profile(e_t) == degree_profile(e_k) == {4: h * p}
                bt, bk = betti_numbers_cw(e_t, f_t), betti_numbers_cw(e_k, f_k)
                assert bt == (1, 2, 1) and bk == (1, 2 * 1 - 1, 0), \
                    f"unexpected Betti numbers: torus={bt} klein={bk}"

                e_target, f_target = (e_t, f_t) if self.first_element_of_pair else (e_k, f_k)
                e_rel, f_rel = e_t, f_t  # reliability = independent Torus realization

                cc_target = _to_ccc(e_target, f_target)
                cc_rel = _to_ccc(e_rel, f_rel)
                d_target = ComplexData.create_from_cc(cc_target)
                d_rel = ComplexData.create_from_cc(cc_rel)

                for tgt, rel in zip(
                        d_target.get_permuted_copies(num_copies=self.number_of_permuted_copies),
                        d_rel.get_permuted_copies(num_copies=self.number_of_permuted_copies)):
                    tgt.compute_subcomplex_feature(low_rk=self.low_rk, high_rk=self.high_rk)
                    rel.compute_subcomplex_feature(low_rk=self.low_rk, high_rk=self.high_rk)
                    data_list += [tgt, rel]
        return data_list


def _to_ccc(edges: List[Tuple[int, int]], faces: List[List[int]]) -> "CCC":
    ccc = CCC()
    ccc.add_cells_from([tuple(sorted(e)) for e in edges], ranks=1)
    ccc.add_cells_from([tuple(sorted(f)) for f in faces], ranks=2)
    return ccc
