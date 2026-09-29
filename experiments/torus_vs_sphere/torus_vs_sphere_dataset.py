"""Dataset generation for the torus-vs-sphere control experiment
 Structural diversity via multiple torus grid sizes
    and multiple sphere subdivision levels.

    Total datapoints = (len(torus_grid_sizes) + len(sphere_subdiv_levels)) * number_of_permuted_copies
    Default: 4 torus grids + 4 sphere subdivisions = 8 unique complexes,
    * 32 permutation copies each = 256 samples, balanced (128 per class).
    
 Purpose: E0 positive control. Should hit ~100% for all models. Not a
benchmark -- a pipeline sanity check. Everything about the pair
(cell counts, chi, Betti, degree profile) is DELIBERATELY unmatched.     
"""
import sys
from pathlib import Path
from typing import List, Tuple

import torch
from torch_geometric.data import InMemoryDataset
from toponetx.classes import CombinatorialComplex as CCC
from tqdm import tqdm

PACKAGE_ROOT = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = EXPERIMENTS_ROOT.parent
for path in (PROJECT_ROOT, EXPERIMENTS_ROOT, PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data.complex_data import ComplexData  # noqa: E402
from orientability.orient_common import build_closed_grid  # noqa: E402


def icosahedron_cells():
    """Return the cells of a standard 12-vertex icosahedron."""
    faces = [(0, 1, 8), (0, 8, 4), (0, 4, 5), (0, 5, 9), (0, 9, 1),
            (1, 6, 8), (8, 6, 10), (8, 10, 4), (4, 10, 2), (4, 2, 5),
            (5, 2, 11), (5, 11, 9), (9, 11, 7), (9, 7, 1), (1, 7, 6),
            (3, 6, 7), (3, 10, 6), (3, 2, 10), (3, 11, 2), (3, 7, 11)]
    edges = set()
    for a, b, c in faces:
        for u, v in [(a, b), (b, c), (a, c)]:
            edges.add((min(u, v), max(u, v)))
    return list(range(12)), sorted(edges), [list(f) for f in faces]


def subdivide_triangular(vertices, edges, faces):
    """Split each triangular face into four triangles."""
    edge_mid = {}
    next_id = max(vertices) + 1
    for edge in edges:
        edge_mid[edge] = next_id
        next_id += 1
    new_vertices = list(vertices) + list(edge_mid.values())
    new_faces = []
    new_edges = set()
    for a, b, c in faces:
        eab = edge_mid[(min(a, b), max(a, b))]
        ebc = edge_mid[(min(b, c), max(b, c))]
        eac = edge_mid[(min(a, c), max(a, c))]
        for tri in [(a, eab, eac), (b, eab, ebc), (c, eac, ebc), (eab, ebc, eac)]:
            new_faces.append(list(tri))
            for u, v in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[0], tri[2])]:
                new_edges.add((min(u, v), max(u, v)))
    return new_vertices, sorted(new_edges), new_faces


def build_icosphere(subdivision_level: int = 1):
    vertices, edges, faces = icosahedron_cells()
    for _ in range(subdivision_level):
        vertices, edges, faces = subdivide_triangular(vertices, edges, faces)
    return edges, faces

class TorusVsSphereDataset(InMemoryDataset):
    """Generate permuted torus and sphere complexes for classification."""

    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        torus_grid_sizes: List[Tuple[int, int]] | None = None,
        sphere_subdiv_levels: List[int] | None = None,
        number_of_permuted_copies: int = 32,
        low_rk: int = 0,
        high_rk: int = 1,
        seed: int = 0,
    ):
        self.torus_grid_sizes = torus_grid_sizes or [(4, 4), (5, 5), (6, 6), (7, 7)]
        self.sphere_subdiv_levels = sphere_subdiv_levels or [0, 1]
        self.number_of_permuted_copies = number_of_permuted_copies
        self.low_rk = low_rk
        self.high_rk = high_rk
        self.seed = seed
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0], data_cls=ComplexData)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        n_t = len(self.torus_grid_sizes)
        n_s = len(self.sphere_subdiv_levels)
        return f"torus_vs_sphere_t{n_t}_s{n_s}_seed{self.seed}.pt"

    def process(self):
        data_list = self._create_data_list()
        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]
        self.save(data_list, self.processed_paths[0])

    def _create_data_list(self):
        data_list = []

        for h, p in tqdm(self.torus_grid_sizes, desc="Generating tori"):
            e, f = build_closed_grid(h, p, twisted=False)
            cc = _to_ccc(e, f)
            data = ComplexData.create_from_cc(cc)
            for copy in data.get_permuted_copies(num_copies=self.number_of_permuted_copies):
                copy.compute_subcomplex_feature(low_rk=self.low_rk, high_rk=self.high_rk)
                copy.y = torch.tensor([0], dtype=torch.long)
                data_list.append(copy)

        for subdiv in tqdm(self.sphere_subdiv_levels, desc="Generating spheres"):
            e, f = build_icosphere(subdiv)
            cc = _to_ccc(e, f)
            data = ComplexData.create_from_cc(cc)
            for copy in data.get_permuted_copies(num_copies=self.number_of_permuted_copies):
                copy.compute_subcomplex_feature(low_rk=self.low_rk, high_rk=self.high_rk)
                copy.y = torch.tensor([1], dtype=torch.long)
                data_list.append(copy)

        return data_list


def _to_ccc(edges: List[Tuple[int, int]], faces: List[List[int]]) -> CCC:
    ccc = CCC()
    ccc.add_cells_from([tuple(sorted(e)) for e in edges], ranks=1)
    ccc.add_cells_from([tuple(sorted(f)) for f in faces], ranks=2)
    return ccc
