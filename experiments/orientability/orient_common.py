"""
orient_common.py
=================
Shared grid-based CW-complex builder for the three orientability benchmarks:
    Closed Torus vs Klein Bottle       (wrap both directions, twist vertical)
    Punctured Torus vs Punctured Klein (same builder + one removed face)
    Cylinder vs Mobius Strip           (wrap horizontal only, twist horizontal,
                                        open/bounded vertical direction)

This module exists specifically to FIX a real inconsistency in the uploaded
files: klien_vs_torus.py built its complex via a central-hub-and-collar
"presentation complex" (like the genus/fundamental-group benchmark), while
punc_torus_vs_klein.py built its complex via an h x p grid. Those are not the
same construction, so "closed Torus/Klein" and "punctured Torus/Klein" were
NOT actually the same manifold discretized two ways, as the pipeline's own
narrative claimed (this is exactly the inconsistency flagged in oreint.md's
"Do Research on following statement authenticity" section -- confirmed real,
see the audit script). Every dataset below is now built from ONE function.
"""
from typing import Any, List, Tuple, Optional, Dict, Set, Mapping
import numpy as np


# ============================================================================
# 1. grid builders
# ============================================================================

def build_closed_grid(h: int, p: int, twisted: bool,
                      punctured: bool = False,
                      vertical_shift: int = 0
                      ) -> Tuple[List[Tuple[int, int]], List[List[int]]]:
    """h x p grid, wrapped in BOTH directions. Horizontal wrap is always
    straight. Vertical wrap is twisted=False: (h-1,j) -> (0, (j+vertical_shift)%p)
    [orientable -- a SHIFT is not a reflection, so this is still a genuine
    torus for ANY integer vertical_shift, homeomorphic to the standard one --
    a shift is an orientation-preserving circle map, and the mapping torus
    of any orientation-preserving circle homeomorphism is homeomorphic to
    T^2]; twisted=True: (h-1,j) -> (0, p-1-j) [reflection, non-orientable,
    Klein bottle].

    vertical_shift is the mechanism for closing the girth/vertical-cycle-
    length leak: with vertical_shift=0 (default), the straight torus has
    vertical-only cycles of length h, while the Klein reflection has cycles
    of length 2h (or h at one fixed column if p is odd) -- a real,
    measured mismatch (see klein_torus_dataset.py). Setting
    vertical_shift=p//2 (REQUIRES p even -- shift must have order exactly 2
    in Z_p, which only exists for even p) makes the torus's vertical cycles
    ALSO uniformly length 2h, matching Klein exactly, while the space
    remains a genuine torus (verified: Betti numbers, chi, degree profile,
    edge-face incidence all identical to the unshifted torus -- only the
    girth/vertical-cycle-length changes). Verified computationally across
    p in {5,6,7,8}: only shift=p/2 achieves period exactly 2h; no shift
    achieves it for odd p (confirmed empirically, consistent with the
    group-theoretic fact that Z_p has no element of additive order 2 when
    p is odd).

    If punctured, the single face at the grid center is omitted, producing
    exactly one boundary loop of length 4 (a genuine, deliberately created
    boundary -- not a discretization artifact).
    Returns (edges, faces); faces are vertex CYCLES (ordered), not just sets,
    because computing a correct cellular boundary map needs the cyclic order.
    """
    if vertical_shift != 0:
        assert p % 2 == 0, ("vertical_shift=p//2 (girth-matching mode) requires "
                            "p even -- no shift achieves period 2h for odd p.")
        assert vertical_shift == p // 2, ("only vertical_shift=0 (standard) or "
                                          "vertical_shift=p//2 (girth-matched) "
                                          "are meaningful; other shifts give "
                                          "periods that match neither h nor 2h.")

    def nid(i, j):
        return i * p + j

    edges: Set[Tuple[int, int]] = set()
    faces: List[List[int]] = []

    def add_edge(a, b):
        edges.add((a, b) if a < b else (b, a))

    def vwrap_col(j):
        return (p - 1 - j) % p if twisted else (j + vertical_shift) % p

    for i in range(h):
        for j in range(p):
            add_edge(nid(i, j), nid(i, (j + 1) % p))               # horizontal
            if i < h - 1:
                add_edge(nid(i, j), nid(i + 1, j))                  # vertical (interior)
            else:
                add_edge(nid(i, j), nid(0, vwrap_col(j)))           # vertical (seam)

    puncture = (h // 2, p // 2) if punctured else None
    for i in range(h):
        for j in range(p):
            if puncture == (i, j):
                continue
            a = nid(i, j)
            b = nid(i, (j + 1) % p)
            if i < h - 1:
                c = nid(i + 1, (j + 1) % p)
                d = nid(i + 1, j)
            else:
                c = nid(0, vwrap_col((j + 1) % p))
                d = nid(0, vwrap_col(j))
            faces.append([a, b, c, d])

    return sorted(edges), faces


def build_bounded_grid(h: int, p: int, twisted: bool
                       ) -> Tuple[List[Tuple[int, int]], List[List[int]]]:
    """h x p grid, wrapped ONLY horizontally (belt direction); vertical
    direction is a genuine open boundary (rows 0 and h-1 have degree 3).
    Horizontal wrap is straight (Cylinder) or flips the row index
    (Mobius strip): (i, p-1) -> (h-1-i, 0)."""
    def nid(i, j):
        return i * p + j

    edges: Set[Tuple[int, int]] = set()
    faces: List[List[int]] = []

    def add_edge(a, b):
        edges.add((a, b) if a < b else (b, a))

    for i in range(h):
        for j in range(p):
            if j < p - 1:
                add_edge(nid(i, j), nid(i, j + 1))
            else:
                target = nid(h - 1 - i, 0) if twisted else nid(i, 0)
                add_edge(nid(i, j), target)
            if i < h - 1:
                add_edge(nid(i, j), nid(i + 1, j))

    for i in range(h - 1):
        for j in range(p):
            a, b = nid(i, j), nid(i + 1, j)
            if j < p - 1:
                c, d = nid(i + 1, j + 1), nid(i, j + 1)
            else:
                if twisted:
                    c = nid(h - 1 - (i + 1), 0)
                    d = nid(h - 1 - i, 0)
                else:
                    c = nid(i + 1, 0)
                    d = nid(i, 0)
            faces.append([a, b, c, d])

    return sorted(edges), faces


# ============================================================================
# 2. certification: degree, Z and Q homology, girth/cycle-length, boundary
# ============================================================================

def degree_profile(edges) -> Dict[int, int]:
    from collections import Counter
    c = Counter()
    for a, b in edges:
        c[a] += 1
        c[b] += 1
    return dict(Counter(c.values()))  # {degree: count_of_nodes}


def betti_numbers_cw(edges: List[Tuple[int, int]], faces: List[List[int]]
                     ) -> Tuple[int, int, int]:
    """Rational Betti numbers via cellular boundary maps d1: C1->C0,
    d2: C2->C1, using the CYCLIC ORDER of each face (proper signed cellular
    boundary, not a simplicial hack). This is the honest way to check whether
    plain rank-based Betti numbers already separate two classes -- do not
    assume; compute."""
    verts = sorted({v for e in edges for v in e})
    vidx = {v: i for i, v in enumerate(verts)}
    eidx = {e: i for i, e in enumerate(edges)}

    d1 = np.zeros((len(verts), len(edges)))
    for e, j in eidx.items():
        u, v = e
        d1[vidx[u], j] = -1.0
        d1[vidx[v], j] = 1.0

    d2 = np.zeros((len(edges), len(faces)))
    for j, face in enumerate(faces):
        n = len(face)
        for k in range(n):
            u, v = face[k], face[(k + 1) % n]
            e = (u, v) if u < v else (v, u)
            sgn = 1.0 if u < v else -1.0
            d2[eidx[e], j] += sgn

    r1 = np.linalg.matrix_rank(d1)
    r2 = np.linalg.matrix_rank(d2)
    b0 = len(verts) - r1
    b1 = len(edges) - r1 - r2
    b2 = len(faces) - r2
    return b0, b1, b2


def z2_betti_numbers(edges: List[Tuple[int, int]], faces: List[List[int]]
                     ) -> Tuple[int, int, int]:
    """Betti numbers over GF(2) via Gaussian elimination mod 2. Confirms the
    oreint.md claim that Torus and Klein look identical over Z2 -- checked
    directly rather than assumed."""
    def rank_mod2(mat: np.ndarray) -> int:
        M = mat.copy().astype(np.int64) % 2
        rows, cols = M.shape
        r = 0
        for c in range(cols):
            piv = None
            for i in range(r, rows):
                if M[i, c] % 2 == 1:
                    piv = i
                    break
            if piv is None:
                continue
            M[[r, piv]] = M[[piv, r]]
            for i in range(rows):
                if i != r and M[i, c] % 2 == 1:
                    M[i] = (M[i] + M[r]) % 2
            r += 1
            if r == rows:
                break
        return r

    verts = sorted({v for e in edges for v in e})
    vidx = {v: i for i, v in enumerate(verts)}
    eidx = {e: i for i, e in enumerate(edges)}
    d1 = np.zeros((len(verts), len(edges)), dtype=np.int64)
    for e, j in eidx.items():
        u, v = e
        d1[vidx[u], j] = 1
        d1[vidx[v], j] = 1
    d2 = np.zeros((len(edges), len(faces)), dtype=np.int64)
    for j, face in enumerate(faces):
        n = len(face)
        for k in range(n):
            u, v = face[k], face[(k + 1) % n]
            e = (u, v) if u < v else (v, u)
            d2[eidx[e], j] = (d2[eidx[e], j] + 1) % 2
    r1 = rank_mod2(d1)
    r2 = rank_mod2(d2)
    b0 = len(verts) - r1
    b1 = len(edges) - r1 - r2
    b2 = len(faces) - r2
    return b0, b1, b2


def vertical_walk_period(h: int, p: int, twisted: bool, j0: int = 0,
                         vertical_shift: int = 0) -> int:
    """Length of the closed walk obtained by repeatedly moving 'up' along
    vertical edges only, starting at column j0. h for standard Torus
    (vertical_shift=0), 2h for Klein (uniformly, if p is even; one fixed
    column at h if p is odd), and 2h for a SHIFTED torus with
    vertical_shift=p//2 (p must be even) -- this is the girth-matching mode,
    see build_closed_grid's docstring."""
    j = j0
    steps = 0
    while True:
        j = (p - 1 - j) % p if twisted else (j + vertical_shift) % p
        steps += h
        if j == j0:
            return steps
        if steps > 4 * h * p:  # safety
            return -1


def belt_loop_lengths(h: int, p: int, twisted: bool) -> List[int]:
    """For the BOUNDED grid: trace the boundary loop(s) (edges belonging to
    exactly one face) and return their lengths. Cylinder -> [p, p].
    Mobius -> [2p] (or [p] once, plus one [2p] loop, if h is even and the
    mirror pairing has no fixed row -- computed directly, not assumed)."""
    edges, faces = build_bounded_grid(h, p, twisted)
    from collections import Counter
    face_count = Counter()
    for face in faces:
        n = len(face)
        for k in range(n):
            u, v = face[k], face[(k + 1) % n]
            e = (u, v) if u < v else (v, u)
            face_count[e] += 1
    boundary_edges = {e for e, c in face_count.items() if c == 1}
    adj: Dict[int, List[int]] = {}
    for u, v in boundary_edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    seen_edges = set()
    loop_lengths = []
    for e in boundary_edges:
        if e in seen_edges:
            continue
        start, cur, prev = e[0], e[1], e[0]
        length = 1
        seen_edges.add(e)
        while cur != start:
            nxts = [n for n in adj[cur] if n != prev]
            nxt = nxts[0] if nxts else prev
            edge = (cur, nxt) if cur < nxt else (nxt, cur)
            seen_edges.add(edge)
            prev, cur = cur, nxt
            length += 1
        loop_lengths.append(length)
    return sorted(loop_lengths)


def wl_stalls_immediately(edges: List[Tuple[int, int]], max_rounds: int = 10
                          ) -> bool:
    """Untyped 1-WL on the plain vertex graph (uniform initial color, no edge
    typing). If every node has identical degree and identical local
    structure, 1-WL can stall at a SINGLE color class after round 1 and never
    refine further, regardless of global twists. Returns True if the
    partition never grows beyond 1 class within max_rounds (evidence that a
    PLAIN untyped message-passing GNN cannot see the twist at all, as
    distinct from a model with typed/ranked edges that could)."""
    adj: Dict[int, List[int]] = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    col = {n: 0 for n in adj}
    for _ in range(max_rounds):
        palette = {}
        new = {}
        for n in adj:
            sig = (col[n], tuple(sorted(col[m] for m in adj[n])))
            if sig not in palette:
                palette[sig] = len(palette)
            new[n] = palette[sig]
        if len(set(new.values())) == 1:
            col = new
            continue
        return False  # refined beyond 1 class => did NOT stall
    return len(set(col.values())) == 1


# ============================================================================
# additional certification utilities added on final review against oreint.md
# ============================================================================

def edge_face_incidence_profile(edges, faces) -> Dict[int, int]:
    """Number of faces each edge bounds -> count of edges with that number.
    Markdown claims 'every edge bounds exactly 2 faces' for closed grids --
    checked directly here rather than assumed."""
    from collections import Counter
    c = Counter()
    for face in faces:
        n = len(face)
        for k in range(n):
            u, v = face[k], face[(k + 1) % n]
            e = (u, v) if u < v else (v, u)
            c[e] += 1
    return dict(Counter(c.values()))


def trace_boundary_loops(edges, faces) -> List[int]:
    """General boundary-loop tracer: finds edges belonging to exactly one
    face and traces them into loops, returning sorted loop lengths. Works
    for bounded grids (Cylinder/Mobius) AND punctured closed grids (which
    have no boundary until punctured) -- one implementation, not two."""
    from collections import Counter
    face_count = Counter()
    for face in faces:
        n = len(face)
        for k in range(n):
            u, v = face[k], face[(k + 1) % n]
            e = (u, v) if u < v else (v, u)
            face_count[e] += 1
    boundary_edges = {e for e, c in face_count.items() if c == 1}
    adj: Dict[int, List[int]] = {}
    for u, v in boundary_edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    seen = set()
    loops = []
    for e in boundary_edges:
        if e in seen:
            continue
        start, cur, prev = e[0], e[1], e[0]
        length = 1
        seen.add(e)
        while cur != start:
            nxts = [n for n in adj[cur] if n != prev]
            nxt = nxts[0] if nxts else prev
            edge = (cur, nxt) if cur < nxt else (nxt, cur)
            seen.add(edge)
            prev, cur = cur, nxt
            length += 1
        loops.append(length)
    return sorted(loops)


def node_face_distance_histogram(
    edges: List[Tuple[int, int]],
    faces: List[List[int]],
    max_dist: int = 10,
) -> Tuple[int, ...]:
    """Exact raw-cell equivalent of ``dist_mlp``'s node-to-face histogram."""
    from collections import deque

    vertices = sorted({vertex for edge in edges for vertex in edge})
    adjacency = {vertex: [] for vertex in vertices}
    for left, right in edges:
        adjacency[left].append(right)
        adjacency[right].append(left)

    histogram = [0] * (max_dist + 2)
    for face in faces:
        distances = {vertex: 0 for vertex in face}
        queue = deque(distances)
        while queue:
            vertex = queue.popleft()
            next_distance = distances[vertex] + 1
            for neighbor in adjacency[vertex]:
                if neighbor not in distances:
                    distances[neighbor] = next_distance
                    queue.append(neighbor)
        for vertex in vertices:
            distance = distances.get(vertex, max_dist + 1)
            histogram[min(distance, max_dist + 1)] += 1
    return tuple(histogram)


def puncture_and_twist_are_separated(h: int, p: int) -> bool:
    """Checks the markdown's specific claim: at the puncture location
    (h//2, p//2), do the 4 corner nodes of the punctured face overlap with
    the twist-seam rows (0 and h-1)? If they do, the puncture and twist
    motifs are adjacent and a local message-passing node could see both at
    once -- exactly the leak the markdown says min_dim=5 avoids. Verified
    directly: True (separated) at h=p=5, False (overlap) at h=p=3 or 4."""
    def nid(i, j):
        return i * p + j
    pi, pj = h // 2, p // 2
    pf_corners = {nid(pi, pj), nid(pi, (pj + 1) % p),
                 nid((pi + 1) % h, pj), nid((pi + 1) % h, (pj + 1) % p)}
    twist_rows = {nid(h - 1, j) for j in range(p)} | {nid(0, j) for j in range(p)}
    return len(pf_corners & twist_rows) == 0


def wl_stabilized_check(edges_a, edges_b, max_rounds: int = 60):
    """Rigorous version of the WL check: distinguishes 'genuinely
    stabilized and still identical' (a real, checked indistinguishability
    result, positively confirming the Isomorphism Check up to any hop
    count) from 'not separated within budget but still refining' (which
    would be inconclusive, NOT a positive result). Returns
    (message, True/False/None)."""
    from collections import Counter
    adjA, adjB = {}, {}
    for u, v in edges_a:
        adjA.setdefault(u, []).append(v)
        adjA.setdefault(v, []).append(u)
    for u, v in edges_b:
        adjB.setdefault(u, []).append(v)
        adjB.setdefault(v, []).append(u)
    colA = {n: 0 for n in adjA}
    colB = {n: 0 for n in adjB}

    def hist(col):
        return tuple(sorted(Counter(col.values()).values()))

    prev_hA, prev_hB = None, None
    for rnd in range(1, max_rounds + 1):
        palette: Dict = {}

        def refine(adj, col):
            new = {}
            for n in adj:
                sig = (col[n], tuple(sorted(col[m] for m in adj[n])))
                if sig not in palette:
                    palette[sig] = len(palette)
                new[n] = palette[sig]
            return new

        colA, colB = refine(adjA, colA), refine(adjB, colB)
        hA, hB = hist(colA), hist(colB)
        if hA != hB:
            return f"separated at round {rnd}", False
        if hA == prev_hA and hB == prev_hB:
            return f"stabilized at round {rnd}, still identical", True
        prev_hA, prev_hB = hA, hB
    return f"not stabilized within {max_rounds} rounds (inconclusive)", None


# ============================================================================
# 3. JSONL sidecar certificates for evaluator runs
# ============================================================================

def _sorted_profile(value) -> Dict[str, int]:
    return {str(key): int(value[key]) for key in sorted(value)}


def _rank_counts(edges, faces) -> List[int]:
    vertices = {v for edge in edges for v in edge}
    return [len(vertices), len(edges), len(faces)]


def _euler(rank_counts: List[int]) -> int:
    return int(rank_counts[0] - rank_counts[1] + rank_counts[2])


def _hasse_graph(edges, faces):
    adjacency: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    colors: Dict[Tuple[int, int], int] = {}
    vertices = sorted({v for edge in edges for v in edge})
    edge_index = {edge: idx for idx, edge in enumerate(edges)}

    def add_node(node, color):
        adjacency.setdefault(node, [])
        colors[node] = color

    def add_link(left, right):
        adjacency.setdefault(left, []).append(right)
        adjacency.setdefault(right, []).append(left)

    for vertex in vertices:
        add_node((0, vertex), 0)
    for idx, (u, v) in enumerate(edges):
        edge_node = (1, idx)
        add_node(edge_node, 1)
        add_link((0, u), edge_node)
        add_link((0, v), edge_node)
    for face_idx, face in enumerate(faces):
        face_node = (2, face_idx)
        add_node(face_node, 2)
        n = len(face)
        for k in range(n):
            u, v = face[k], face[(k + 1) % n]
            edge = (u, v) if u < v else (v, u)
            add_link((1, edge_index[edge]), face_node)
    return adjacency, colors


def _colored_wl_stabilized_check(left, right, max_rounds: int = 60):
    from collections import Counter

    adj_a, col_a = left
    adj_b, col_b = right

    def hist(colors):
        return tuple(sorted(Counter(colors.values()).values()))

    prev_a, prev_b = None, None
    for rnd in range(1, max_rounds + 1):
        palette = {}

        def refine(adjacency, colors):
            new = {}
            for node in adjacency:
                sig = (
                    colors[node],
                    tuple(sorted(colors[neighbor] for neighbor in adjacency[node])),
                )
                if sig not in palette:
                    palette[sig] = len(palette)
                new[node] = palette[sig]
            return new

        col_a, col_b = refine(adj_a, col_a), refine(adj_b, col_b)
        hist_a, hist_b = hist(col_a), hist(col_b)
        if hist_a != hist_b:
            return f"separated at round {rnd}", False, rnd
        if hist_a == prev_a and hist_b == prev_b:
            return f"stabilized at round {rnd}, still identical", True, rnd
        prev_a, prev_b = hist_a, hist_b
    return f"not stabilized within {max_rounds} rounds (inconclusive)", None, max_rounds


def _get(mapping: Mapping[str, Any], key: str, default: Any = None) -> Any:
    return mapping.get(key, default) if isinstance(mapping, Mapping) else default


def _family_cellulations(family: str, dataset: Mapping[str, Any], h: int, p: int):
    girth_matched = bool(_get(dataset, "girth_matched", False))
    vertical_shift = (p // 2) if girth_matched else 0
    if family == "cylinder_vs_mobius":
        return (
            build_bounded_grid(h, p, twisted=False),
            build_bounded_grid(h, p, twisted=True),
        )
    if family == "torus_vs_klein":
        return (
            build_closed_grid(h, p, twisted=False, vertical_shift=vertical_shift),
            build_closed_grid(h, p, twisted=True),
        )
    if family == "punc_torus_vs_punc_klein":
        return (
            build_closed_grid(
                h, p, twisted=False, punctured=True, vertical_shift=vertical_shift
            ),
            build_closed_grid(h, p, twisted=True, punctured=True),
        )
    raise ValueError(f"No orientability certificate builder for family={family!r}")


def build_orientability_certificate(
    *,
    family: str,
    family_label: str,
    dataset: Mapping[str, Any],
    pair_index: int,
    sequence_index: int,
    h: int,
    p: int,
) -> Dict[str, Any]:
    """Build the reviewer-facing certificate written beside every run.

    This mirrors the fundamental-group sidecar pattern: heavy training metrics
    stay in pair_results.csv/metrics.json, while structural audit facts live
    in certificates.jsonl, one JSON object per benchmark pair.
    """
    (edges_a, faces_a), (edges_b, faces_b) = _family_cellulations(family, dataset, h, p)
    ranks_a, ranks_b = _rank_counts(edges_a, faces_a), _rank_counts(edges_b, faces_b)
    metric_matched = bool(_get(dataset, "metric_matched", False))
    if family == "punc_torus_vs_punc_klein" and metric_matched:
        # Both punctured surfaces deformation retract to a rank-two graph.
        # Avoid repeating dense boundary-matrix reductions for every run/seed.
        betti_a = betti_b = (1, 2, 0)
        z2_a = z2_b = (1, 2, 0)
    else:
        betti_a = betti_numbers_cw(edges_a, faces_a)
        betti_b = betti_numbers_cw(edges_b, faces_b)
        z2_a = z2_betti_numbers(edges_a, faces_a)
        z2_b = z2_betti_numbers(edges_b, faces_b)
    boundary_a = trace_boundary_loops(edges_a, faces_a)
    boundary_b = trace_boundary_loops(edges_b, faces_b)
    distance_max = int(_get(dataset, "metric_match_max_dist", 10))
    distance_a = node_face_distance_histogram(edges_a, faces_a, distance_max)
    distance_b = node_face_distance_histogram(edges_b, faces_b, distance_max)
    degree_a, degree_b = degree_profile(edges_a), degree_profile(edges_b)
    incidence_a = edge_face_incidence_profile(edges_a, faces_a)
    incidence_b = edge_face_incidence_profile(edges_b, faces_b)
    vertex_wl_msg, vertex_wl_equal = wl_stabilized_check(edges_a, edges_b)
    hasse_msg, hasse_equal, hasse_rounds = _colored_wl_stabilized_check(
        _hasse_graph(edges_a, faces_a),
        _hasse_graph(edges_b, faces_b),
    )

    certificate: Dict[str, Any] = {
        "pair_index": int(pair_index),
        "sequence_index": int(sequence_index),
        "family": str(family),
        "benchmark": str(family_label),
        "h": int(h),
        "p": int(p),
        "first_class": str(_get(dataset, "first_class", "first")),
        "second_class": str(_get(dataset, "second_class", "second")),
        "reliability_class": str(_get(dataset, "reliability_class", "first")),
        "rank_counts_first": ranks_a,
        "rank_counts_second": ranks_b,
        "rank_counts_equal": ranks_a == ranks_b,
        "euler_characteristic_first": _euler(ranks_a),
        "euler_characteristic_second": _euler(ranks_b),
        "euler_characteristic_equal": _euler(ranks_a) == _euler(ranks_b),
        "rational_betti_first": list(map(int, betti_a)),
        "rational_betti_second": list(map(int, betti_b)),
        "rational_betti_equal": betti_a == betti_b,
        "z2_betti_first": list(map(int, z2_a)),
        "z2_betti_second": list(map(int, z2_b)),
        "z2_betti_equal": z2_a == z2_b,
        "degree_profile_first": _sorted_profile(degree_a),
        "degree_profile_second": _sorted_profile(degree_b),
        "degree_profile_equal": degree_a == degree_b,
        "edge_face_incidence_profile_first": _sorted_profile(incidence_a),
        "edge_face_incidence_profile_second": _sorted_profile(incidence_b),
        "edge_face_incidence_profiles_equal": incidence_a == incidence_b,
        "boundary_loop_lengths_first": list(map(int, boundary_a)),
        "boundary_loop_lengths_second": list(map(int, boundary_b)),
        "boundary_loop_lengths_equal": boundary_a == boundary_b,
        "node_face_distance_histogram_first": list(map(int, distance_a)),
        "node_face_distance_histogram_second": list(map(int, distance_b)),
        "node_face_distance_histograms_equal": distance_a == distance_b,
        "plain_vertex_1wl_equivalent": vertex_wl_equal,
        "plain_vertex_1wl_message": vertex_wl_msg,
        "hasse_wl_equivalent": hasse_equal,
        "hasse_wl_message": hasse_msg,
        "hasse_wl_rounds_to_stability": hasse_rounds if hasse_equal else None,
        "girth_matched": _get(dataset, "girth_matched", None),
        "metric_matched": metric_matched,
        "binary_marking": _get(dataset, "binary_marking", None),
    }

    if family == "cylinder_vs_mobius":
        certificate.update(
            {
                "expert_signal": "orientability_with_boundary",
                "cylinder_boundary_loops_expected": [p, p],
                "mobius_boundary_loops_expected": [2 * p],
            }
        )
    elif family == "torus_vs_klein":
        vertical_shift = (p // 2) if bool(_get(dataset, "girth_matched", False)) else 0
        certificate.update(
            {
                "expert_signal": "closed_orientability",
                "vertical_period_first": vertical_walk_period(
                    h, p, twisted=False, vertical_shift=vertical_shift
                ),
                "vertical_period_second": vertical_walk_period(h, p, twisted=True),
            }
        )
    elif family == "punc_torus_vs_punc_klein":
        vertical_shift = (p // 2) if bool(_get(dataset, "girth_matched", False)) else 0
        cycle_periods_a = [
            vertical_walk_period(h, p, False, column, vertical_shift)
            for column in range(p)
        ]
        cycle_periods_b = [
            vertical_walk_period(h, p, True, column) for column in range(p)
        ]
        certificate.update(
            {
                "expert_signal": "punctured_orientability",
                "puncture_and_twist_separated": puncture_and_twist_are_separated(h, p),
                "vertical_period_first": vertical_walk_period(
                    h, p, twisted=False, vertical_shift=vertical_shift
                ),
                "vertical_period_second": vertical_walk_period(h, p, twisted=True),
                "vertical_cycle_periods_first": cycle_periods_a,
                "vertical_cycle_periods_second": cycle_periods_b,
                "vertical_cycle_periods_equal": cycle_periods_a == cycle_periods_b,
            }
        )
    return certificate
