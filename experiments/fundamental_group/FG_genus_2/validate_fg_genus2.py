"""Full validation of the (fixed) tight genus-2 generator.

Re-runs the leakage audit on the SHIPPED artifact across N seeds -- the
original delivered file did not import (a stray token), so its "16 seeds
passed" claim could not have been produced by that file. This reproduces it on
the corrected code, and ADDS an adversarial probe strictly stronger than the
generator's own vertex-link filter: exact vertex links as MULTIGRAPHS (parallel
edges kept), which is the theorem-motivated attack (a closed surface has circle
links everywhere; the free class cannot). Passing means even that probe fails
to separate certified pairs.

Run: python validate_tight.py [num_seeds]
"""
from __future__ import annotations

import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")

SELF = Path(__file__).resolve().parent
sys.path.insert(0, str(SELF))

import networkx as nx  # noqa: E402
from fundamental_group_dataset_genus2 import generate_tight_pair  # noqa: E402
from fg_core import integral_homology  # noqa: E402


def multigraph_link_profile(edges, faces):
    """Exact vertex links as weighted graphs (weight = parallel multiplicity),
    WL-hashed with edge attributes. Strictly stronger than the simple-graph
    link filter inside the generator."""
    inc = defaultdict(list)
    for f in faces:
        for v in f:
            inc[v].append(f)
    keys = []
    for v, fs in inc.items():
        w = Counter()
        for f in fs:
            a, b = sorted(x for x in f if x != v)
            w[(a, b)] += 1
        G = nx.Graph()
        for (a, b), m in w.items():
            G.add_edge(a, b, w=str(m))
        h = nx.weisfeiler_lehman_graph_hash(G, edge_attr="w", iterations=10)
        keys.append((G.number_of_nodes(), sum(w.values()), h))
    return Counter(keys)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    fails = []
    attempts_seen = []
    for seed in range(n):
        pair = generate_tight_pair(seed=seed, max_attempts=5000)
        c = pair.certificate
        attempts_seen.append(c.attempts)

        checks = {
            "euler=-2": c.euler_characteristic == -2,
            "float_betti=(1,4,1)": tuple(c.betti_numbers) == (1, 4, 1),
            "integral_betti=(1,4,1)": tuple(c.integral_betti) == (1, 4, 1),
            "torsion_free": c.integral_homology_torsion_free and tuple(c.torsion_H1) == (),
            "hasse_wl_equivalent": c.hasse_wl_equivalent,
            "local_incidence_equal": c.local_incidence_profiles_equal,
            "vertex_link_wl_equal": c.vertex_link_wl_profiles_equal,
            "boundary_edges=0": c.boundary_edges == 0,
        }
        # recompute integral homology independently of the cached certificate
        hs = integral_homology(pair.surface_edges, pair.surface_faces)
        hf = integral_homology(pair.free_edges, pair.free_faces)
        checks["indep_homology_match_torsionfree"] = (
            hs["betti"] == hf["betti"] == (1, 4, 1)
            and hs["torsion_free"] and hf["torsion_free"]
        )
        # ADVERSARIAL: exact multigraph vertex-link profiles must NOT separate
        pa = multigraph_link_profile(pair.surface_edges, pair.surface_faces)
        pb = multigraph_link_profile(pair.free_edges, pair.free_faces)
        checks["multigraph_link_indistinguishable"] = (pa == pb)

        bad = [k for k, ok in checks.items() if not ok]
        status = "PASS" if not bad else f"FAIL({bad})"
        print(f"seed={seed:>2} attempts={c.attempts:>4} wl_rounds={c.hasse_wl_rounds_to_stability} -> {status}")
        if bad:
            fails.append((seed, bad))

    print(f"\nattempts/pair: min={min(attempts_seen)} mean={sum(attempts_seen)/len(attempts_seen):.1f} max={max(attempts_seen)}")
    if fails:
        print(f"RESULT: {len(fails)}/{n} seeds FAILED: {fails}")
        sys.exit(1)
    print(f"RESULT: all {n} seeds passed every invariant AND resisted the "
          f"exact-multigraph-link adversarial probe.")


if __name__ == "__main__":
    main()
