"""
validate_control.py -- audit the pi_1-matched control families (E-G)
====================================================================
Independently re-verifies, per pair, that each control pair really does pass
the same gate stack as the cross-class family -- recomputing every invariant
from the complexes rather than trusting the emitted certificate.

Additionally measures the GLOBAL METRIC CHANNEL (C5) at two granularities,
because that is the open question in the metric-channel reviewer concern:

  coarse : node->node SPD histogram, eccentricity histogram, diameter, and the
           aggregated node->cell marking histogram. This is what DIST_MLP sees.
  fine   : the per-vertex marking profile
               m(v,e) = min_{u < e} d(v,u)   (as a sorted vector over cells)
           This is the object SMCN's shortest-path marking actually consumes,
           and is strictly finer than the aggregated histogram.

Reporting the fine-openness rate for the CONTROL families alongside the
cross-class family is what lets you say whether metric openness and pi_1
class vary independently.

Run:
    python validate_control.py --pairs 16
    python validate_control.py --pairs 16 --family surface_vs_surface
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from collections import Counter, defaultdict

warnings.filterwarnings("ignore")

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from control_family_dataset import (  # noqa: E402
    FAMILIES,
    FG_DIR,
    generate_control_pair,
)

if str(FG_DIR) not in sys.path:
    sys.path.insert(0, str(FG_DIR))

from fg_core import (  # noqa: E402
    create_complex_edges_faces,
    integral_homology,
    slot_letter_counts,
    slot_signed_exponent_sums,
    verify_class_A_reduces_to_commutator,
    verify_class_B_reduces_to_identity,
    word_backtracks,
)
from fundamental_group_dataset_genus2 import (  # noqa: E402
    local_incidence_profile,
    stable_rank_coloured_hasse_wl,
    vertex_link_wl_profile,
)


# --------------------------------------------------------------------------
# C5: global metric channel, at both granularities
# --------------------------------------------------------------------------
def metric_profiles(edges, faces):
    G = nx.Graph()
    G.add_edges_from(edges)
    sp = dict(nx.all_pairs_shortest_path_length(G))
    verts = sorted(G.nodes())

    nn_hist = Counter()
    ecc = {}
    for u in verts:
        d = sp[u]
        ecc[u] = max(d.values())
        for v in verts:
            nn_hist[d[v]] += 1

    # SMCN marking with low_rk=0, high_rk=1: cells are EDGES
    cells = sorted(tuple(sorted(e)) for e in edges)
    per_vertex = {}
    for v in verts:
        dv = sp[v]
        per_vertex[v] = tuple(min(dv[a], dv[b]) for (a, b) in cells)

    coarse_marking = Counter()
    for v in verts:
        coarse_marking.update(per_vertex[v])

    return {
        "nn_hist": tuple(sorted(nn_hist.items())),
        "ecc_hist": tuple(sorted(Counter(ecc.values()).items())),
        "diameter": max(ecc.values()),
        "coarse_marking": tuple(sorted(coarse_marking.items())),
        "fine_marking": tuple(sorted(Counter(
            tuple(sorted(per_vertex[v])) for v in verts).items())),
    }


# --------------------------------------------------------------------------
# independent gate recomputation
# --------------------------------------------------------------------------
def audit_pair(pair, family, genus=2):
    wa, wb = pair.surface_word_path, pair.free_word_path
    sa, sb = pair.surface_slots, pair.free_slots
    ea, fa = create_complex_edges_faces(wa)
    eb, fb = create_complex_edges_faces(wb)

    va = len({v for e in ea for v in e})
    vb = len({v for e in eb for v in e})
    counts_a, counts_b = (va, len(ea), len(fa)), (vb, len(eb), len(fb))

    if family == "surface_vs_surface":
        cls_a = verify_class_A_reduces_to_commutator(sa, genus)
        cls_b = verify_class_A_reduces_to_commutator(sb, genus)
    else:
        cls_a = verify_class_B_reduces_to_identity(sa)
        cls_b = verify_class_B_reduces_to_identity(sb)

    ha, hb = integral_homology(ea, fa), integral_homology(eb, fb)
    expected = (1, 2 * genus, 1)
    wl_eq, wl_rounds = stable_rank_coloured_hasse_wl((ea, fa), (eb, fb))

    checks = {
        "both_members_same_pi1_class": cls_a and cls_b,
        "cell_counts_equal": counts_a == counts_b,
        "euler_equal": (counts_a[0] - counts_a[1] + counts_a[2]
                        == counts_b[0] - counts_b[1] + counts_b[2]),
        "integral_betti_expected": (tuple(ha["betti"]) == expected
                                    and tuple(hb["betti"]) == expected),
        "torsion_free_both": ha["torsion_free"] and hb["torsion_free"],
        "torsion_equal": ha["torsion_H1"] == hb["torsion_H1"],
        "word_length_equal": len(wa) == len(wb),
        "slot_multiset_equal": Counter(sa) == Counter(sb),
        "letter_counts_equal": slot_letter_counts(sa) == slot_letter_counts(sb),
        "exponent_sums_equal": (slot_signed_exponent_sums(sa, genus)
                                == slot_signed_exponent_sums(sb, genus)),
        "node_token_multiset_equal": Counter(wa) == Counter(wb),
        "backtracks_equal": word_backtracks(wa) == word_backtracks(wb),
        "local_incidence_equal": (local_incidence_profile(ea, fa)
                                  == local_incidence_profile(eb, fb)),
        "vertex_link_wl_equal": (vertex_link_wl_profile(ea, fa)
                                 == vertex_link_wl_profile(eb, fb)),
        "hasse_wl_equivalent": wl_eq,
        "words_distinct": pair.certificate.words_distinct_up_to_symmetry,
    }

    ma, mb = metric_profiles(ea, fa), metric_profiles(eb, fb)
    metric = {
        "nn_hist_matched": ma["nn_hist"] == mb["nn_hist"],
        "ecc_hist_matched": ma["ecc_hist"] == mb["ecc_hist"],
        "diameter_matched": ma["diameter"] == mb["diameter"],
        "coarse_marking_matched": ma["coarse_marking"] == mb["coarse_marking"],
        "fine_marking_matched": ma["fine_marking"] == mb["fine_marking"],
    }
    return checks, metric, wl_rounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--family", choices=FAMILIES + ("both",), default="both")
    ap.add_argument("--subdivision", type=int, default=3)
    args = ap.parse_args()

    families = FAMILIES if args.family == "both" else (args.family,)
    overall_ok = True

    for fam in families:
        print(f"\n{'='*72}\n=== {fam} ({args.pairs} pairs) ===\n{'='*72}")
        gate_tally = defaultdict(int)
        metric_tally = defaultdict(int)
        attempts = []
        failures = []

        for i in range(args.pairs):
            pair = generate_control_pair(
                seed=args.seed + i, family=fam,
                subdivision=args.subdivision, max_attempts=5000,
            )
            checks, metric, wl_rounds = audit_pair(pair, fam)
            attempts.append(pair.certificate.attempts)
            for k, v in checks.items():
                gate_tally[k] += int(bool(v))
            for k, v in metric.items():
                metric_tally[k] += int(bool(v))
            bad = [k for k, v in checks.items() if not v]
            if bad:
                failures.append((args.seed + i, bad))
                overall_ok = False
            print(f"  seed={args.seed+i:>3} attempts={pair.certificate.attempts:>4} "
                  f"wl={wl_rounds} gates={'PASS' if not bad else 'FAIL'+str(bad)} "
                  f"fine_metric_matched={metric['fine_marking_matched']}")

        n = args.pairs
        print(f"\n  --- GATE STACK (must be {n}/{n}) ---")
        for k in sorted(gate_tally):
            flag = "" if gate_tally[k] == n else "   <-- FAILURE"
            print(f"    {k:34s} {gate_tally[k]:>3}/{n}{flag}")

        print(f"\n  --- C5 GLOBAL METRIC CHANNEL ---")
        for k in ["nn_hist_matched", "ecc_hist_matched", "diameter_matched",
                  "coarse_marking_matched", "fine_marking_matched"]:
            print(f"    {k:34s} {metric_tally[k]:>3}/{n}")
        open_fine = n - metric_tally["fine_marking_matched"]
        print(f"    -> fine metric channel OPEN in {open_fine}/{n} pairs")

        print(f"\n  attempts/pair: min={min(attempts)} "
              f"mean={sum(attempts)/len(attempts):.1f} max={max(attempts)}")
        if failures:
            print(f"  FAILURES: {failures}")

    print(f"\n{'='*72}")
    print("RESULT: all control pairs pass the full gate stack"
          if overall_ok else "RESULT: GATE FAILURES PRESENT -- do not use these pairs")
    print(f"{'='*72}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
