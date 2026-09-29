"""
validate_relator_pairs.py -- PHASE 3: independent audit of relator-family pairs
==============================================================================
Recomputes every invariant from the complexes rather than trusting the emitted
certificate, and additionally reports:

  * the Hom-counting NON-ISOMORPHISM certificate for the pairing's two groups
    (proof that the classes really are different groups -- homology cannot
    show this, since both families lie in [F,F]);
  * the C5 metric channel at coarse and fine granularity;
  * the C6 spectral channel.

The last two are reported because the surface-vs-free family leaks on both,
and we should know up front whether a new relator family inherits the same
leak rather than discovering it after running models.

Run:  python validate_relator_pairs.py --pairing deep_vs_free --pairs 16
"""
from __future__ import annotations

import argparse
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from relator_pair_dataset import (  # noqa: E402
    PAIRINGS, FG_ROOT, FG_G2, generate_relator_pair, verify_reduces_to,
)
from relator_library import RELATORS  # noqa: E402
from hom_certificate import certify as hom_certify  # noqa: E402

for _p in (str(FG_ROOT), str(FG_G2)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fg_core import (  # noqa: E402
    create_complex_edges_faces, integral_homology, slot_letter_counts,
    slot_signed_exponent_sums, word_backtracks,
)
from fundamental_group_dataset_genus2 import (  # noqa: E402
    local_incidence_profile, stable_rank_coloured_hasse_wl, vertex_link_wl_profile,
)

TOL = 1e-8


# ---------------------------------------------------------------- channels
def metric_profiles(edges, faces):
    G = nx.Graph(); G.add_edges_from(edges)
    sp = dict(nx.all_pairs_shortest_path_length(G))
    verts = sorted(G.nodes())
    nn = Counter(); ecc = {}
    for u in verts:
        d = sp[u]; ecc[u] = max(d.values())
        for v in verts:
            nn[d[v]] += 1
    cells = sorted(tuple(sorted(e)) for e in edges)
    per_v = {v: tuple(min(sp[v][a], sp[v][b]) for (a, b) in cells) for v in verts}
    coarse = Counter()
    for v in verts:
        coarse.update(per_v[v])
    return {"nn": tuple(sorted(nn.items())),
            "ecc": tuple(sorted(Counter(ecc.values()).items())),
            "coarse": tuple(sorted(coarse.items())),
            "fine": tuple(sorted(Counter(tuple(sorted(per_v[v])) for v in verts).items()))}


def spectra(edges, faces):
    G = nx.Graph(); G.add_edges_from(edges)
    nodes = sorted(G.nodes())
    A = nx.to_numpy_array(G, nodelist=nodes)
    d = A.sum(1)
    out = {"adj": np.sort(np.linalg.eigvalsh(A)),
           "lap": np.sort(np.linalg.eigvalsh(np.diag(d) - A))}
    dinv = np.divide(1.0, np.sqrt(d), out=np.zeros_like(d), where=d > 0)
    out["nlap"] = np.sort(np.linalg.eigvalsh(
        np.eye(len(nodes)) - (dinv[:, None] * A * dinv[None, :])))
    return out


# ---------------------------------------------------------------- audit
def audit(pair, g, rel_a, rel_b):
    wa, wb = pair.surface_word_path, pair.free_word_path
    sa, sb = pair.surface_slots, pair.free_slots
    ea, fa = create_complex_edges_faces(wa)
    eb, fb = create_complex_edges_faces(wb)
    va = len({v for e in ea for v in e}); vb = len({v for e in eb for v in e})
    ca, cb = (va, len(ea), len(fa)), (vb, len(eb), len(fb))
    ha, hb = integral_homology(ea, fa), integral_homology(eb, fb)
    exp = (1, 2 * g, 1)
    wl_eq, wl_r = stable_rank_coloured_hasse_wl((ea, fa), (eb, fb))

    checks = {
        "A_reduces_to_relator_A": verify_reduces_to(sa, RELATORS[rel_a](g)),
        "B_reduces_to_relator_B": verify_reduces_to(sb, RELATORS[rel_b](g)),
        "cell_counts_equal": ca == cb,
        "euler_equal": ca[0] - ca[1] + ca[2] == cb[0] - cb[1] + cb[2],
        "integral_betti_expected": tuple(ha["betti"]) == exp and tuple(hb["betti"]) == exp,
        "torsion_free_both": ha["torsion_free"] and hb["torsion_free"],
        "torsion_equal": ha["torsion_H1"] == hb["torsion_H1"],
        "word_length_equal": len(wa) == len(wb),
        "slot_multiset_equal": Counter(sa) == Counter(sb),
        "letter_counts_equal": slot_letter_counts(sa) == slot_letter_counts(sb),
        "exponent_sums_equal": slot_signed_exponent_sums(sa, g) == slot_signed_exponent_sums(sb, g),
        "node_token_multiset_equal": Counter(wa) == Counter(wb),
        "backtracks_equal": word_backtracks(wa) == word_backtracks(wb),
        "local_incidence_equal": local_incidence_profile(ea, fa) == local_incidence_profile(eb, fb),
        "vertex_link_wl_equal": vertex_link_wl_profile(ea, fa) == vertex_link_wl_profile(eb, fb),
        "hasse_wl_equivalent": wl_eq,
    }
    ma, mb = metric_profiles(ea, fa), metric_profiles(eb, fb)
    pa, pb = spectra(ea, fa), spectra(eb, fb)
    chan = {
        "nn_spd_matched": ma["nn"] == mb["nn"],
        "coarse_metric_matched": ma["coarse"] == mb["coarse"],
        "fine_metric_matched": ma["fine"] == mb["fine"],
        "spectral_matched": all(
            pa[k].shape == pb[k].shape and np.allclose(pa[k], pb[k], atol=TOL, rtol=0)
            for k in pa),
    }
    return checks, chan, wl_r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairing", default="deep_vs_free", choices=list(PAIRINGS))
    ap.add_argument("--pairs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    g = 2
    rel_a, rel_b = PAIRINGS[a.pairing]

    print("=" * 78)
    print(f"PHASE 3 AUDIT -- pairing={a.pairing}  ({rel_a} vs {rel_b})")
    print("=" * 78)

    # --- group-level non-isomorphism certificate ---
    res = hom_certify(g=g, families=[rel_a, rel_b])
    counts = res["counts"]
    sep = [h for h in res["panel"] if counts[rel_a][h] != counts[rel_b][h]]
    print("\npi_1 NON-ISOMORPHISM CERTIFICATE (|Hom(G,H)|)")
    print(f"  {'H':<6}" + "".join(f"{h:>12}" for h in res["panel"]))
    for fam in (rel_a, rel_b):
        print(f"  {fam:<20}" + "".join(f"{counts[fam][h]:>12,}" for h in res["panel"]))
    print(f"  -> {'CERTIFIED DISTINCT via ' + ', '.join(sep) if sep else 'NOT SEPARATED by this panel'}")
    if not sep:
        print("  !! groups not proven distinct -- do not run this pairing")
        return 1

    gate = defaultdict(int); chan = defaultdict(int); att = []; fails = []
    for i in range(a.pairs):
        pr = generate_relator_pair(seed=a.seed + i, pairing=a.pairing)
        ck, ch, wl = audit(pr, g, rel_a, rel_b)
        att.append(pr.certificate.attempts)
        for k, v in ck.items():
            gate[k] += int(bool(v))
        for k, v in ch.items():
            chan[k] += int(bool(v))
        bad = [k for k, v in ck.items() if not v]
        if bad:
            fails.append((a.seed + i, bad))
        print(f"  seed={a.seed+i:>3} attempts={pr.certificate.attempts:>4} wl={wl} "
              f"|V,E,F|={tuple(pr.certificate.rank_counts)} "
              f"gates={'PASS' if not bad else 'FAIL' + str(bad)}")

    n = a.pairs
    print(f"\n  --- GATE STACK (must be {n}/{n}) ---")
    for k in sorted(gate):
        print(f"    {k:<32} {gate[k]:>3}/{n}" + ("" if gate[k] == n else "   <-- FAILURE"))
    print(f"\n  --- CHANNELS (higher = more matched = more closed) ---")
    for k in ["nn_spd_matched", "coarse_metric_matched", "fine_metric_matched",
              "spectral_matched"]:
        print(f"    {k:<32} {chan[k]:>3}/{n}")
    print(f"\n  attempts/pair: min={min(att)} mean={sum(att)/len(att):.1f} max={max(att)}")
    print("\n" + ("RESULT: all pairs pass the full gate stack"
                  if not fails else f"RESULT: FAILURES {fails}"))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
