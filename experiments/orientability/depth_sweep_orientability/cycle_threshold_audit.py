#!/usr/bin/env python3
"""Deterministic cycle-visibility audit for the punctured orientability family.

For every (h, p) in the paper's punctured sweep grid this computes, exactly:

  L*(h,p)  smallest closed-walk length at which the punctured torus and the
           punctured Klein bottle differ: tr(A_T^l) != tr(A_K^l). Computed in
           exact integer arithmetic modulo two 61-bit primes, so no
           floating-point tolerance enters.
  R*(h,p)  ceil(L*/2), the receptive radius at which that signature first fits
           inside a message-passing model's receptive field.
  W*(h,p)  round at which rank-coloured Hasse 1-WL first separates the pair
           (independent refinement of each complex, compared by colour-class
           histogram) -- the analogous bound for typed message passing.

Closed walks rather than simple cycles because they are exactly computable at
every grid size and are precisely what a depth-L model can access: its
receptive field is the ball of radius L, so walk statistics of length <= 2L are
reachable. Simple-cycle counts are computed by brute force on the small grids
as a cross-check that the two statistics agree on the threshold.
"""
from __future__ import annotations
import sys, json, csv
from pathlib import Path
from collections import Counter

import numpy as np

NEW_EXP = Path.home() / "Documents/Experiments/SMCN/new_experiments"
sys.path.insert(0, str(NEW_EXP))
sys.path.insert(0, str(NEW_EXP / "orientability"))
from orient_common import build_closed_grid  # noqa: E402

sys.path.insert(0, str(Path.home() / "Documents/LLM-agents/code-claude-smcn/punc_torus_vs_klein_run"))
from brutal_audit import hasse_wl, count_cycles, adj_list  # noqa: E402

HERE = Path(__file__).resolve().parent
H_RANGE = list(range(4, 14))
P_RANGE = [4, 6, 8, 10, 12]
MAX_WALK = 40
PRIMES = (2305843009213693951, 4611686018427387847)   # two 61/62-bit primes


def build_pair(h, p):
    e_t, f_t = build_closed_grid(h, p, twisted=False, punctured=True,
                                 vertical_shift=p // 2)
    e_k, f_k = build_closed_grid(h, p, twisted=True, punctured=True)
    return (e_t, f_t), (e_k, f_k)


def adjacency(edges, n):
    A = np.zeros((n, n), dtype=object)
    for a, b in edges:
        A[a, b] = 1
        A[b, a] = 1
    return A


def traces_mod(edges, n, max_len, prime):
    """[tr(A^l) mod prime for l = 1..max_len], exact integer arithmetic."""
    A = np.zeros((n, n), dtype=np.int64)
    for a, b in edges:
        A[a, b] = 1
        A[b, a] = 1
    Al = np.eye(n, dtype=object) % prime
    A_obj = A.astype(object)
    out = []
    for _ in range(max_len):
        Al = (Al @ A_obj) % prime
        out.append(int(np.trace(Al) % prime))
    return out


def first_walk_difference(e_t, e_k, n, max_len=MAX_WALK):
    for prime in PRIMES:
        tt = traces_mod(e_t, n, max_len, prime)
        tk = traces_mod(e_k, n, max_len, prime)
        diff = [l + 1 for l in range(max_len) if tt[l] != tk[l]]
        if not diff:
            return None, None, None
        cand = diff[0]
        if prime == PRIMES[0]:
            first = cand
        else:
            if cand != first:          # modular collision -> take the later, safer value
                first = max(first, cand)
    return first, tt[first - 1], tk[first - 1]


def wl_separation_round(pa, pb, n):
    ha = hasse_wl(pa[0], pa[1], n)
    hb = hasse_wl(pb[0], pb[1], n)
    for r in range(min(len(ha), len(hb))):
        if ha[r] != hb[r]:
            return r
    return None


def main():
    rows = []
    print("# Deterministic cycle-visibility audit, punctured orientability family\n")
    print("| h | p | V/E/F | L* | L*-(h+1) | R*=ceil(L*/2) | Hasse-WL round |")
    print("|---|---|---|---|---|---|---|")
    for h in H_RANGE:
        for p in P_RANGE:
            (e_t, f_t), (e_k, f_k) = build_pair(h, p)
            e_t = [tuple(sorted(e)) for e in e_t]
            e_k = [tuple(sorted(e)) for e in e_k]
            n = max(max(e) for e in e_t) + 1
            L, tt, tk = first_walk_difference(e_t, e_k, n)
            W = wl_separation_round((e_t, f_t), (e_k, f_k), n)
            R = (L + 1) // 2 if L else None
            rows.append(dict(h=h, p=p, V=n, E=len(e_t), F=len(f_t),
                             L_star=L, R_star=R, wl_round=W,
                             delta=(L - (h + 1)) if L else None))
            print(f"| {h} | {p} | {n}/{len(e_t)}/{len(f_t)} | {L} | "
                  f"{(L-(h+1)) if L else '-'} | {R} | {W} |")
    with open(HERE / "cycle_threshold_audit.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    json.dump(rows, open(HERE / "cycle_threshold_audit.json", "w"), indent=1)

    ok = [r for r in rows if r["L_star"]]
    print(f"\n- L* == h+1 exactly on **{sum(1 for r in ok if r['delta']==0)}/{len(ok)}** pairs")
    print(f"- deviations observed: {sorted({r['delta'] for r in ok})}")
    wl = [r["wl_round"] for r in ok if r["wl_round"] is not None]
    print(f"- Hasse-WL separation round: min {min(wl)}, max {max(wl)}")
    print("\n## simple-cycle cross-check (brute force, small grids)\n")
    print("| h | p | length | torus | klein |")
    print("|---|---|---|---|---|")
    for (h, p) in [(4, 6), (5, 6), (6, 6), (7, 6)]:
        (e_t, f_t), (e_k, f_k) = build_pair(h, p)
        n = max(max(e) for e in e_t) + 1
        at, ak = adj_list(n, e_t), adj_list(n, e_k)
        for L in (h, h + 1):
            print(f"| {h} | {p} | {L} | {count_cycles(n, at, L)} | {count_cycles(n, ak, L)} |")
    print(f"\nwrote {HERE/'cycle_threshold_audit.csv'}")


if __name__ == "__main__":
    main()
