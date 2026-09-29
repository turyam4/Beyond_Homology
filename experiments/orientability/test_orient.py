"""
test_orient.py  (FINAL)
========================
Certification suite for the three orientability benchmarks, checked against
every claim in oreint.md -- including two questions the document itself
leaves as literal open "?" marks, answered here with actual numbers.
"""
import numpy as np

try:
    from ..orient_common import (
        build_closed_grid,
        build_bounded_grid,
        degree_profile,
        betti_numbers_cw,
        z2_betti_numbers,
        vertical_walk_period,
        belt_loop_lengths,
        edge_face_incidence_profile,
        trace_boundary_loops,
        node_face_distance_histogram,
        puncture_and_twist_are_separated,
        wl_stabilized_check,
    )
except ImportError:  # direct script execution
    from orient_common import (
        build_closed_grid,
        build_bounded_grid,
        degree_profile,
        betti_numbers_cw,
        z2_betti_numbers,
        vertical_walk_period,
        belt_loop_lengths,
        edge_face_incidence_profile,
        trace_boundary_loops,
        node_face_distance_histogram,
        puncture_and_twist_are_separated,
        wl_stabilized_check,
    )

FAIL = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    if not cond:
        FAIL.append(name)
    print(f"[{status}] {name}" + (f"  -- {detail}" if detail else ""))
    return cond


print("=" * 78)
print("PAIR 1: Torus vs Klein Bottle (closed grid)")
print("=" * 78)
h, p = 6, 6
eT, fT = build_closed_grid(h, p, twisted=False)
eK, fK = build_closed_grid(h, p, twisted=True)

check("cell counts match", (h * p, len(eT), len(fT)) == (h * p, len(eK), len(fK)),
      f"T=({h*p},{len(eT)},{len(fT)}) K=({h*p},{len(eK)},{len(fK)})")

degT, degK = degree_profile(eT), degree_profile(eK)
check("every vertex degree 4, both classes", degT == degK == {4: h * p}, f"T={degT} K={degK}")

efiT, efiK = edge_face_incidence_profile(eT, fT), edge_face_incidence_profile(eK, fK)
check("every edge bounds exactly 2 faces, both classes", efiT == efiK == {2: len(eT)},
      f"T={efiT} K={efiK}")

bT, bK = betti_numbers_cw(eT, fT), betti_numbers_cw(eK, fK)
z2T, z2K = z2_betti_numbers(eT, fT), z2_betti_numbers(eK, fK)
print(f"  RATIONAL Betti: Torus={bT}  Klein={bK}")
print(f"  Z2 Betti:       Torus={z2T}  Klein={z2K}")
check("Z2 Betti numbers match (the markdown's claim)", z2T == z2K == (1, 2, 1))
check("RATIONAL Betti numbers DIFFER -- markdown's Z2 analysis is INCOMPLETE, "
     "not wrong but not the whole story: a trivial rank-based Betti "
     "baseline already separates the CLOSED pair without any twist-tracking",
     bT != bK, f"T={bT} K={bK}")

perT = vertical_walk_period(h, p, twisted=False)
perK = vertical_walk_period(h, p, twisted=True)
print(f"  vertical-only closed-walk length: Torus={perT} (want {h})  Klein={perK} (want up to {2*h})")
check("girth/vertical-cycle-length claim confirmed (h vs 2h)", perT == h and perK in (h, 2 * h))

msg, conclusive = wl_stabilized_check(eT, eK, max_rounds=60)
print(f"  1-WL check: {msg}")
check("1-WL genuinely stabilizes indistinguishable (not just 'not yet separated')",
      conclusive is True, msg)
print("  NOTE: this proves PLAIN UNTYPED 1-WL cannot see the twist AT ALL, at any")
print("  depth. The markdown's 'L>=h layers will separate them' claim is only true")
print("  for a model with TYPED/ranked edges (can distinguish vertical vs horizontal)")
print("  -- not for a generic untyped GNN, which this check shows stalls completely.")

print()
print("=" * 78)
print("PAIR 2: Punctured Torus vs Punctured Klein Bottle")
print("=" * 78)
for h, p in [(5, 5), (6, 6)]:
    eT, fT = build_closed_grid(h, p, twisted=False, punctured=True)
    eK, fK = build_closed_grid(h, p, twisted=True, punctured=True)
    print(f"\n h={h} p={p}")
    check(f"  F = h*p-1 exactly (markdown's 'F=24' claim, generalized)",
          len(fT) == len(fK) == h * p - 1, f"F_T={len(fT)} F_K={len(fK)} expect {h*p-1}")
    chiT = h * p - len(eT) + len(fT)
    chiK = h * p - len(eK) + len(fK)
    check("  chi = -1 for both (markdown's claim)", chiT == chiK == -1, f"T={chiT} K={chiK}")

    loopsT = trace_boundary_loops(eT, fT)
    loopsK = trace_boundary_loops(eK, fK)
    print(f"  >>> ANSWERING THE MARKDOWN'S OPEN QUESTION ('SCL will compute ?'):")
    print(f"      punctured Torus boundary loops = {loopsT}")
    print(f"      punctured Klein  boundary loops = {loopsK}")
    check("  exactly 1 boundary loop of length 4, both classes",
          loopsT == loopsK == [4], f"T={loopsT} K={loopsK}")

    sep = puncture_and_twist_are_separated(h, p)
    print(f"  >>> ANSWERING THE MARKDOWN'S OTHER OPEN QUESTION "
         f"('hole and twist never touch ?'): {sep}")
    check(f"  puncture and twist motifs are node-disjoint at h=p={h}", sep)

print("\n checking the markdown's claim that SMALLER grids DO leak (puncture/twist touch):")
for h, p in [(3, 3), (4, 4)]:
    sep = puncture_and_twist_are_separated(h, p)
    check(f"  h=p={h}: puncture and twist OVERLAP as markdown warns (separated={sep})",
          sep is False, f"separated={sep}")

print("\n checking WHY punctured Cylinder/Mobius can't be used instead (markdown's claim):")
def puncture_bounded_faces(h, p, twisted):
    edges, faces = build_bounded_grid(h, p, twisted)
    pi, pj = h // 2, p // 2

    def nid(i, j):
        return i * p + j
    faces2 = [f for f in faces if not (pj < p - 1 and set(f) ==
             {nid(pi, pj), nid(pi, pj + 1), nid(pi + 1, pj), nid(pi + 1, pj + 1)})]
    assert len(faces2) == len(faces) - 1
    return edges, faces2

h, p = 6, 6
eC, fC = puncture_bounded_faces(h, p, twisted=False)
eM, fM = puncture_bounded_faces(h, p, twisted=True)
loopsC = trace_boundary_loops(eC, fC)
loopsM = trace_boundary_loops(eM, fM)
print(f"  punctured Cylinder boundary loops = {loopsC} (count={len(loopsC)}, markdown claims 3)")
print(f"  punctured Mobius   boundary loops = {loopsM} (count={len(loopsM)}, markdown claims 2)")
check("punctured Cylinder has 3 boundary loops as markdown claims", len(loopsC) == 3)
check("punctured Mobius has 2 boundary loops as markdown claims", len(loopsM) == 2)
check("counts mismatch (3 != 2) confirms why this substitution doesn't work",
      len(loopsC) != len(loopsM))

print()
print("=" * 78)
print("PAIR 3: Cylinder vs Mobius Strip (bounded grid)")
print("=" * 78)
h, p = 6, 6
eC, fC = build_bounded_grid(h, p, twisted=False)
eM, fM = build_bounded_grid(h, p, twisted=True)

check("cell counts match", (len(eC), len(fC)) == (len(eM), len(fM)))
degC, degM = degree_profile(eC), degree_profile(eM)
check("degree profile matches (4=interior, 3=boundary)", degC == degM,
      f"C={degC} M={degM}")

bC, bM = betti_numbers_cw(eC, fC), betti_numbers_cw(eM, fM)
check("RATIONAL Betti numbers match (both homotopy-eq to S^1, no leak here at all)",
      bC == bM == (1, 1, 0), f"C={bC} M={bM}")

loopsC = trace_boundary_loops(eC, fC)
loopsM = trace_boundary_loops(eM, fM)
print(f"  boundary loops: Cylinder={loopsC}  Mobius={loopsM}")
check("boundary LOOP COUNT differs (2 vs 1) -- the intended Prop 6.3 signal",
      len(loopsC) == 2 and len(loopsM) == 1)
check("boundary LOOP LENGTH also differs (p,p vs 2p) -- an ADDITIONAL leak "
     "the markdown documents for Torus/Klein's vertical cycles but never "
     "explicitly calls out here, even though it's the same mechanism",
     sorted(loopsC) == [p, p] and loopsM == [2 * p])

msg, conclusive = wl_stabilized_check(eC, eM, max_rounds=60)
print(f"  1-WL check: {msg}")
check("1-WL genuinely stabilizes indistinguishable at the GRAPH level",
      conclusive is True, msg)
print("  (boundary-loop COUNTING is a different, higher-level operation --")
print("  SMCN's SCL layer explicitly counts degree-3-bounding single-face edges,")
print("  which is NOT plain 1-WL; this is the internal tension in the markdown,")
print("  flagged below.)")

print()
print("=" * 78)
print("INTERNAL CONTRADICTION IN THE MARKDOWN (flagged, not resolved by code)")
print("=" * 78)
print("""
The markdown states TWO incompatible things about boundary-counting for
Cylinder vs Mobius:
  (a) Early on: 'As proven in the SMCN paper (Prop 6.3), this is the exact,
      valid topological method the model should use to distinguish them.'
      -- i.e. boundary-loop counting is presented as the CORRECT, intended
      signal.
  (b) Later, in the Master List (Leak 3, 'The Boundary-Counting Proxy'):
      'the model simply counts 2 loops vs 1 loop. It learns arithmetic,
      not continuous manifold twisting.' -- i.e. the SAME mechanism is
      presented as a CHEAT/loophole.
These cannot both be the intended framing. Since I have not independently
verified the actual content of 'Proposition 6.3' in the SMCN paper (I only
have the markdown's characterization of it), I can't resolve which reading
is correct -- but the document as given is internally inconsistent on this
specific point, and a reviewer would flag it. Recommend checking the actual
proposition text before finalizing how this pair is described in a paper.
""")

print()
print("=" * 78)
print("GIRTH EQUALIZATION: Torus vs Klein, girth_matched mode")
print("=" * 78)
for h, p in [(6, 6), (8, 8), (5, 6)]:
    eT, fT = build_closed_grid(h, p, twisted=False, vertical_shift=p // 2)
    eK, fK = build_closed_grid(h, p, twisted=True)
    print(f"\n h={h} p={p} (girth_matched)")
    check("  cell counts still match", (len(eT), len(fT)) == (len(eK), len(fK)))
    chiT, chiK = h * p - len(eT) + len(fT), h * p - len(eK) + len(fK)
    check("  chi still matches", chiT == chiK)
    bT, bK = betti_numbers_cw(eT, fT), betti_numbers_cw(eK, fK)
    check("  rational Betti UNCHANGED from standard torus (still a genuine torus)",
          bT == (1, 2, 1), f"T={bT}")
    check("  degree profile still matches", degree_profile(eT) == degree_profile(eK))
    check("  edge-face incidence still {2:...} for both",
          set(edge_face_incidence_profile(eT, fT)) == set(edge_face_incidence_profile(eK, fK)) == {2})
    perT = [vertical_walk_period(h, p, False, j0=j, vertical_shift=p // 2) for j in range(p)]
    perK = [vertical_walk_period(h, p, True, j0=j) for j in range(p)]
    print(f"  vertical periods (all columns): T={perT}  K={perK}")
    check("  GIRTH NOW MATCHES: both uniformly 2h", all(x == 2 * h for x in perT) and
         all(x == 2 * h for x in perK))
    msg, conclusive = wl_stabilized_check(eT, eK, max_rounds=60)
    check("  still 1-WL indistinguishable after the fix", conclusive is True, msg)

print("\n odd p correctly rejected for girth_matched mode:")
try:
    build_closed_grid(6, 5, twisted=False, vertical_shift=2)
    check("  raises assertion for odd p", False)
except AssertionError:
    check("  raises assertion for odd p", True)

print()
print("=" * 78)
print("METRIC-MATCHED PUNCTURED TORUS/KLEIN: 50-PAIR CERTIFICATION")
print("=" * 78)
metric_checks = []
for h in range(6, 106, 2):
    p = 4
    torus = build_closed_grid(
        h, p, twisted=False, punctured=True, vertical_shift=p // 2
    )
    klein = build_closed_grid(h, p, twisted=True, punctured=True)
    torus_periods = tuple(
        vertical_walk_period(h, p, False, j0=j, vertical_shift=p // 2)
        for j in range(p)
    )
    klein_periods = tuple(
        vertical_walk_period(h, p, True, j0=j) for j in range(p)
    )
    metric_checks.append(
        node_face_distance_histogram(*torus, max_dist=10)
        == node_face_distance_histogram(*klein, max_dist=10)
        and trace_boundary_loops(*torus) == trace_boundary_loops(*klein) == [4]
        and degree_profile(torus[0]) == degree_profile(klein[0])
        and torus_periods == klein_periods
    )
check(
    "all 50 pairs match distance, boundary, degree, and vertical-cycle probes",
    all(metric_checks) and len(metric_checks) == 50,
)

print()
print("=" * 78)
print("BOUNDARY EQUALIZATION INVESTIGATION: Cylinder vs Mobius")
print("=" * 78)
print("""
First attempt (kept here as a documented mistake, not swept away): tried
mixing rows in Cylinder's horizontal wrap via row -> row+shift (mod h), by
direct analogy to the Torus/Klein girth fix. This looked promising at
shift=1 (2 boundary loops, length 7 instead of 6) -- but checking the
complex's validity caught that it was BROKEN: faces referenced edges
between shifted rows that were never added, because the construction
silently assumed the vertical direction is cyclic (mod h) when it
explicitly is not (that's what "bounded" means). Confirmed two ways: (a)
betti_numbers_cw raised KeyError -- a face cited a nonexistent edge; (b)
checking node-level boundary degree directly found a genuine non-manifold
point (one node touching 4 boundary edges instead of 2) even in cases that
didn't crash. So that route doesn't work, and the initial "it looks
partially fixed" reading was wrong -- recorded here rather than deleted.
""")

from itertools import permutations


def is_involution(perm):
    return all(perm[perm[i]] == i for i in range(len(perm)))


def preserves_adjacency(perm, h):
    return all(abs(perm[i + 1] - perm[i]) == 1 for i in range(h - 1))


print("The actual reason (exhaustively verified, not just argued):")
for h in [4, 5, 6]:
    valid = [perm for perm in permutations(range(h))
            if is_involution(list(perm)) and preserves_adjacency(perm, h)]
    print(f"  h={h}: adjacency-preserving involutions of the OPEN chain = {valid}")
check("only identity and full-reversal preserve adjacency on an open chain "
     "(h=4,5,6, exhaustive)",
     all(len([perm for perm in permutations(range(h))
             if is_involution(list(perm)) and preserves_adjacency(perm, h)]) == 2
        for h in [4, 5, 6]))
print("""
Torus/Klein's vertical direction IS cyclic (wrapped), so its identification
can use any element of the p-cycle's dihedral symmetry group (p rotations
+ p reflections) -- that's where vertical_shift=p//2 comes from. Cylinder/
Mobius's row structure is an OPEN chain (not wrapped -- that's what bounded
means), and an open chain's adjacency-preserving symmetry group is just Z_2:
identity (Cylinder) and full reversal (Mobius). There is no third option
and no shift-like freedom to borrow the girth fix from. Structural, checked,
not a failure to search harder.
""")

print("=" * 78)
if FAIL:
    print(f"RESULT: {len(FAIL)} check(s) FAILED: {FAIL}")
else:
    print("RESULT: all checks passed")
print("=" * 78)
