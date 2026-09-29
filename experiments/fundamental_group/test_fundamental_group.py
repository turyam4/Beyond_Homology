"""
test_fundamental_group.py  (FINAL)
====================================
Verification suite for the Free-vs-Surface-Group benchmark, checked against
the full researcher checklist. Every test prints measured numbers; nothing
here is asserted without being computed first.

Run: python3 test_fundamental_group.py
"""
import os
import sys
from collections import Counter
from datetime import datetime
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SCRIPT_DIR = os.path.dirname(__file__)
FG_GENUS_2_DIR = os.path.join(SCRIPT_DIR, "FG_genus_2")
for path in (FG_GENUS_2_DIR, SCRIPT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from fg_core import (
    generate_words,
    create_complex_edges_faces,
    degree_profile,
    betti_numbers,
    wl_rounds_to_separate,
    word_backtracks,
    build_base_commutator_slots,
    signed_word,
    free_reduce,
    is_rotation,
    word_inverse,
    verify_class_A_reduces_to_commutator,
    verify_class_B_reduces_to_identity,
    slot_letter_counts,
    slot_signed_exponent_sums,
    random_trivial_slots,
)

FAIL = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    if not cond:
        FAIL.append(name)
    print(f"[{status}] {name}" + (f"  -- {detail}" if detail else ""))
    return cond


# ============================================================================
# 1. same cell count
# ============================================================================
def test_cell_count(rng, n_trials=40):
    print("\n=== 1. Same cell count ===")
    ok = True
    for _ in range(n_trials):
        g = int(rng.integers(1, 5))
        noise = int(rng.integers(0, 4))
        WA, WB, p = generate_words(g, noise, rng, min_subdiv=3, max_subdiv=6)
        eA, fA = create_complex_edges_faces(WA)
        eB, fB = create_complex_edges_faces(WB)
        vA = {v for e in eA for v in e}
        vB = {v for e in eB for v in e}
        same = (len(vA), len(eA), len(fA)) == (len(vB), len(eB), len(fB))
        ok &= same
        if not same:
            print(f"  MISMATCH g={g} p={p} noise={noise}")
    check("same cell count (|V|,|E|,|F|) across 40 random (g, noise, p)", ok)


# ============================================================================
# 2. same Euler characteristic
# ============================================================================
def test_euler(rng, n_trials=40):
    print("\n=== 2. Same Euler characteristic ===")
    ok = True
    for _ in range(n_trials):
        g = int(rng.integers(1, 5))
        noise = int(rng.integers(0, 4))
        WA, WB, p = generate_words(g, noise, rng, min_subdiv=3, max_subdiv=6)
        eA, fA = create_complex_edges_faces(WA)
        eB, fB = create_complex_edges_faces(WB)
        vA = {v for e in eA for v in e}
        vB = {v for e in eB for v in e}
        chiA = len(vA) - len(eA) + len(fA)
        chiB = len(vB) - len(eB) + len(fB)
        ok &= (chiA == chiB)
    check("chi(A) == chi(B) across 40 random (g, noise, p)", ok)


# ============================================================================
# 3. same homology
# ============================================================================
def test_homology(rng, n_trials=30):
    print("\n=== 3. Same homology (rational Betti numbers) ===")
    ok = True
    for _ in range(n_trials):
        g = int(rng.integers(1, 4))
        noise = int(rng.integers(0, 3))
        WA, WB, p = generate_words(g, noise, rng, min_subdiv=3, max_subdiv=5)
        eA, fA = create_complex_edges_faces(WA)
        eB, fB = create_complex_edges_faces(WB)
        bA, bB = betti_numbers(eA, fA), betti_numbers(eB, fB)
        expected = (1, 2 * g, 1)
        match = bA == bB == expected
        ok &= match
        if not match:
            print(f"  MISMATCH g={g}: betti_A={bA} betti_B={bB} expected={expected}")
    check(f"betti(A) == betti(B) == (1,2g,1) across {n_trials} random samples", ok)


# ============================================================================
# 4. same local incidence statistics
# ============================================================================
def test_local_incidence(rng, n_trials=60):
    print("\n=== 4. Same local incidence statistics (degree profile) ===")
    n_match = 0
    for _ in range(n_trials):
        g = int(rng.integers(1, 4))
        noise = int(rng.integers(0, 5))
        WA, WB, p = generate_words(g, noise, rng, min_subdiv=3, max_subdiv=6)
        eA, fA = create_complex_edges_faces(WA)
        eB, fB = create_complex_edges_faces(WB)
        n_match += int(degree_profile(eA) == degree_profile(eB))
    check(f"degree profile exact match: {n_match}/{n_trials} trials",
          n_match == n_trials, f"{n_match}/{n_trials}")


# ============================================================================
# 5. DIFFERENT pi_1 -- the algebraic claim itself, not just its topological
#    consequences (this is the check that was missing before this rewrite)
# ============================================================================
def test_group_theory(rng, n_trials=2000):
    print("\n=== 5. Different pi_1 (algebraic verification, not just topological consequences) ===")
    ok_A, ok_B = 0, 0
    for _ in range(n_trials):
        g = int(rng.integers(1, 5))
        noise = int(rng.integers(0, 5))
        WA, WB, p, (A_slots, B_slots) = generate_words(
            g, noise, rng, min_subdiv=3, max_subdiv=6, return_slots=True)
        ok_A += int(verify_class_A_reduces_to_commutator(A_slots, g))
        ok_B += int(verify_class_B_reduces_to_identity(B_slots))
    check(f"class A free-reduces EXACTLY to the base commutator: {ok_A}/{n_trials}",
          ok_A == n_trials, f"{ok_A}/{n_trials}")
    check(f"class B free-reduces to the identity (freely null relator): {ok_B}/{n_trials}",
          ok_B == n_trials, f"{ok_B}/{n_trials}")
    print("  (class A reducing exactly to the commutator is a theorem given")
    print("   Tietze insertion -- this checks the CODE implements it correctly,")
    print("   not whether the theorem is true.)")


def test_shift_reversal_are_conjugation_inversion(rng, n_trials=500):
    print("\n=== 5b. Cyclic shift = conjugation, reversal = inversion (generic check) ===")
    p = 3
    g = 2

    def node_id(mg, t):
        return 1 + (mg - 1) * (p - 1) + (t - 1)

    def get_path(gi, is_inv):
        inner = [node_id(gi, t) for t in range(1, p)]
        if is_inv:
            inner = inner[::-1]
        return [0] + inner

    base_slots = build_base_commutator_slots(g)
    base_word = signed_word(base_slots)
    path0 = []
    for gi, inv in base_slots:
        path0 += get_path(gi, inv)

    def parse_path_to_slots(path):
        n = len(path)
        zeros = [i for i, v in enumerate(path) if v == 0]
        slots = []
        for idx, z in enumerate(zeros):
            znext = zeros[(idx + 1) % len(zeros)]
            length = (znext - z) % n or n
            chunk = [path[(z + k) % n] for k in range(length)]
            nonzero = chunk[1:]
            mg = 1 + (nonzero[0] - 1) // (p - 1)
            is_inv = nonzero != sorted(nonzero)
            slots.append((mg, is_inv))
        return slots

    fails = 0
    for _ in range(n_trials):
        shift = int(rng.integers(0, len(path0) // p)) * p
        shifted = path0[shift:] + path0[:shift]
        do_rev = rng.random() > 0.5
        final = shifted[::-1] if do_rev else shifted
        rec = signed_word(parse_path_to_slots(final))
        target = word_inverse(base_word) if do_rev else base_word
        if not is_rotation(target, rec):
            fails += 1
    check(f"shift/reversal correctly implement conjugation/inversion: "
          f"{n_trials - fails}/{n_trials}", fails == 0, f"{n_trials-fails}/{n_trials}")


# ============================================================================
# 6. signed exponent sums (necessary, not remarkable -- stated plainly)
# ============================================================================
def test_exponent_sums(rng, n_trials=30):
    print("\n=== 6. Signed exponent sums ===")
    ok = True
    for _ in range(n_trials):
        g = int(rng.integers(1, 5))
        noise = int(rng.integers(0, 5))
        WA, WB, p, (A_slots, B_slots) = generate_words(
            g, noise, rng, return_slots=True)
        sA = slot_signed_exponent_sums(A_slots, g)
        sB = slot_signed_exponent_sums(B_slots, g)
        ok &= all(v == 0 for v in sA.values()) and all(v == 0 for v in sB.values())
    check("signed exponent sum is 0 for every generator, both classes", ok)
    print("  (true for ANY commutator product and ANY freely-trivial word --")
    print("   a necessary consequence of the construction, not a special")
    print("   achievement; included for completeness, not to overclaim.)")


# ============================================================================
# 7. absolute occurrence counts -- checked at the WORD level directly,
#    not just inferred from the downstream degree profile
# ============================================================================
def test_occurrence_counts(rng, n_trials=60):
    print("\n=== 7. Absolute occurrence counts (word-level, direct check) ===")
    n_match = 0
    for _ in range(n_trials):
        g = int(rng.integers(1, 5))
        noise = int(rng.integers(0, 5))
        WA, WB, p, (A_slots, B_slots) = generate_words(
            g, noise, rng, return_slots=True)
        cA, cB = slot_letter_counts(A_slots), slot_letter_counts(B_slots)
        n_match += int(cA == cB)
    check(f"per-generator occurrence count exact match (word level): "
          f"{n_match}/{n_trials}", n_match == n_trials, f"{n_match}/{n_trials}")


# ============================================================================
# 8. randomized permutation
# ============================================================================
def test_randomized_permutation(rng, n_trials=20):
    print("\n=== 8. Randomized permutation (relabel/flip/shift/reversal) ===")
    seen = set()
    for _ in range(n_trials):
        WA, WB, p = generate_words(2, 1, rng, min_subdiv=3, max_subdiv=3)
        seen.add(tuple(WA))
    check(f"distinct WA sequences across {n_trials} same-(g,noise,p) draws: {len(seen)}",
          len(seen) == n_trials, f"{len(seen)}/{n_trials} unique")


# ============================================================================
# 9. randomized subdivision
# ============================================================================
def test_randomized_subdivision(rng, n_trials=30):
    print("\n=== 9. Randomized subdivision ===")
    ps = []
    for _ in range(n_trials):
        WA, WB, p = generate_words(2, 1, rng, min_subdiv=3, max_subdiv=8)
        ps.append(p)
    distinct = len(set(ps))
    check(f"subdivision level p varies across samples: observed values {sorted(set(ps))}",
          distinct >= 3, f"{distinct} distinct values in [3,8]")
    g = 2
    WA, WB, p = generate_words(g, 1, rng, min_subdiv=5, max_subdiv=5)
    eA, fA = create_complex_edges_faces(WA)
    eB, fB = create_complex_edges_faces(WB)
    vA, vB = {v for e in eA for v in e}, {v for e in eB for v in e}
    check("A and B share the same p within one pair (cell counts still match at p=5)",
          (len(vA), len(eA), len(fA)) == (len(vB), len(eB), len(fB)))
    print("  Note: 'subdivide then simplify' Tietze rewrites are NOT")
    print("  implemented literally -- the equivalent outcome (variable")
    print("  discretization density) is achieved directly via min/max_subdiv.")
    print("  Deliberate design choice, not a gap.")


# ============================================================================
# 10. immediate inverse adjacency (backtrack) count matching
# ============================================================================
def test_backtrack_matching(rng, n_trials=3000):
    print("\n=== 10. Immediate inverse adjacency (backtrack) count matching ===")
    for label, match in [("match_backtracks=False (no mitigation)", False),
                        ("match_backtracks=True  (reject + top-up)", True)]:
        btA_all, btB_all = [], []
        for _ in range(n_trials):
            g = int(rng.integers(1, 5))
            noise = int(rng.integers(1, 4))
            WA, WB, p = generate_words(g, noise, rng, min_subdiv=3, max_subdiv=3,
                                       match_backtracks=match, max_tries=25)
            btA_all.append(word_backtracks(WA))
            btB_all.append(word_backtracks(WB))
        btA_all, btB_all = np.array(btA_all), np.array(btB_all)
        exact = np.mean(btA_all == btB_all)
        labels = np.array([0] * n_trials + [1] * n_trials)
        scores = np.concatenate([btA_all, btB_all])
        best_acc = max(((scores > t).astype(int) == labels).mean()
                      for t in range(int(scores.max()) + 1))
        print(f"  {label}: exact match={exact*100:.1f}%  "
              f"best-threshold exploit accuracy={best_acc:.4f} (chance=0.500)")
        if match:
            check("match_backtracks=True brings exploit accuracy to chance (<=0.505)",
                  best_acc <= 0.505, f"{best_acc:.4f}")
            check("match_backtracks=True achieves exact backtrack match >=99% of the time",
                  exact >= 0.99, f"{exact*100:.1f}%")
            deg_ok, betti_ok = True, True
            for _ in range(30):
                g2 = int(rng.integers(1, 4))
                WA2, WB2, p2 = generate_words(g2, 2, rng, min_subdiv=3, max_subdiv=6,
                                             match_backtracks=True, max_tries=25)
                eA2, fA2 = create_complex_edges_faces(WA2)
                eB2, fB2 = create_complex_edges_faces(WB2)
                deg_ok &= (degree_profile(eA2) == degree_profile(eB2))
                betti_ok &= (betti_numbers(eA2, fA2) == betti_numbers(eB2, fB2) == (1, 2 * g2, 1))
            check("degree profile & homology still match with match_backtracks=True",
                  deg_ok and betti_ok)
        else:
            check("match_backtracks=False reproduces the known ~0.61 exploit (informational)",
                  best_acc > 0.55, f"{best_acc:.4f}")


# ============================================================================
# 11. WL separation behavior -- the fact that must not go unstated
# ============================================================================
def test_wl_behavior(rng, n_trials=8):
    print("\n=== 11. 1-WL separation behavior ===")
    print(" match_backtracks=True (default), noise_pairs_per_g=0:")
    print(" (NOTE: no longer expected to be indistinguishable -- see test 12)")
    for _ in range(n_trials):
        g = int(rng.integers(1, 4))
        WA, WB, p = generate_words(g, 0, rng, min_subdiv=3, max_subdiv=3,
                                   match_backtracks=True)
        eA, fA = create_complex_edges_faces(WA)
        eB, fB = create_complex_edges_faces(WB)
        r = wl_rounds_to_separate((eA, fA), (eB, fB), max_rounds=25, hasse=False)
        print(f"   g={g} noise=0  -> separated at round: {r}")
    print(" match_backtracks=False, noise_pairs_per_g=0 (the ORIGINAL indistinguishable case):")
    none_count = 0
    for _ in range(n_trials):
        g = int(rng.integers(1, 4))
        WA, WB, p = generate_words(g, 0, rng, min_subdiv=3, max_subdiv=3,
                                   match_backtracks=False)
        eA, fA = create_complex_edges_faces(WA)
        eB, fB = create_complex_edges_faces(WB)
        r = wl_rounds_to_separate((eA, fA), (eB, fB), max_rounds=25, hasse=False)
        none_count += int(r is None)
        print(f"   g={g} noise=0  -> separated at round: {r}")
    check("match_backtracks=False + noise=0 reproduces 1-WL-indistinguishability",
          none_count == n_trials, f"{none_count}/{n_trials} indistinguishable")
    print(" match_backtracks=True, noise_pairs_per_g=2 (typical training config):")
    for _ in range(n_trials):
        g = int(rng.integers(1, 4))
        WA, WB, p = generate_words(g, 2, rng, min_subdiv=3, max_subdiv=3)
        eA, fA = create_complex_edges_faces(WA)
        eB, fB = create_complex_edges_faces(WB)
        r = wl_rounds_to_separate((eA, fA), (eB, fB), max_rounds=25, hasse=False)
        print(f"   g={g} noise=2  -> separated at round: {r}")
    print(" (informational only -- documented design property, not pass/fail.)")


# ============================================================================
# 12. the match_backtracks / noise_pairs_per_g=0 interaction, made explicit
# ============================================================================
def test_backtrack_noise_interaction(rng, n_trials=200):
    print("\n=== 12. match_backtracks vs noise_pairs_per_g=0 interaction ===")
    fired = 0
    for _ in range(n_trials):
        g = int(rng.integers(1, 4))
        WA, WB, p, (A_slots, B_slots) = generate_words(
            g, 0, rng, min_subdiv=3, max_subdiv=3,
            match_backtracks=True, return_slots=True)
        if len(B_slots) > 2 * g:
            fired += 1
    print(f"  with noise_pairs_per_g=0, match_backtracks=True still injects shared "
          f"noise in {fired}/{n_trials} samples")
    print("  This is EXPECTED given how the fix works (see fg_core.py docstring),")
    print("  not a bug -- but it means noise_pairs_per_g=0 alone does NOT give a")
    print("  noise-free word once match_backtracks=True. Use match_backtracks=False")
    print("  too if you specifically want the zero-noise, 1-WL-indistinguishable case.")


# ============================================================================
# plotting: visualize a Class A / Class B pair side by side
# ============================================================================
def plot_pair(g: int, noise_pairs_per_g: int, seed: int = 0,
             min_subdiv: int = 3, max_subdiv: int = 3,
             match_backtracks: bool = True,
             save_path: str = "fg_pair.png"):
    """Schematic visualization of one generated (Class A, Class B) pair:
    1-skeleton graph layout, colored by node role, plus a degree-histogram
    overlay. NOT a faithful surface embedding (genus-g surfaces can't be
    drawn in the plane without self-intersection) -- a structural
    diagnostic, for spot-checking before trusting the printed assertions."""
    rng = np.random.default_rng(seed)
    WA, WB, p = generate_words(g, noise_pairs_per_g, rng,
                               min_subdiv=min_subdiv, max_subdiv=max_subdiv,
                               match_backtracks=match_backtracks, max_tries=25)
    eA, fA = create_complex_edges_faces(WA)
    eB, fB = create_complex_edges_faces(WB)

    def node_role(n, W_path):
        c_center = max(W_path) + 1
        if n == 0:
            return "basepoint"
        if n == c_center:
            return "disk center"
        if n < c_center:
            return "generator node"
        return "disk boundary"

    role_color = {
        "basepoint": "#d62728",
        "generator node": "#1f77b4",
        "disk center": "#ff7f0e",
        "disk boundary": "#2ca02c",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (label, W_path, edges) in zip(
            axes[:2], [("Class A (surface relator)", WA, eA),
                      ("Class B (scrambled trivial word)", WB, eB)]):
        Gnx = nx.Graph()
        Gnx.add_edges_from(edges)
        colors = [role_color[node_role(n, W_path)] for n in Gnx.nodes()]
        pos = nx.spring_layout(Gnx, seed=0, k=None)
        nx.draw_networkx_edges(Gnx, pos, ax=ax, alpha=0.25, width=0.6)
        nx.draw_networkx_nodes(Gnx, pos, ax=ax, node_size=40,
                               node_color=colors, linewidths=0)
        ax.set_title(f"{label}\n|V|={Gnx.number_of_nodes()} |E|={Gnx.number_of_edges()} "
                    f"(g={g}, p={p}, noise/g={noise_pairs_per_g}, "
                    f"match_backtracks={match_backtracks})")
        ax.axis("off")

    handles = [plt.Line2D([0], [0], marker='o', color='w', label=k,
                          markerfacecolor=v, markersize=8)
              for k, v in role_color.items()]
    axes[0].legend(handles=handles, loc="lower left", fontsize=8)

    dpA, dpB = degree_profile(eA), degree_profile(eB)
    all_degs = sorted(set(dpA) | set(dpB))
    width = 0.35
    x = np.arange(len(all_degs))
    axes[2].bar(x - width / 2, [dpA.get(d, 0) for d in all_degs], width, label="Class A")
    axes[2].bar(x + width / 2, [dpB.get(d, 0) for d in all_degs], width, label="Class B")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(all_degs)
    axes[2].set_xlabel("node degree")
    axes[2].set_ylabel("# nodes")
    axes[2].set_title(f"Degree histogram overlay\n(exact match: {dpA == dpB})")
    axes[2].legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=140)
    plt.close()
    print(f"\nSaved pair visualization to {save_path}")
    return save_path


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    test_cell_count(rng)
    test_euler(rng)
    test_homology(rng)
    test_local_incidence(rng)
    test_group_theory(rng)
    test_shift_reversal_are_conjugation_inversion(rng)
    test_exponent_sums(rng)
    test_occurrence_counts(rng)
    test_randomized_permutation(rng)
    test_randomized_subdivision(rng)
    test_backtrack_matching(rng)
    test_wl_behavior(rng)
    test_backtrack_noise_interaction(rng)

    print("\n" + "=" * 70)
    if FAIL:
        print(f"RESULT: {len(FAIL)} check(s) FAILED: {FAIL}")
    else:
        print("RESULT: all checks passed")
    print("=" * 70)

    output_dir = os.path.join(PROJECT_ROOT, "test_outputs")
    plot_pair(
        g=2,
        noise_pairs_per_g=2,
        seed=1,
        min_subdiv=3,
        max_subdiv=3,
        save_path=os.path.join(
            output_dir,
            f"fg_pair_example_{datetime.now():%Y%m%d_%H%M%S}.png",
        ),
    )
