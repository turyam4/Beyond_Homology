"""
relator_library.py -- presentation families with IDENTICAL homology
====================================================================
Every relator here lies in the commutator subgroup [F,F] of the free group on
2g generators. Consequence, and the reason this whole extension is cheap:

    r in [F,F]  =>  r abelianises to 0
                =>  H_1(X_r) = Z^{2g},  H_2(X_r) = Z,  H_0 = Z
                =>  every family has homology (1, 2g, 1), torsion-free,
                    BY CONSTRUCTION -- the same value the existing
                    surface-vs-free benchmark gates.

So homology matching across families is free. What is NOT free, and must be
certified separately, is that the resulting GROUPS actually differ: see
hom_certificate.py. Nothing here asserts non-isomorphism.

A second consequence of r in [F,F] that the generator relies on: the exponent
sum of every generator is zero, hence #positive = #negative occurrences, hence
EVERY GENERATOR OCCURS AN EVEN NUMBER OF TIMES. That is what makes the
per-generator noise budgeting in the pair generator always solvable (a deficit
is always even and can be filled with x x^-1 pairs).

Slot convention (identical to fg_core): a word is a list of (gen_idx, is_inv)
with gen_idx in 1..2g. For g=2 the generators are a1=1, b1=2, a2=3, b2=4.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

Slot = Tuple[int, bool]


# ---------------------------------------------------------------------------
# word algebra on slots
# ---------------------------------------------------------------------------
def inv(word: List[Slot]) -> List[Slot]:
    """Formal inverse: reverse order, flip each letter."""
    return [(g, not i) for (g, i) in reversed(word)]


def comm(x: List[Slot], y: List[Slot]) -> List[Slot]:
    """[x, y] = x y x^-1 y^-1."""
    return list(x) + list(y) + inv(x) + inv(y)


def gen(i: int) -> List[Slot]:
    return [(i, False)]


def power(word: List[Slot], n: int) -> List[Slot]:
    if n < 0:
        return power(inv(word), -n)
    out: List[Slot] = []
    for _ in range(n):
        out += list(word)
    return out


# ---------------------------------------------------------------------------
# relator families (all in [F,F])
# ---------------------------------------------------------------------------
def r_trivial(g: int) -> List[Slot]:
    """The empty relator: the 2-cell imposes no relation, pi_1 = F_{2g}."""
    return []


def r_surface(g: int) -> List[Slot]:
    """[a1,b1]...[ag,bg] -- the genus-g orientable surface relator."""
    out: List[Slot] = []
    for k in range(1, g + 1):
        out += comm(gen(2 * k - 1), gen(2 * k))
    return out


def r_deep_commutator(g: int) -> List[Slot]:
    """[[a1,b1],[a2,b2]] -- a commutator OF commutators (second derived
    series). Requires g >= 2. Length 16 at g=2."""
    if g < 2:
        raise ValueError("deep_commutator needs g >= 2")
    x = comm(gen(1), gen(2))
    y = comm(gen(3), gen(4))
    return comm(x, y)


def r_commutator_square(g: int) -> List[Slot]:
    """[a1,b1]^2 -- a proper power of a commutator. Note this involves ONLY
    a1,b1; the remaining generators stay free. Length 8 at any g."""
    return power(comm(gen(1), gen(2)), 2)


def r_surface_cubed_first(g: int) -> List[Slot]:
    """[a1,b1]^3 [a2,b2] -- unequal powers across the two handles."""
    if g < 2:
        raise ValueError("needs g >= 2")
    return power(comm(gen(1), gen(2)), 3) + comm(gen(3), gen(4))


RELATORS: Dict[str, Callable[[int], List[Slot]]] = {
    "trivial": r_trivial,
    "surface": r_surface,
    "deep_commutator": r_deep_commutator,
    "commutator_square": r_commutator_square,
    "surface_cubed_first": r_surface_cubed_first,
}


# ---------------------------------------------------------------------------
# structural facts used by the generator (all verifiable, none assumed)
# ---------------------------------------------------------------------------
def signed(word: List[Slot]) -> List[int]:
    return [g if not i else -g for (g, i) in word]


def free_reduce_signed(w: List[int]) -> List[int]:
    stack: List[int] = []
    for x in w:
        if stack and stack[-1] == -x:
            stack.pop()
        else:
            stack.append(x)
    return stack


def exponent_sums(word: List[Slot], g: int) -> Dict[int, int]:
    s = {k: 0 for k in range(1, 2 * g + 1)}
    for (k, i) in word:
        s[k] += -1 if i else 1
    return s


def letter_counts(word: List[Slot], g: int) -> Dict[int, int]:
    c = {k: 0 for k in range(1, 2 * g + 1)}
    for (k, _) in word:
        c[k] += 1
    return c


def is_in_commutator_subgroup(word: List[Slot], g: int) -> bool:
    """r in [F,F]  <=>  every generator has exponent sum 0. This is exactly
    the condition that makes homology (1,2g,1) automatic."""
    return all(v == 0 for v in exponent_sums(word, g).values())


def is_cyclically_reduced(word: List[Slot]) -> bool:
    w = signed(word)
    if free_reduce_signed(w) != w:
        return False
    return not w or w[0] != -w[-1]


def describe(name: str, g: int) -> dict:
    w = RELATORS[name](g)
    red = free_reduce_signed(signed(w))
    return {
        "name": name,
        "length": len(w),
        "freely_reduced_length": len(red),
        "in_commutator_subgroup": is_in_commutator_subgroup(w, g),
        "cyclically_reduced": is_cyclically_reduced(w),
        "letter_counts": letter_counts(w, g),
        "exponent_sums": exponent_sums(w, g),
        "all_counts_even": all(v % 2 == 0 for v in letter_counts(w, g).values()),
        "is_trivial": len(red) == 0,
    }


if __name__ == "__main__":
    import json
    g = 2
    print(f"=== relator families at g={g} (generators a1=1 b1=2 a2=3 b2=4) ===")
    for name in RELATORS:
        d = describe(name, g)
        ok = d["in_commutator_subgroup"] and d["all_counts_even"]
        print(f"\n{name}")
        print(f"  length={d['length']}  reduced_len={d['freely_reduced_length']}"
              f"  trivial={d['is_trivial']}")
        print(f"  in [F,F]={d['in_commutator_subgroup']}  "
              f"cyclically_reduced={d['cyclically_reduced']}  "
              f"all_counts_even={d['all_counts_even']}   -> usable={ok}")
        print(f"  counts={d['letter_counts']}  expsums={d['exponent_sums']}")
