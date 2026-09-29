"""
aggregate_control.py -- fill the R3 pi_1-matched control table
==============================================================
Scans the control-family run trees, sums BREC-solved over model seeds, and
emits the three-row table for the rebuttal, in both plain text and LaTeX.

    Family                            | SMCN binary | SMCN SPD | ESAN
    surface vs. free (cross-class)    |    26/64    |  51/64   | 37/64
    surface vs. surface (pi_1-matched)|    ...      |   ...    |  ...
    free vs. free (pi_1-matched)      |    ...      |   ...    |  ...

Also reports the RELIABILITY pass counts, which is the number the controls
must be read against: a control family sitting at the reliability floor is the
outcome that supports the pi_1 reading, and "floor" means separation ~0 while
reliability stays at 64/64.

Usage:
    python aggregate_control.py
    python aggregate_control.py --root . --cross 26/64,51/64,37/64
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

FAMILY_LABEL = {
    "surface_vs_surface": "surface vs. surface (pi_1-matched)",
    "free_vs_free": "free vs. free (pi_1-matched)",
}
COLUMNS = [("smcn_binary", "SMCN binary"),
           ("smcn_spd", "SMCN SPD"),
           ("esan", "ESAN")]


def collect(root: Path, family: str, column: str):
    """Sum brec_solved / totals across seed dirs for one (family, column)."""
    base = root / f"runs_control_{family}" / column
    solved = total = rel = rel_total = 0
    seeds = []
    if not base.is_dir():
        return None
    for metrics_path in sorted(base.glob("seed*/metrics.json")):
        m = json.loads(metrics_path.read_text())
        solved += int(m.get("brec_solved_count", 0))
        total += int(m.get("brec_solved_total", 0))
        rel += int(m.get("reliability_pass_count", 0))
        rel_total += int(m.get("reliability_total", 0))
        seeds.append((metrics_path.parent.name,
                      int(m.get("brec_solved_count", 0)),
                      int(m.get("target_success_count", 0)),
                      int(m.get("reliability_pass_count", 0)),
                      float(m.get("mean_t2_target", 0.0))))
    if not seeds:
        return None
    return {"solved": solved, "total": total, "rel": rel,
            "rel_total": rel_total, "seeds": seeds}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(SCRIPT_DIR),
                    help="dir containing runs_control_* (default: this folder)")
    ap.add_argument("--cross", default="26/64,51/64,37/64",
                    help="cross-class row: smcn_binary,smcn_spd,esan")
    args = ap.parse_args()
    root = Path(args.root)

    rows = [("surface vs. free (cross-class)",
             [c.strip() for c in args.cross.split(",")], None)]

    for family in ("surface_vs_surface", "free_vs_free"):
        cells, details = [], {}
        for col_key, _ in COLUMNS:
            got = collect(root, family, col_key)
            if got is None:
                cells.append("[[not run]]")
            else:
                cells.append(f"{got['solved']}/{got['total']}")
                details[col_key] = got
        rows.append((FAMILY_LABEL[family], cells, details))

    width = max(len(r[0]) for r in rows) + 2
    print("\n" + "=" * (width + 40))
    print(f"{'Family':<{width}}" + "".join(f"{lbl:>13}" for _, lbl in COLUMNS))
    print("-" * (width + 40))
    for label, cells, _ in rows:
        print(f"{label:<{width}}" + "".join(f"{c:>13}" for c in cells))
    print("=" * (width + 40))

    # reliability + per-seed detail
    for label, _, details in rows:
        if not details:
            continue
        print(f"\n--- {label} ---")
        for col_key, col_lbl in COLUMNS:
            d = details.get(col_key)
            if not d:
                print(f"  {col_lbl:<12}: not run")
                continue
            print(f"  {col_lbl:<12}: solved {d['solved']}/{d['total']}  "
                  f"reliability {d['rel']}/{d['rel_total']}")
            for name, sv, tg, rl, t2 in d["seeds"]:
                print(f"      {name:>8}  brec={sv:<3} target={tg:<3} "
                      f"reliab={rl:<3} mean_t2_target={t2:.4f}")

    print("\n% ---- LaTeX ----")
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(r"Family & SMCN binary & SMCN SPD & ESAN \\")
    print(r"\midrule")
    for label, cells, _ in rows:
        safe = label.replace("pi_1", r"$\pi_1$").replace("_", r"\_")
        print(f"{safe} & " + " & ".join(cells) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

    print("\nREADING THE RESULT")
    print("  controls at the reliability floor (separation ~0, reliability full)")
    print("    -> cross-class separation is class-level, not a realization artifact.")
    print("  controls separating at a high rate")
    print("    -> the family measures realization structure. Report it as the finding.")


if __name__ == "__main__":
    main()
