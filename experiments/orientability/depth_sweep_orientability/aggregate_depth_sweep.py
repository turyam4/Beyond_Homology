#!/usr/bin/env python3
"""Aggregate the punctured-orientability depth sweep into the appendix tables.

Emits:
  Table A  solved / 50 per depth per seed, plus the all-seed count.
  Table B  pass rate per (depth, h), the empirical cutoff h_max(L), and the
           deterministic prediction h <= 2L + 1 from the cycle audit.
"""
from __future__ import annotations
import csv, json, re
from pathlib import Path
import collections

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
AUDIT = HERE / "cycle_threshold_audit.csv"


def load_runs():
    out = {}
    for d in sorted(RUNS.glob("L*_seed*")):
        m = re.match(r"L(\d+)_seed(\d+)", d.name)
        if not m:
            continue
        f = d / "pair_results.csv"
        if not f.exists():
            continue
        rows = [r for r in csv.DictReader(open(f)) if r["trial_type"].startswith("target")]
        rel = [r for r in csv.DictReader(open(f)) if r["trial_type"].startswith("reliability")]
        if not rows:
            continue
        out[(int(m.group(1)), int(m.group(2)))] = dict(
            target={(int(r["h"]), int(r["p"])): r["passed"] == "True" for r in rows},
            t2={(int(r["h"]), int(r["p"])): float(r["t2_score"]) for r in rows},
            rel_pass=sum(r["passed"] == "True" for r in rel), n_rel=len(rel))
    return out


def main():
    runs = load_runs()
    if not runs:
        print("no completed runs yet"); return
    audit = {(int(r["h"]), int(r["p"])): int(r["L_star"]) for r in csv.DictReader(open(AUDIT))} \
        if AUDIT.exists() else {}
    depths = sorted({L for L, _ in runs})
    seeds = sorted({s for _, s in runs})
    keys = sorted(set().union(*[set(v["target"]) for v in runs.values()]))
    hs = sorted({h for h, _ in keys})

    print("# Depth sweep, punctured orientability pair, binary marking, fresh model per pair\n")
    print("## Table A — pairs solved (of %d)\n" % len(keys))
    print("| subgraph layers L | " + " | ".join(f"seed {s}" for s in seeds)
          + " | mean | solved by all seeds | reliability |")
    print("|---|" + "---|" * (len(seeds) + 3))
    tableA = {}
    for L in depths:
        per, rel = [], []
        for s in seeds:
            r = runs.get((L, s))
            per.append(sum(r["target"].values()) if r else None)
            if r: rel.append(f"{r['rel_pass']}/{r['n_rel']}")
        avail = [(L, s) for s in seeds if (L, s) in runs]
        allseed = sum(1 for k in keys
                      if all(runs[(L, s)]["target"].get(k) for (L, s) in avail)) if avail else 0
        mean = sum(x for x in per if x is not None) / max(1, len([x for x in per if x is not None]))
        tableA[L] = dict(per_seed=per, all_seed=allseed, mean=mean)
        print(f"| {L} | " + " | ".join("-" if x is None else str(x) for x in per)
              + f" | {mean:.1f} | **{allseed}/{len(keys)}** | {', '.join(sorted(set(rel)))} |")

    print("\n## Table B — pass rate by grid height h (all seeds pooled)\n")
    print("| L | " + " | ".join(f"h={h}" for h in hs) + " | empirical cutoff | bound h ≤ 2L+1 |")
    print("|---|" + "---|" * (len(hs) + 2))
    tableB = {}
    for L in depths:
        avail = [(L, s) for s in seeds if (L, s) in runs]
        cells, cutoff = [], None
        for h in hs:
            ks = [k for k in keys if k[0] == h]
            tot = len(ks) * len(avail)
            got = sum(1 for k in ks for a in avail if runs[a]["target"].get(k))
            cells.append(f"{got}/{tot}")
            if got > 0:
                cutoff = h
        tableB[L] = dict(cells=cells, cutoff=cutoff, bound=2 * L + 1)
        print(f"| {L} | " + " | ".join(cells) + f" | **h ≤ {cutoff}** | h ≤ {2*L+1} |")

    print("\n## Table C — empirical cutoff against the deterministic bound\n")
    print("| L | shortest differing walk visible? | predicted cutoff h ≤ 2L+1 | observed cutoff | slack |")
    print("|---|---|---|---|---|")
    for L in depths:
        c, b = tableB[L]["cutoff"], tableB[L]["bound"]
        print(f"| {L} | walks of length ≤ {2*L+2} | h ≤ {b} | "
              f"{'none' if c is None else f'h ≤ {c}'} | "
              f"{'-' if c is None else b - c} |")

    json.dump(dict(tableA=tableA, tableB={k: v for k, v in tableB.items()}),
              open(HERE / "depth_sweep_summary.json", "w"), indent=1, default=str)
    print(f"\nruns aggregated: {sorted(runs)}")


if __name__ == "__main__":
    main()
