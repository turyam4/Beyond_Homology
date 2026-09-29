#!/usr/bin/env python3
"""Generate clean orientability BREC summary figures.

Reads the latest four seed runs per model from:
  <family>/runs/<model>/seed*/metrics.json

Writes one PNG per family into test_outputs/.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "test_outputs" / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = ROOT / "test_outputs"

FAMILIES = {
    "cylinder_vs_mobius": "Cylinder vs Möbius",
    "torus_vs_klein": "Torus vs Klein",
    "punc_torus_vs_punc_klein": "Punctured Torus vs Klein",
}

MODEL_ORDER = [
    "mlp",
    "boundary_mlp",
    "dist_mlp",
    "gin",
    "hasse",
    "homp",
    "esan",
    "smcn",
]

MODEL_LABELS = {
    "mlp": "MLP",
    "boundary_mlp": "Boundary\nMLP",
    "dist_mlp": "Dist\nMLP",
    "gin": "GIN",
    "hasse": "Hasse\nGIN",
    "homp": "HOMP",
    "esan": "ESAN",
    "smcn": "SMCN",
}


def latest_four_metrics(family: str, model: str) -> list[dict]:
    run_dir = ROOT / family / "runs" / model
    if not run_dir.is_dir():
        return []

    candidates = []
    for metrics_path in run_dir.glob("seed*/metrics.json"):
        seed_dir = metrics_path.parent.name
        if not re.match(r"seed\d+_", seed_dir):
            continue
        try:
            metrics = json.loads(metrics_path.read_text())
        except Exception:
            continue
        if "mean_t2_target" not in metrics or "mean_t2_reliability" not in metrics:
            continue
        candidates.append((metrics_path.stat().st_mtime, metrics_path, metrics))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return [metrics for _, _, metrics in candidates[:4]]


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def collect_family(family: str) -> list[dict]:
    rows = []
    for model in MODEL_ORDER:
        metrics_list = latest_four_metrics(family, model)
        if not metrics_list:
            continue

        target = [float(m["mean_t2_target"]) for m in metrics_list]
        reliability = [float(m["mean_t2_reliability"]) for m in metrics_list]
        margin = [t - r for t, r in zip(target, reliability)]
        solved = []
        for metrics in metrics_list:
            total = float(metrics.get("brec_solved_total") or metrics.get("target_total") or 0)
            solved.append(float(metrics.get("brec_solved_count", 0)) / total if total else math.nan)

        target_mean, target_std = mean_std(target)
        reliability_mean, reliability_std = mean_std(reliability)
        margin_mean, margin_std = mean_std(margin)
        solved_mean, solved_std = mean_std(solved)

        rows.append(
            {
                "model": model,
                "n": len(metrics_list),
                "target_mean": target_mean,
                "target_std": target_std,
                "reliability_mean": reliability_mean,
                "reliability_std": reliability_std,
                "margin_mean": margin_mean,
                "margin_std": margin_std,
                "solved_mean": solved_mean,
                "solved_std": solved_std,
            }
        )
    return rows


def positive_for_log(values: list[float], floor: float = 1e-8) -> list[float]:
    return [max(float(v), floor) for v in values]


def plot_family(family: str, title: str) -> Path:
    rows = collect_family(family)
    if not rows:
        raise RuntimeError(f"No metrics found for {family}")

    labels = [MODEL_LABELS.get(row["model"], row["model"]) for row in rows]
    x = np.arange(len(rows))

    solved = [row["solved_mean"] for row in rows]
    solved_err = [row["solved_std"] for row in rows]
    target = [row["target_mean"] for row in rows]
    target_err = [row["target_std"] for row in rows]
    reliability = [row["reliability_mean"] for row in rows]
    reliability_err = [row["reliability_std"] for row in rows]

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 160,
        }
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.5, 4.8),
        gridspec_kw={"width_ratios": [1.0, 1.35]},
        constrained_layout=True,
    )

    ax = axes[0]
    ax.bar(x, solved, yerr=solved_err, capsize=3, color="#4C78A8", edgecolor="black", linewidth=0.5)
    ax.set_title("BREC solved fraction")
    ax.set_ylabel("fraction solved")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.25)
    for idx, val in enumerate(solved):
        ax.text(idx, min(val + 0.035, 1.02), f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    ax = axes[1]
    width = 0.38
    target_plot = positive_for_log(target)
    reliability_plot = positive_for_log(reliability)
    target_err_plot = [min(e, max(v - 1e-8, 0.0)) for v, e in zip(target_plot, target_err)]
    reliability_err_plot = [min(e, max(v - 1e-8, 0.0)) for v, e in zip(reliability_plot, reliability_err)]

    ax.bar(
        x - width / 2,
        target_plot,
        width,
        yerr=target_err_plot,
        capsize=3,
        label="target T2",
        color="#F58518",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        x + width / 2,
        reliability_plot,
        width,
        yerr=reliability_err_plot,
        capsize=3,
        label="reliability T2",
        color="#54A24B",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_title("Target vs reliability separation")
    ax.set_ylabel("mean T2 score, log scale")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", which="both", alpha=0.25)
    ax.legend(frameon=False, loc="upper left")

    fig.suptitle(title, fontsize=16, fontweight="bold")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"{family}_brec_summary.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    for family, title in FAMILIES.items():
        out = plot_family(family, title)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
