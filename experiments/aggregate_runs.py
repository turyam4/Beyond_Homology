"""Aggregate tight-FG metrics for SMCN, HOMP, and all baseline models.

Supported directory layouts:

* current: ``runs/<model>/<DDMM_HH:MM>/seed<k>/metrics.json``;
* previous: ``runs/<model>/seed<k>_<DDMM_HH:MM>/metrics.json``;
* legacy:  ``runs/tight_<model>/seed<k>/metrics.json``.

Usage:
    python aggregate_runs.py
    python aggregate_runs.py /path/to/runs
    python aggregate_runs.py runs gin mlp homp
"""
from __future__ import annotations

import json
import argparse
import re
import sys
from pathlib import Path
from statistics import mean, pstdev

MODEL_ALIASES = {
    "smcn": ("smcn", "subcomplex", "tight_subcomplex", "tight_smcn"),
    "mlp": ("mlp",),
    "dist_mlp": ("dist_mlp", "dist", "distance_mlp"),
    "boundary_mlp": ("boundary_mlp", "boundary", "boundary_loop_mlp"),
    "gin": ("gin",),
    "homp": ("homp", "homp_cin", "tight_homp"),
    "hasse": ("hasse", "hasse_gin"),
    "esan": ("esan",),
}
MODEL_LABELS = {
    "smcn": "SMCN (subcomplex)",
    "mlp": "Count MLP",
    "dist_mlp": "Distance Histogram MLP",
    "boundary_mlp": "Boundary Loop MLP",
    "gin": "GIN (1-skeleton)",
    "homp": "HOMP/CIN",
    "hasse": "Hasse GIN",
    "esan": "ESAN",
}


def _seed_number(name: str) -> int:
    match = re.match(r"seed[_-]?(\d+)", name)
    return int(match.group(1)) if match else 10**9


def _model_directories(runs_root: Path, model: str) -> list[Path]:
    directories = []
    for alias in MODEL_ALIASES[model]:
        candidate = runs_root / alias
        if candidate.is_dir() and candidate not in directories:
            directories.append(candidate)
    return directories


def _timestamp_tag(directory_name: str) -> str | None:
    match = re.match(r"seed[_-]?\d+_(\d{4}_\d{2}:\d{2})$", directory_name)
    return match.group(1) if match else None


def _path_timestamp_tag(metrics_path: Path) -> str | None:
    parent = metrics_path.parent
    if re.fullmatch(r"\d{4}_\d{2}:\d{2}", parent.parent.name):
        return parent.parent.name
    return _timestamp_tag(parent.name)


def load(runs_root: Path, model: str, run_tag: str | None = None) -> list[dict]:
    rows = []
    seen_metrics = set()
    metric_paths = []
    for model_dir in _model_directories(runs_root, model):
        for metrics_path in list(model_dir.glob("seed*/metrics.json")) + list(
            model_dir.glob("[0-9][0-9][0-9][0-9]_[0-9][0-9]:[0-9][0-9]/seed*/metrics.json")
        ):
            resolved = metrics_path.resolve()
            if resolved in seen_metrics:
                continue
            seen_metrics.add(resolved)
            metric_paths.append(metrics_path)

    if run_tag:
        metric_paths = [
            path
            for path in metric_paths
            if _path_timestamp_tag(path) == run_tag
        ]
    else:
        timestamped = [
            path for path in metric_paths if _path_timestamp_tag(path)
        ]
        if timestamped:
            # Directory timestamps omit the year, so filesystem modification
            # time is the reliable way to identify the newest launch cohort.
            newest_path = max(timestamped, key=lambda path: path.stat().st_mtime)
            newest_tag = _path_timestamp_tag(newest_path)
            metric_paths = [
                path
                for path in timestamped
                if _path_timestamp_tag(path) == newest_tag
            ]

    for metrics_path in metric_paths:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        config_path = metrics_path.parent / "config.json"
        config = (
            json.loads(config_path.read_text(encoding="utf-8"))
            if config_path.is_file()
            else {}
        )
        total = int(metrics.get("brec_solved_total", 0))
        if total <= 0:
            total = int(metrics.get("target_total", 0))
        denominator = max(total, 1)
        rows.append(
            {
                "run": metrics_path.parent.name,
                "seed_number": _seed_number(metrics_path.parent.name),
                "target": int(metrics.get("target_success_count", 0)),
                "reliability": int(metrics.get("reliability_pass_count", 0)),
                "brec": int(metrics.get("brec_solved_count", 0)),
                "total": total,
                "brec_frac": int(metrics.get("brec_solved_count", 0))
                / denominator,
                "mean_t2_target": float(metrics.get("mean_t2_target", 0.0)),
                "config": config,
            }
        )
    return sorted(rows, key=lambda row: (row["seed_number"], row["run"]))


def summarize(model: str, rows: list[dict]) -> None:
    label = MODEL_LABELS[model]
    if not rows:
        print(f"[{label}] no metrics.json found")
        return

    fractions = [row["brec_frac"] for row in rows]
    print(f"\n=== {label} ({len(rows)} runs) ===")
    print(
        f"{'run':>22} {'brec':>6} {'target':>7} {'reliab':>7} "
        f"{'total':>6} {'mean_t2_tgt':>12}"
    )
    for row in rows:
        print(
            f"{row['run']:>22} {row['brec']:>6} {row['target']:>7} "
            f"{row['reliability']:>7} {row['total']:>6} "
            f"{row['mean_t2_target']:>12.4f}"
        )
    print(
        f"BREC-solved fraction: mean={mean(fractions):.3f} "
        f"std={pstdev(fractions):.3f}"
    )
    print(
        f"FINAL BREC-solved: {mean(fractions):.3f} "
        f"+/- {pstdev(fractions):.3f} across {len(rows)} runs"
    )
    print_configuration(rows)


def _common_value(configs: list[dict], key: str):
    values = [config.get(key) for config in configs]
    first = values[0]
    return first if all(value == first for value in values) else values


def print_configuration(rows: list[dict]) -> None:
    configs = [row["config"] for row in rows if row["config"]]
    if not configs:
        print("Configuration: no config.json files found")
        return

    baseline_models = [
        model
        for model in (config.get("baseline_model", {}) for config in configs)
        if isinstance(model, dict)
    ]
    print("\nFinal resolved configuration:")
    print(f"  model_seeds: {[config.get('model_seed') for config in configs]}")
    for key in ("data_seed", "permutation_seed", "order_seed"):
        print(f"  {key}: {_common_value(configs, key)}")
    for key in (
        "num_epochs",
        "number_of_permuted_copies",
        "batch_size",
        "eval_batch_size",
        "learning_rate",
        "weight_decay",
        "grad_clip",
        "margin",
        "threshold",
        "samples_per_g",
    ):
        print(f"  {key}: {_common_value(configs, key)}")
    for key in (
        "hidden_dim",
        "num_layers",
        "dropout",
        "number_of_mlp_layers",
        "include_rank2_updates",
        "output_ranks",
        "max_dist",
        "high_rank",
        "num_hops",
        "chunk_size",
    ):
        values = [model.get(key) for model in baseline_models if key in model]
        if values:
            value = values[0] if all(item == values[0] for item in values) else values
            print(f"  model.{key}: {value}")
    if not any(baseline_models):
        for key in (
            "model_type",
            "embedding_dim",
            "number_of_mlp_layers",
            "number_cin_layers_for_subcomplex",
            "number_subgraph_layers",
            "dropout",
            "in_dropout",
            "activation",
            "output_ranks",
        ):
            print(f"  model.{key}: {_common_value(configs, key)}")


def print_comparison(all_rows: dict[str, list[dict]]) -> None:
    available = {model: rows for model, rows in all_rows.items() if rows}
    if not available:
        return
    print("\n=== Mean BREC-solved comparison ===")
    print(f"{'model':<22} {'runs':>6} {'mean':>9} {'std':>9}")
    for model, rows in available.items():
        fractions = [row["brec_frac"] for row in rows]
        print(
            f"{MODEL_LABELS[model]:<22} {len(rows):>6} "
            f"{mean(fractions):>9.3f} {pstdev(fractions):>9.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_root", nargs="?", default="runs")
    parser.add_argument("models", nargs="*")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()
    runs_root = Path(args.runs_root)
    requested = [name.lower() for name in args.models]
    models = requested or list(MODEL_ALIASES)
    unknown = [model for model in models if model not in MODEL_ALIASES]
    if unknown:
        raise SystemExit(
            f"Unknown models {unknown}; choose from {list(MODEL_ALIASES)}"
        )

    all_rows = {
        model: load(runs_root, model, run_tag=args.tag) for model in models
    }
    for model, rows in all_rows.items():
        summarize(model, rows)
    if len(models) > 1:
        print_comparison(all_rows)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        try:
            sys.stdout.close()
        except BrokenPipeError:
            pass
