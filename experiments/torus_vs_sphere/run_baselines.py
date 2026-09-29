"""Run the baseline models on the torus-vs-sphere control experiment.

Usage:
    python run_baselines.py
    python run_baselines.py --copies 8 --epochs 10 --mlp-epochs 40
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parent
NEW_EXPERIMENTS_DIR = PACKAGE_ROOT.parent
SMCN_ROOT = NEW_EXPERIMENTS_DIR.parent
for _path in (SMCN_ROOT.parent, SMCN_ROOT, NEW_EXPERIMENTS_DIR, PACKAGE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

try:
    from benchmark_figures import plot_torus_sphere_summary  # noqa: E402
except ModuleNotFoundError:  # noqa: E402
    plot_torus_sphere_summary = None

from common import (  # noqa: E402
    RunnerConfig,
    prepare_splits,
    train_graph_model,
    train_probe_mlp,
)
from baselines.cin.model import build_homp_model  # noqa: E402
from baselines.esan.model import build_esan_model  # noqa: E402
from baselines.gin.model import GINClassifier  # noqa: E402
from baselines.hasse.model import HasseGINClassifier  # noqa: E402

PASS_THRESHOLD = 0.99
EXPECTED_CHANCE = ("boundary_mlp",)

HOMP_REQUIRED_FIELDS = (
    "x_0",
    "x_1",
    "x_2",
    "num_cells_0",
    "num_cells_1",
    "num_cells_2",
    "edge_index_incidence_0_1",
    "edge_index_incidence_0_2",
    "edge_index_incidence_1_2",
    "edge_index_adjacency_0_1",
    "edge_index_adjacency_1_2",
    "bridge_index_adjacency_0_1",
    "bridge_index_adjacency_1_2",
    "edge_index_coadjacency_1_0",
    "edge_index_coadjacency_2_1",
    "bridge_index_coadjacency_1_0",
    "bridge_index_coadjacency_2_1",
)


def build_gin(config: RunnerConfig, _device):
    return GINClassifier(
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_classes=2,
    )


def build_hasse(config: RunnerConfig, _device):
    return HasseGINClassifier(
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_classes=2,
    )


def build_homp_cin(config: RunnerConfig, device):
    return build_homp_model(
        number_of_layers=config.num_layers,
        embedding_dim=config.hidden_dim,
        device=str(device),
        num_classes=2,
        output_ranks=[0, 2],
    )


def build_esan(config: RunnerConfig, device):
    return build_esan_model(
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_classes=2,
        device=str(device),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all non-SMCN baselines on E0."
    )
    parser.add_argument("--copies", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--mlp-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> RunnerConfig:
    config = RunnerConfig()
    overrides = {
        key: value
        for key, value in (
            ("copies", args.copies),
            ("epochs", args.epochs),
            ("mlp_epochs", args.mlp_epochs),
            ("batch_size", args.batch_size),
            ("seed", args.seed),
        )
        if value is not None
    }
    return replace(config, **overrides) if overrides else config


def format_summary_text(results: dict[str, dict]) -> str:
    lines = ["PARAMETER TABLE", "| Model | Parameters |", "|---|---:|"]
    for name, metrics in results.items():
        lines.append(f"| {name} | {metrics['parameter_count']} |")

    lines.extend(
        [
            "",
            "PERFORMANCE TABLE",
            "| Model | Train Acc | Val Acc | Test Acc | E0 expectation | Verdict |",
            "|---|---:|---:|---:|---|---|",
        ]
    )
    for name, metrics in results.items():
        test_acc = metrics["test"]["accuracy"]
        if name in EXPECTED_CHANCE:
            expectation = "~50% (closed surfaces have no boundary signal)"
            verdict = (
                "as expected"
                if test_acc < PASS_THRESHOLD
                else "SUSPICIOUS (leak?)"
            )
        else:
            expectation = ">= 99%"
            verdict = "PASS" if test_acc >= PASS_THRESHOLD else "FAIL"
        lines.append(
            f"| {name} | {metrics['train']['accuracy']:.4f} | "
            f"{metrics['val']['accuracy']:.4f} | {test_acc:.4f} | "
            f"{expectation} | {verdict} |"
        )
    return "\n".join(lines)


def all_required_models_pass(results: dict[str, dict]) -> bool:
    return all(
        metrics["test"]["accuracy"] >= PASS_THRESHOLD
        for name, metrics in results.items()
        if name not in EXPECTED_CHANCE
    )


def save_run_summary(
    results: dict[str, dict],
    config: RunnerConfig,
    verdict: str,
) -> Path:
    now = datetime.now()
    timestamp_filename = now.strftime("%Y%m%d_%H%M%S")
    timestamp_label = now.strftime("%d-%m %H:%M")
    log_dir = PACKAGE_ROOT / "run_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"summary_{timestamp_filename}.log"

    config_dict = asdict(config)
    config_dict["timestamp"] = timestamp_label
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"RUN TIMESTAMP: {timestamp_label}\n")
        handle.write(f"VERDICT: {verdict}\n")
        handle.write("\nRUN PARAMETERS:\n")
        handle.write(json.dumps(config_dict, indent=2, sort_keys=True))
        handle.write("\n\n")
        handle.write(format_summary_text(results))
        handle.write("\n")
    return log_path


def main() -> None:
    config = make_config(parse_args())
    print(
        f"config: copies={config.copies} epochs={config.epochs} "
        f"mlp_epochs={config.mlp_epochs} batch_size={config.batch_size} "
        f"seed={config.seed}"
    )

    splits = prepare_splits(config)
    results: dict[str, dict] = {}

    for probe in ("count_mlp", "dist_mlp", "boundary_mlp"):
        results[probe] = train_probe_mlp(
            model_name=probe,
            splits=splits,
            config=config,
        )

    results["gin"] = train_graph_model(
        model_name="gin",
        model_factory=build_gin,
        follow_batch=["x_0"],
        required_fields=("x_0", "edge_index_adjacency_0_1"),
        splits=splits,
        config=config,
    )
    results["esan"] = train_graph_model(
        model_name="esan",
        model_factory=build_esan,
        follow_batch=["x_0"],
        required_fields=("x_0", "edge_index_adjacency_0_1"),
        splits=splits,
        config=config,
    )
    results["hasse"] = train_graph_model(
        model_name="hasse",
        model_factory=build_hasse,
        follow_batch=["x_0", "x_1", "x_2"],
        required_fields=(
            "x_0",
            "x_1",
            "x_2",
            "edge_index_incidence_0_1",
            "edge_index_incidence_1_2",
        ),
        splits=splits,
        config=config,
    )
    results["homp_cin"] = train_graph_model(
        model_name="homp_cin",
        model_factory=build_homp_cin,
        follow_batch=["x_0", "x_1", "x_2"],
        required_fields=HOMP_REQUIRED_FIELDS,
        splits=splits,
        config=config,
    )

    summary_text = format_summary_text(results)
    print(f"\n{summary_text}")
    passed = all_required_models_pass(results)
    verdict = (
        "PASS -- required baselines met the accuracy threshold"
        if passed
        else "FAIL -- one or more required baselines missed the accuracy threshold"
    )
    print(f"\nE0 VERDICT: {verdict}")

    log_path = save_run_summary(results, config, verdict)
    print(f"\nSaved run summary log to: {log_path}")
    if plot_torus_sphere_summary is not None:
        figure_tag = log_path.stem.replace("summary_", "")
        figure_path = plot_torus_sphere_summary(
            results,
            PACKAGE_ROOT / "runs" / "figures",
            figure_tag,
        )
        print(f"Saved benchmark figure to: {figure_path}")


if __name__ == "__main__":
    main()
