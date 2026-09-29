#!/usr/bin/env python3
"""Audit spectral leakage in the old and metric-matched punctured pairs.

The operators intentionally match the information available to the repository's
spectral probe: the 1-skeleton combinatorial and normalized Laplacians, the
unsigned rank-1 incidence Laplacian, and the colored Hasse-graph Laplacian.
Full spectra are saved in NPZ form and pair/operator comparisons in CSV/JSON.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "data").is_dir() and (
            candidate / "experiments" / "orientability"
        ).is_dir():
            return candidate
    raise RuntimeError(f"could not locate Beyond_Homology above {start}")


REPO_ROOT = find_repo_root(HERE)
for path in (REPO_ROOT, HERE.parent, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from baselines.spec_mlp.model import _dense_from_edge_index, _incidence_dense


PAIR_RE = re.compile(r"h(\d+)_p(\d+)_first\.pt$")
OPERATORS = ("graph_L", "graph_Lnorm", "unsigned_hodge1", "hasse_Lnorm")


def parse_args() -> argparse.Namespace:
    data_root = REPO_ROOT / "data"
    parser = argparse.ArgumentParser(
        description="Compare Laplacian spectra of old and metric-matched punctured pairs."
    )
    parser.add_argument(
        "--old-dir",
        type=Path,
        default=data_root / "punc_torus_vs_punc_klein",
    )
    parser.add_argument(
        "--metric-dir",
        type=Path,
        default=data_root / "punc_torus_vs_punc_klein_metric_matched",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE,
    )
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--rtol", type=float, default=1e-7)
    return parser.parse_args()


def load_complex(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def adjacency_laplacians(data) -> tuple[np.ndarray, np.ndarray]:
    n0 = int(data.x_0.size(0))
    adjacency = _dense_from_edge_index(
        getattr(data, "edge_index_adjacency_0_1", None), n0
    )
    degree = adjacency.sum(axis=1)
    laplacian = np.diag(degree) - adjacency
    inv_sqrt = np.divide(
        1.0,
        np.sqrt(degree),
        out=np.zeros_like(degree),
        where=degree > 0,
    )
    normalized = np.eye(n0) - inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]
    normalized[degree == 0, degree == 0] = 0.0
    return laplacian, normalized


def unsigned_hodge1(data) -> np.ndarray:
    n0 = int(data.x_0.size(0))
    n1 = int(data.x_1.size(0))
    n2 = int(data.x_2.size(0)) if getattr(data, "x_2", None) is not None else 0
    b1 = _incidence_dense(data.edge_index_incidence_0_1, n0, n1)
    b2 = _incidence_dense(data.edge_index_incidence_1_2, n1, n2)
    return b1.T @ b1 + b2 @ b2.T


def hasse_normalized_laplacian(data) -> np.ndarray:
    counts = [int(data.x_0.size(0)), int(data.x_1.size(0))]
    counts.append(int(data.x_2.size(0)) if getattr(data, "x_2", None) is not None else 0)
    offsets = np.cumsum([0, *counts])
    adjacency = np.zeros((sum(counts), sum(counts)), dtype=np.float64)
    for low_rank, high_rank in ((0, 1), (1, 2)):
        edge_index = getattr(data, f"edge_index_incidence_{low_rank}_{high_rank}", None)
        if edge_index is None or edge_index.numel() == 0:
            continue
        edges = edge_index.detach().cpu().numpy()
        low = edges[0] + offsets[low_rank]
        high = edges[1] + offsets[high_rank]
        adjacency[low, high] = 1.0
        adjacency[high, low] = 1.0
    degree = adjacency.sum(axis=1)
    inv_sqrt = np.divide(
        1.0,
        np.sqrt(degree),
        out=np.zeros_like(degree),
        where=degree > 0,
    )
    result = np.eye(adjacency.shape[0]) - inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]
    result[degree == 0, degree == 0] = 0.0
    return result


def spectra(data) -> dict[str, np.ndarray]:
    graph_l, graph_lnorm = adjacency_laplacians(data)
    matrices = {
        "graph_L": graph_l,
        "graph_Lnorm": graph_lnorm,
        "unsigned_hodge1": unsigned_hodge1(data),
        "hasse_Lnorm": hasse_normalized_laplacian(data),
    }
    return {
        name: np.linalg.eigvalsh((matrix + matrix.T) / 2.0)
        for name, matrix in matrices.items()
    }


def list_pairs(directory: Path) -> list[tuple[int, int, Path, Path]]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Pair directory does not exist: {directory}")
    pairs = []
    for first in directory.glob("h*_p*_first.pt"):
        match = PAIR_RE.fullmatch(first.name)
        if not match:
            continue
        h, p = map(int, match.groups())
        second = directory / f"h{h}_p{p}_second.pt"
        if not second.is_file():
            raise FileNotFoundError(f"Missing pair member: {second}")
        pairs.append((h, p, first, second))
    if not pairs:
        raise RuntimeError(f"No saved pairs found in {directory}")
    return sorted(pairs)


def audit_dataset(
    label: str,
    directory: Path,
    atol: float,
    rtol: float,
    archive: dict[str, np.ndarray],
) -> tuple[list[dict], dict]:
    rows = []
    leak_counts = {operator: 0 for operator in OPERATORS}
    pairs = list_pairs(directory)
    for pair_index, (h, p, first_path, second_path) in enumerate(pairs):
        first_spectra = spectra(load_complex(first_path))
        second_spectra = spectra(load_complex(second_path))
        for operator in OPERATORS:
            first = first_spectra[operator]
            second = second_spectra[operator]
            same_shape = first.shape == second.shape
            matched = same_shape and np.allclose(first, second, atol=atol, rtol=rtol)
            max_abs = float(np.max(np.abs(first - second))) if same_shape else None
            scale = max(float(np.max(np.abs(first))), float(np.max(np.abs(second))), atol)
            max_rel = (max_abs / scale) if max_abs is not None else None
            leak_counts[operator] += int(not matched)
            rows.append(
                {
                    "dataset": label,
                    "pair_index": pair_index,
                    "h": h,
                    "p": p,
                    "operator": operator,
                    "dimension_first": len(first),
                    "dimension_second": len(second),
                    "spectra_match": matched,
                    "max_abs_difference": max_abs,
                    "max_relative_difference": max_rel,
                }
            )
            prefix = f"{label}_h{h}_p{p}_{operator}"
            archive[f"{prefix}_first"] = first
            archive[f"{prefix}_second"] = second
        print(f"[{label}] {pair_index + 1}/{len(pairs)} h={h} p={p}", flush=True)
    summary = {
        "directory": str(directory.resolve()),
        "pair_count": len(pairs),
        "leaking_pairs_by_operator": leak_counts,
        "all_operators_cospectral_pairs": sum(
            all(row["spectra_match"] for row in rows if row["pair_index"] == index)
            for index in range(len(pairs))
        ),
        "pairs_with_any_spectral_leak": sum(
            any(not row["spectra_match"] for row in rows if row["pair_index"] == index)
            for index in range(len(pairs))
        ),
    }
    return rows, summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive: dict[str, np.ndarray] = {}
    all_rows = []
    summaries = {}
    for label, directory in (("old", args.old_dir), ("metric_matched", args.metric_dir)):
        rows, summary = audit_dataset(label, directory, args.atol, args.rtol, archive)
        all_rows.extend(rows)
        summaries[label] = summary

    csv_path = args.output_dir / "pairwise_spectral_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    summary = {
        "method": {
            "absolute_tolerance": args.atol,
            "relative_tolerance": args.rtol,
            "operators": list(OPERATORS),
            "note": "unsigned_hodge1 matches the unsigned incidence available to the model",
        },
        "datasets": summaries,
    }
    summary_path = args.output_dir / "spectral_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(args.output_dir / "full_spectra.npz", **archive)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {csv_path}, {summary_path}, and full_spectra.npz")


if __name__ == "__main__":
    main()
