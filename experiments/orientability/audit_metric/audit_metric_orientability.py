#!/usr/bin/env python3
"""Directly certify orientability of the saved metric-matched pair complexes.

For each saved complex, reconstruct oriented face-boundary traversals from the
0--1 and 1--2 incidence relations. Adjacent faces must induce opposite
directions on every shared edge. Orientability is exactly the satisfiability of
these +/-1 constraints; a contradiction is a concrete non-orientability
witness. Boundary edges impose no constraint.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict, deque
from pathlib import Path

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
DATA_ROOT = REPO_ROOT / "data"
for path in (REPO_ROOT, DATA_ROOT, HERE.parent, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

PAIR_RE = re.compile(r"h(\d+)_p(\d+)_first\.pt$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct face-orientation audit for metric-matched punctured pairs."
    )
    parser.add_argument(
        "--pair-dir",
        type=Path,
        default=DATA_ROOT / "punc_torus_vs_punc_klein_metric_matched",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE,
    )
    return parser.parse_args()


def load_complex(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def incidence_pairs(data, name: str) -> list[tuple[int, int]]:
    index = getattr(data, name, None)
    if index is None:
        raise ValueError(f"saved complex lacks {name}")
    return [tuple(map(int, pair)) for pair in index.detach().cpu().t().tolist()]


def reconstruct_cells(data):
    n0, n1, n2 = int(data.x_0.size(0)), int(data.x_1.size(0)), int(data.x_2.size(0))
    edge_vertices: list[set[int]] = [set() for _ in range(n1)]
    for vertex, edge in incidence_pairs(data, "edge_index_incidence_0_1"):
        edge_vertices[edge].add(vertex)
    malformed_edges = [edge for edge, vertices in enumerate(edge_vertices) if len(vertices) != 2]
    if malformed_edges:
        raise ValueError(f"edges without exactly two endpoints: {malformed_edges[:10]}")

    edges = [tuple(sorted(vertices)) for vertices in edge_vertices]
    face_edges: list[set[int]] = [set() for _ in range(n2)]
    edge_faces: list[set[int]] = [set() for _ in range(n1)]
    for edge, face in incidence_pairs(data, "edge_index_incidence_1_2"):
        face_edges[face].add(edge)
        edge_faces[edge].add(face)

    nonmanifold = [edge for edge, faces in enumerate(edge_faces) if len(faces) not in (1, 2)]
    if nonmanifold:
        raise ValueError(f"edges without one or two incident faces: {nonmanifold[:10]}")
    return n0, edges, face_edges, edge_faces


def face_edge_directions(
    face_index: int,
    incident_edges: set[int],
    edges: list[tuple[int, int]],
) -> dict[int, int]:
    """Return each boundary edge direction relative to its sorted orientation."""
    vertex_edges: dict[int, list[int]] = defaultdict(list)
    for edge in incident_edges:
        u, v = edges[edge]
        vertex_edges[u].append(edge)
        vertex_edges[v].append(edge)
    if not incident_edges or any(len(items) != 2 for items in vertex_edges.values()):
        raise ValueError(f"face {face_index} does not have a cycle boundary")

    start = min(vertex_edges)
    current = start
    previous_edge = None
    directions: dict[int, int] = {}
    for _ in range(len(incident_edges)):
        choices = sorted(edge for edge in vertex_edges[current] if edge != previous_edge)
        if not choices:
            raise ValueError(f"face {face_index} boundary traversal stopped early")
        edge = choices[0]
        if edge in directions:
            raise ValueError(f"face {face_index} repeats an edge before closing")
        u, v = edges[edge]
        nxt = v if current == u else u
        directions[edge] = 1 if (current, nxt) == (u, v) else -1
        previous_edge, current = edge, nxt
    if current != start or set(directions) != incident_edges:
        raise ValueError(f"face {face_index} boundary is disconnected or not a cycle")
    return directions


def contradiction_cycle(parent: list[int | None], first: int, second: int) -> list[int]:
    first_path, cursor = [], first
    while cursor is not None:
        first_path.append(cursor)
        cursor = parent[cursor]
    second_path, cursor = [], second
    while cursor is not None:
        second_path.append(cursor)
        cursor = parent[cursor]
    first_position = {face: index for index, face in enumerate(first_path)}
    common = next(face for face in second_path if face in first_position)
    return first_path[: first_position[common] + 1] + list(
        reversed(second_path[: second_path.index(common)])
    )


def verify_orientability(data) -> dict:
    n0, edges, face_edges, edge_faces = reconstruct_cells(data)
    directions = [
        face_edge_directions(face, incident, edges)
        for face, incident in enumerate(face_edges)
    ]
    constraints: list[list[tuple[int, int, int]]] = [[] for _ in face_edges]
    boundary_edges = 0
    for edge, faces in enumerate(edge_faces):
        if len(faces) == 1:
            boundary_edges += 1
            continue
        first, second = sorted(faces)
        # Flipping a face multiplies all its induced edge directions by -1.
        # Shared directions must be opposite in a coherent global orientation.
        factor = -directions[first][edge] * directions[second][edge]
        constraints[first].append((second, factor, edge))
        constraints[second].append((first, factor, edge))

    signs: list[int | None] = [None] * len(face_edges)
    parent: list[int | None] = [None] * len(face_edges)
    components = 0
    for root in range(len(face_edges)):
        if signs[root] is not None:
            continue
        components += 1
        signs[root] = 1
        queue = deque([root])
        while queue:
            face = queue.popleft()
            for neighbor, factor, edge in constraints[face]:
                required = signs[face] * factor
                if signs[neighbor] is None:
                    signs[neighbor] = required
                    parent[neighbor] = face
                    queue.append(neighbor)
                elif signs[neighbor] != required:
                    return {
                        "orientable": False,
                        "vertices": n0,
                        "edges": len(edges),
                        "faces": len(face_edges),
                        "boundary_edges": boundary_edges,
                        "dual_components": components,
                        "contradiction_edge": edge,
                        "contradiction_faces": [face, neighbor],
                        "orientation_reversing_dual_cycle": contradiction_cycle(
                            parent, face, neighbor
                        ),
                    }
    return {
        "orientable": True,
        "vertices": n0,
        "edges": len(edges),
        "faces": len(face_edges),
        "boundary_edges": boundary_edges,
        "dual_components": components,
        "contradiction_edge": None,
        "contradiction_faces": None,
        "orientation_reversing_dual_cycle": None,
    }


def list_pairs(directory: Path):
    pairs = []
    for first in directory.glob("h*_p*_first.pt"):
        match = PAIR_RE.fullmatch(first.name)
        if match:
            h, p = map(int, match.groups())
            second = directory / f"h{h}_p{p}_second.pt"
            if not second.is_file():
                raise FileNotFoundError(f"missing pair member: {second}")
            pairs.append((h, p, first, second))
    if not pairs:
        raise RuntimeError(f"no pairs found in {directory}")
    return sorted(pairs)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, witnesses = [], []
    pairs = list_pairs(args.pair_dir)
    for pair_index, (h, p, first_path, second_path) in enumerate(pairs):
        results = {
            "first": verify_orientability(load_complex(first_path)),
            "second": verify_orientability(load_complex(second_path)),
        }
        passed = results["first"]["orientable"] and not results["second"]["orientable"]
        for member, result in results.items():
            rows.append(
                {
                    "pair_index": pair_index,
                    "h": h,
                    "p": p,
                    "member": member,
                    "expected_orientable": member == "first",
                    **{key: value for key, value in result.items() if "cycle" not in key},
                }
            )
            if not result["orientable"]:
                witnesses.append(
                    {"pair_index": pair_index, "h": h, "p": p, "member": member, **result}
                )
        print(
            f"{pair_index + 1}/{len(pairs)} h={h} p={p} "
            f"first={results['first']['orientable']} "
            f"second={results['second']['orientable']} passed={passed}",
            flush=True,
        )
        if not passed:
            raise AssertionError(f"direct orientability expectation failed at h={h}, p={p}")

    csv_path = args.output_dir / "pairwise_direct_orientability.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    witness_path = args.output_dir / "nonorientability_witnesses.jsonl"
    witness_path.write_text(
        "".join(json.dumps(witness, sort_keys=True) + "\n" for witness in witnesses),
        encoding="utf-8",
    )
    summary = {
        "pair_directory": str(args.pair_dir.resolve()),
        "pair_count": len(pairs),
        "first_orientable": sum(row["orientable"] for row in rows if row["member"] == "first"),
        "second_nonorientable": sum(not row["orientable"] for row in rows if row["member"] == "second"),
        "all_pairs_verified": all(
            row["orientable"] == row["expected_orientable"] for row in rows
        ),
        "method": "face-orientation parity propagation across shared edges",
    }
    summary_path = args.output_dir / "direct_orientability_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {csv_path}, {witness_path}, and {summary_path}")


if __name__ == "__main__":
    main()
