"""Validate the topology of the torus-vs-sphere control pair."""

from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parent
NEW_EXPERIMENTS_ROOT = PACKAGE_ROOT.parent
if str(NEW_EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(NEW_EXPERIMENTS_ROOT))

from orientability.orient_common import (  # noqa: E402
    betti_numbers_cw,
    build_closed_grid,
    degree_profile,
    edge_face_incidence_profile,
    wl_stabilized_check,
)


def icosahedron_cells():
    faces = [
        (0, 1, 8), (0, 8, 4), (0, 4, 5), (0, 5, 9), (0, 9, 1),
        (1, 6, 8), (8, 6, 10), (8, 10, 4), (4, 10, 2), (4, 2, 5),
        (5, 2, 11), (5, 11, 9), (9, 11, 7), (9, 7, 1), (1, 7, 6),
        (3, 6, 7), (3, 10, 6), (3, 2, 10), (3, 11, 2), (3, 7, 11),
    ]
    edges = set()
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (a, c)):
            edges.add((min(u, v), max(u, v)))
    return list(range(12)), sorted(edges), [list(face) for face in faces]


def subdivide_triangular(vertices, edges, faces):
    edge_midpoints = {
        edge: index
        for index, edge in enumerate(edges, start=max(vertices) + 1)
    }
    new_vertices = list(vertices) + list(edge_midpoints.values())
    new_faces = []
    new_edges = set()
    for a, b, c in faces:
        ab = edge_midpoints[(min(a, b), max(a, b))]
        bc = edge_midpoints[(min(b, c), max(b, c))]
        ac = edge_midpoints[(min(a, c), max(a, c))]
        triangles = ((a, ab, ac), (b, ab, bc), (c, ac, bc), (ab, bc, ac))
        for triangle in triangles:
            new_faces.append(list(triangle))
            for u, v in (
                (triangle[0], triangle[1]),
                (triangle[1], triangle[2]),
                (triangle[0], triangle[2]),
            ):
                new_edges.add((min(u, v), max(u, v)))
    return new_vertices, sorted(new_edges), new_faces


def build_icosphere(subdivision_level: int = 1):
    vertices, edges, faces = icosahedron_cells()
    for _ in range(subdivision_level):
        vertices, edges, faces = subdivide_triangular(vertices, edges, faces)
    return edges, faces


def run_checks() -> list[str]:
    failures = []

    def check(name, condition, detail=""):
        status = "PASS" if condition else "FAIL"
        if not condition:
            failures.append(name)
        suffix = f" -- {detail}" if detail else ""
        print(f"[{status}] {name}{suffix}")

    torus_edges, torus_faces = build_closed_grid(6, 6, twisted=False)
    sphere_edges, sphere_faces = build_icosphere(subdivision_level=1)
    torus_vertices = {vertex for edge in torus_edges for vertex in edge}
    sphere_vertices = {vertex for edge in sphere_edges for vertex in edge}

    torus_chi = len(torus_vertices) - len(torus_edges) + len(torus_faces)
    sphere_chi = len(sphere_vertices) - len(sphere_edges) + len(sphere_faces)
    check("torus Euler characteristic is 0", torus_chi == 0)
    check("sphere Euler characteristic is 2", sphere_chi == 2)
    check("Euler characteristics differ", torus_chi != sphere_chi)

    torus_betti = betti_numbers_cw(torus_edges, torus_faces)
    sphere_betti = betti_numbers_cw(sphere_edges, sphere_faces)
    check("torus Betti numbers are (1, 2, 1)", torus_betti == (1, 2, 1))
    check("sphere Betti numbers are (1, 0, 1)", sphere_betti == (1, 0, 1))
    check("Betti numbers differ", torus_betti != sphere_betti)

    check(
        "degree profiles differ",
        degree_profile(torus_edges) != degree_profile(sphere_edges),
    )
    torus_incidence = edge_face_incidence_profile(torus_edges, torus_faces)
    sphere_incidence = edge_face_incidence_profile(sphere_edges, sphere_faces)
    closed_surfaces = (
        torus_incidence == sphere_incidence == {2: len(torus_edges)}
        or set(torus_incidence) == set(sphere_incidence) == {2}
    )
    check("both complexes are closed surfaces", closed_surfaces)

    message, indistinguishable = wl_stabilized_check(torus_edges, sphere_edges, 30)
    check("1-WL separates the complexes", not indistinguishable, message)
    return failures


def main() -> None:
    failures = run_checks()
    if failures:
        raise SystemExit(f"{len(failures)} check(s) failed: {failures}")
    print("All torus-vs-sphere topology checks passed.")


if __name__ == "__main__":
    main()
