"""Generate base-pair `.pt` files for the benchmark families.

Only base pairs are written.  No permutation-copy cache is built here; the
baseline evaluator creates copies pair-wise during evaluation and frees them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  
from omegaconf import OmegaConf  

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
ORIENTABILITY_DIR = EXPERIMENTS_DIR / "orientability"
ORIENTABILITY_CONFIG = ORIENTABILITY_DIR / "config.yaml"
FG_CONFIG = EXPERIMENTS_DIR / "fundamental_group" / "config.yaml"

for _p in (
    PROJECT_ROOT,
    EXPERIMENTS_DIR,
    ORIENTABILITY_DIR,
    ORIENTABILITY_DIR / "cylinder_vs_mobius",
    ORIENTABILITY_DIR / "torus_vs_klein",
    ORIENTABILITY_DIR / "punc_torus_vs_punc_klein",
    EXPERIMENTS_DIR / "fundamental_group",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

FAMILY_ALIASES = {
    "cyl_vs_mob": "cylinder_vs_mobius",
    "cylinder_vs_mobius": "cylinder_vs_mobius",
    "closed": "torus_vs_klein",
    "torus_vs_klein": "torus_vs_klein",
    "punc": "punc_torus_vs_punc_klein",
    "punctured": "punc_torus_vs_punc_klein",
    "punc_torus_vs_punc_klein": "punc_torus_vs_punc_klein",
    "fg": "free_vs_surface",
    "free_vs_surface": "free_vs_surface",
    "fundamental_group":"free_vs_surface"
}
ORIENTABILITY_FAMILIES = (
    "cylinder_vs_mobius",
    "torus_vs_klein",
    "punc_torus_vs_punc_klein",
)
MANIFEST_VERSION = 1


def _orient_imports():
    from data.complex_data import ComplexData
    from orient_common import build_closed_grid, build_bounded_grid
    from cyl_vs_mob import _to_ccc as _to_ccc_bounded
    from klein_torus_dataset import _to_ccc as _to_ccc_closed
    from punc_torus_vs_klein import _to_ccc as _to_ccc_punc

    return (
        ComplexData,
        build_closed_grid,
        build_bounded_grid,
        _to_ccc_bounded,
        _to_ccc_closed,
        _to_ccc_punc,
    )


def _fg_imports():
    from data.complex_data import ComplexData
    from fg_core import generate_words
    from fundamental_group_dataset import FundamentalGroupDataset

    return ComplexData, generate_words, FundamentalGroupDataset


def _load_config(path: Path, label: str):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label} config: {_display_path(path)}")
    return OmegaConf.load(path)


def _family_output_dir(name: str, cfg=None) -> Path:
    if cfg is not None and name in cfg.families:
        family_dir = ORIENTABILITY_DIR / str(cfg.families[name].family_dir)
        return (family_dir / str(cfg.families[name].base_pair_dir)).resolve()
    return SCRIPT_DIR / name


def _display_path(path: Path) -> str:
    path = Path(path)
    for base in (PROJECT_ROOT,):
        try:
            return str(path.resolve().relative_to(base.resolve()))
        except ValueError:
            pass
    return path.name


def _save_pair(family_dir: Path, name: str, first, second, overwrite: bool) -> bool:
    import torch

    family_dir.mkdir(parents=True, exist_ok=True)
    first_path = family_dir / f"{name}_first.pt"
    second_path = family_dir / f"{name}_second.pt"
    if not overwrite and first_path.is_file() and second_path.is_file():
        return False
    torch.save(first, first_path)
    torch.save(second, second_path)
    return True


def _existing_pairs(family_dir: Path) -> int:
    return len(list(family_dir.glob("*_first.pt"))) if family_dir.exists() else 0


def _manifest_path(family_dir: Path) -> Path:
    return family_dir / "manifest.json"


def _stable_payload(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _fingerprint(payload: dict) -> str:
    return hashlib.sha256(_stable_payload(payload).encode("utf-8")).hexdigest()


def _construction_fingerprint(payload: dict) -> str:
    payload = dict(payload)
    payload.pop("config_path", None)
    payload.pop("base_pair_dir", None)
    return _fingerprint(payload)


def _read_manifest(family_dir: Path) -> dict | None:
    path = _manifest_path(family_dir)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_manifest(family_dir: Path, payload: dict, pair_count: int) -> None:
    family_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "config_hash": _construction_fingerprint(payload),
        "pair_count": pair_count,
        "construction": payload,
    }
    _manifest_path(family_dir).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _should_skip_family(family_dir: Path, overwrite: bool, manifest_payload: dict) -> bool:
    existing = _existing_pairs(family_dir)
    manifest = _read_manifest(family_dir)
    expected_hash = _construction_fingerprint(manifest_payload)
    manifest_hash = manifest.get("config_hash") if manifest else None
    manifest_payload_hash = (
        _construction_fingerprint(manifest.get("construction", {})) if manifest else None
    )
    if (
        manifest
        and expected_hash in {manifest_hash, manifest_payload_hash}
        and existing
        and not overwrite
    ):
        print(
            f"  {family_dir.name}: manifest matches ({existing} pairs), skipping generation"
        )
        return True
    if existing and not overwrite:
        reason = "missing manifest" if manifest is None else "manifest/config mismatch"
        raise SystemExit(
            f"{_display_path(family_dir)} already contains {existing} pairs but {reason}. "
            "Use --overwrite to regenerate, or move/delete the old directory."
        )
    return False


def _orient_manifest_payload(family_name: str, cfg) -> dict:
    family = cfg.families[family_name]
    dataset = OmegaConf.to_container(family.dataset, resolve=True)
    return {
        "family": family_name,
        "generator": "get_data.py",
        "generator_manifest_version": MANIFEST_VERSION,
        "config_path": _display_path(ORIENTABILITY_CONFIG),
        "base_pair_dir": _display_path(_family_output_dir(family_name, cfg)),
        "dataset": dataset,
    }


def _fg_manifest_payload(cfg) -> dict:
    keys = (
        "min_g",
        "max_g",
        "samples_per_g",
        "noise_pairs_per_g",
        "min_subdiv",
        "max_subdiv",
        "match_backtracks",
        "max_tries",
        "max_expansions",
        "seed",
    )
    dataset = {key: OmegaConf.select(cfg, key) for key in keys}
    return {
        "family": "free_vs_surface",
        "generator": "get_data.py",
        "generator_manifest_version": MANIFEST_VERSION,
        "config_path": _display_path(FG_CONFIG),
        "base_pair_dir": _display_path(SCRIPT_DIR / "free_vs_surface"),
        "dataset": dataset,
    }


def generate_free_vs_surface(cfg, overwrite: bool = False) -> int:
    """Free-vs-Surface base pairs, saved under generate_data/free_vs_surface."""
    ComplexData, generate_words, FundamentalGroupDataset = _fg_imports()
    family_dir = SCRIPT_DIR / "free_vs_surface"
    manifest_payload = _fg_manifest_payload(cfg)
    if _should_skip_family(family_dir, overwrite, manifest_payload):
        return _existing_pairs(family_dir)

    rng = np.random.default_rng(int(cfg.seed))
    min_g = int(cfg.min_g)
    max_g = int(cfg.max_g)
    samples_per_g = int(cfg.samples_per_g)
    count = 0
    for g in range(min_g, max_g + 1):
        for sample in range(samples_per_g):
            word_a, word_b, _ = generate_words(
                g,
                noise_pairs_per_g=int(cfg.noise_pairs_per_g),
                rng=rng,
                min_subdiv=int(cfg.min_subdiv),
                max_subdiv=int(cfg.max_subdiv),
                match_backtracks=bool(cfg.match_backtracks),
                max_tries=int(cfg.max_tries),
                max_expansions=int(cfg.max_expansions),
            )
            surface = ComplexData.create_from_cc(FundamentalGroupDataset._to_ccc(word_a))
            free = ComplexData.create_from_cc(FundamentalGroupDataset._to_ccc(word_b))
            if _save_pair(family_dir, f"g{g}_sample{sample:02d}", surface, free, overwrite):
                count += 1
    total = _existing_pairs(family_dir)
    _write_manifest(family_dir, manifest_payload, total)
    return total


def _dim_range(dataset_cfg) -> range:
    return range(int(dataset_cfg.min_dim), int(dataset_cfg.max_dim) + 1)


def generate_cylinder_vs_mobius(cfg, overwrite: bool = False) -> int:
    ComplexData, _build_closed_grid, build_bounded_grid, _to_ccc_bounded, _to_ccc_closed, _to_ccc_punc = _orient_imports()
    family = cfg.families.cylinder_vs_mobius
    dataset = family.dataset
    family_dir = _family_output_dir("cylinder_vs_mobius", cfg)
    manifest_payload = _orient_manifest_payload("cylinder_vs_mobius", cfg)
    if _should_skip_family(family_dir, overwrite, manifest_payload):
        return _existing_pairs(family_dir)

    count = 0
    for h in _dim_range(dataset):
        for p in _dim_range(dataset):
            edges_c, faces_c = build_bounded_grid(h, p, twisted=False)
            edges_m, faces_m = build_bounded_grid(h, p, twisted=True)
            cylinder = ComplexData.create_from_cc(_to_ccc_bounded(edges_c, faces_c))
            mobius = ComplexData.create_from_cc(_to_ccc_bounded(edges_m, faces_m))
            if _save_pair(family_dir, f"h{h}_p{p}", cylinder, mobius, overwrite):
                count += 1
    total = _existing_pairs(family_dir)
    _write_manifest(family_dir, manifest_payload, total)
    return total


def generate_torus_vs_klein(cfg, overwrite: bool = False) -> int:
    ComplexData, build_closed_grid, _build_bounded_grid, _to_ccc_bounded, _to_ccc_closed, _to_ccc_punc = _orient_imports()
    family = cfg.families.torus_vs_klein
    dataset = family.dataset
    family_dir = _family_output_dir("torus_vs_klein", cfg)
    manifest_payload = _orient_manifest_payload("torus_vs_klein", cfg)
    if _should_skip_family(family_dir, overwrite, manifest_payload):
        return _existing_pairs(family_dir)

    girth_matched = bool(dataset.girth_matched)
    count = 0
    for h in _dim_range(dataset):
        for p in _dim_range(dataset):
            if girth_matched and p % 2 != 0:
                continue
            vshift = p // 2 if girth_matched else 0
            edges_t, faces_t = build_closed_grid(h, p, twisted=False, vertical_shift=vshift)
            edges_k, faces_k = build_closed_grid(h, p, twisted=True)
            torus = ComplexData.create_from_cc(_to_ccc_closed(edges_t, faces_t))
            klein = ComplexData.create_from_cc(_to_ccc_closed(edges_k, faces_k))
            if _save_pair(family_dir, f"h{h}_p{p}", torus, klein, overwrite):
                count += 1
    total = _existing_pairs(family_dir)
    _write_manifest(family_dir, manifest_payload, total)
    return total


def generate_punc_torus_vs_klein(cfg, overwrite: bool = False) -> int:
    ComplexData, build_closed_grid, _build_bounded_grid, _to_ccc_bounded, _to_ccc_closed, _to_ccc_punc = _orient_imports()
    family = cfg.families.punc_torus_vs_punc_klein
    dataset = family.dataset
    family_dir = _family_output_dir("punc_torus_vs_punc_klein", cfg)
    manifest_payload = _orient_manifest_payload("punc_torus_vs_punc_klein", cfg)
    if _should_skip_family(family_dir, overwrite, manifest_payload):
        return _existing_pairs(family_dir)

    girth_matched = bool(dataset.girth_matched)
    metric_matched = bool(dataset.get("metric_matched", False))
    if metric_matched:
        from experiments.orientability.orient_common import (
            degree_profile,
            node_face_distance_histogram,
            trace_boundary_loops,
            vertical_walk_period,
        )

        p = int(dataset.metric_match_p)
        h_start = int(dataset.metric_match_min_h)
        h_step = int(dataset.metric_match_h_step)
        pair_count = int(dataset.metric_match_pair_count)
        max_dist = int(dataset.metric_match_max_dist)
        shapes = [(h_start + index * h_step, p) for index in range(pair_count)]
    else:
        shapes = [
            (h, p)
            for h in _dim_range(dataset)
            for p in _dim_range(dataset)
            if not girth_matched or p % 2 == 0
        ]

    count = 0
    for h, p in shapes:
        if girth_matched and p % 2 != 0:
            raise ValueError("girth-matched punctured pairs require even p")
        vshift = p // 2 if girth_matched else 0
        edges_t, faces_t = build_closed_grid(
            h, p, twisted=False, punctured=True, vertical_shift=vshift
        )
        edges_k, faces_k = build_closed_grid(h, p, twisted=True, punctured=True)

        if metric_matched:
            distance_t = node_face_distance_histogram(edges_t, faces_t, max_dist)
            distance_k = node_face_distance_histogram(edges_k, faces_k, max_dist)
            cycle_t = tuple(
                vertical_walk_period(h, p, False, column, vshift)
                for column in range(p)
            )
            cycle_k = tuple(
                vertical_walk_period(h, p, True, column)
                for column in range(p)
            )
            assert distance_t == distance_k, f"distance mismatch at h={h}, p={p}"
            assert trace_boundary_loops(edges_t, faces_t) == trace_boundary_loops(
                edges_k, faces_k
            ), f"boundary mismatch at h={h}, p={p}"
            assert degree_profile(edges_t) == degree_profile(edges_k), (
                f"degree mismatch at h={h}, p={p}"
            )
            rank_counts_t = (h * p, len(edges_t), len(faces_t))
            rank_counts_k = (h * p, len(edges_k), len(faces_k))
            assert rank_counts_t == rank_counts_k, f"rank mismatch at h={h}, p={p}"
            assert h * p - len(edges_t) + len(faces_t) == -1
            assert h * p - len(edges_k) + len(faces_k) == -1
            assert cycle_t == cycle_k, f"cycle-period mismatch at h={h}, p={p}"

        torus = ComplexData.create_from_cc(_to_ccc_punc(edges_t, faces_t))
        klein = ComplexData.create_from_cc(_to_ccc_punc(edges_k, faces_k))
        if _save_pair(family_dir, f"h{h}_p{p}", torus, klein, overwrite):
            count += 1
    total = _existing_pairs(family_dir)
    _write_manifest(family_dir, manifest_payload, total)
    return total


def _parse_families(value: str) -> list[str]:
    value = value.strip()
    if value == "all":
        return [*ORIENTABILITY_FAMILIES, "free_vs_surface"]
    if value in ("orientability", "oreintability"):
        return list(ORIENTABILITY_FAMILIES)
    families = []
    for item in value.replace(",", " ").split():
        canonical = FAMILY_ALIASES.get(item.lower())
        if canonical is None:
            raise SystemExit(
                f"Unknown family {item!r}; choose all, orientability, "
                f"{', '.join(sorted(set(FAMILY_ALIASES.values())))}"
            )
        families.append(canonical)
    return families


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        default="orientability",
        help="Family to generate: orientability, all, cylinder_vs_mobius, torus_vs_klein, punc_torus_vs_punc_klein, free_vs_surface. Comma-separated is allowed.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing pair files. Default skips a family if any pairs already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved output paths and expected pair counts without generating files.",
    )
    parser.add_argument(
        "--write-manifest-only",
        action="store_true",
        help="Write manifest.json for existing data if the existing pair count matches the current config. Does not generate tensors.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ORIENTABILITY_CONFIG,
        help="Shared orientability config path.",
    )
    parser.add_argument(
        "--fg-config",
        type=Path,
        default=FG_CONFIG,
        help="Free-vs-surface / fundamental_group config path.",
    )
    return parser.parse_args()


def _expected_orientability_count(family: str, cfg) -> int:
    dataset = cfg.families[family].dataset
    if bool(dataset.get("metric_matched", False)):
        return int(dataset.metric_match_pair_count)
    hs = list(_dim_range(dataset))
    ps = list(_dim_range(dataset))
    if bool(dataset.girth_matched):
        ps = [p for p in ps if p % 2 == 0]
    return len(hs) * len(ps)

def main() -> None:
    args = parse_args()
    global ORIENTABILITY_CONFIG, FG_CONFIG
    ORIENTABILITY_CONFIG = args.config
    FG_CONFIG = args.fg_config
    orient_cfg = _load_config(ORIENTABILITY_CONFIG, "orientability")
    fg_cfg = _load_config(FG_CONFIG, "fundamental-group")
    selected = _parse_families(args.family)

    print(f"Writing base pairs under: {_display_path(SCRIPT_DIR)}")
    print(f"Using orientability config: {_display_path(ORIENTABILITY_CONFIG)}")
    print(f"Using fundamental-group config: {_display_path(FG_CONFIG)}")
    if args.dry_run:
        for family in selected:
            expected = (
                _expected_orientability_count(family, orient_cfg)
                if family in ORIENTABILITY_FAMILIES
                else "config-dependent"
            )
            out_dir = (
                _family_output_dir(family, orient_cfg)
                if family in ORIENTABILITY_FAMILIES
                else SCRIPT_DIR / family
            )
            print(f"  {family}: expected={expected}, output={_display_path(out_dir)}")
        return

    totals: dict[str, int] = {}
    for family in selected:
        print(f"\nGenerating {family}...")
        if family == "cylinder_vs_mobius":
            totals[family] = generate_cylinder_vs_mobius(orient_cfg, args.overwrite)
        elif family == "torus_vs_klein":
            totals[family] = generate_torus_vs_klein(orient_cfg, args.overwrite)
        elif family == "punc_torus_vs_punc_klein":
            totals[family] = generate_punc_torus_vs_klein(orient_cfg, args.overwrite)
        elif family == "free_vs_surface":
            totals[family] = generate_free_vs_surface(fg_cfg, overwrite=args.overwrite)
        else:  # defensive; _parse_families already validates
            raise RuntimeError(f"Unhandled family: {family}")
        print(f"  -> {totals[family]} pairs")

    print("\nDone.")
    for family, total in totals.items():
        out_dir = _family_output_dir(family, orient_cfg) if family in ORIENTABILITY_FAMILIES else SCRIPT_DIR / family
        print(f"  {family}: {total} pairs in {_display_path(out_dir)}")


if __name__ == "__main__":
    main()
