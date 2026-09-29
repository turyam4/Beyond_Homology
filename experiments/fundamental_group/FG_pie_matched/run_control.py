"""
run_control.py -- run the model ladder on a pi_1-matched control family
========================================================================
Reuses the current FG_genus_2 engine and outer fundamental_group/config.yaml.
It swaps ONLY the pair source, by rebinding two functions on the genus-2
generator module:

    generate_tight_pair_descriptor      -> control descriptor
    build_tight_pair_copies_compatible  -> control copies

Both are imported INSIDE utils.py function bodies, so rebinding the module
attribute takes effect at call time. Nothing in the read-only tree is edited.

Cache safety: `cache_version` is suffixed with the family, which flows into
BOTH the cache directory name and the manifest written by cache_spec, and
`control_family` is injected into the manifest as well. A control cache can
therefore never be silently cross-loaded as a cross-class cache -- validate_cache
compares the whole manifest and will refuse.

USAGE  (identical override syntax to run.py, plus control_family=...)

  # 1. build the fixed control dataset once
  python run_control.py control_family=surface_vs_surface cache_only=true

  # 2. SMCN, binary marking (the metric-free configuration)
  python run_control.py control_family=surface_vs_surface \
      model_names=[smcn] binary_marking=true model_seed=0

  # 3. SMCN, shortest-path marking
  python run_control.py control_family=surface_vs_surface \
      model_names=[smcn] binary_marking=false model_seed=0

  # 4. the baseline ladder
  python run_control.py control_family=surface_vs_surface \
      model_names=[esan,homp_cin,gin,hasse_gin,mlp,dist_mlp,boundary_mlp]

  # then repeat all of the above with control_family=free_vs_free

Run the SAME seeds and the SAME model configs you used for the cross-class
family, or the comparison is not controlled.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

SELF_DIR = Path(__file__).resolve().parent
if str(SELF_DIR) not in sys.path:
    sys.path.insert(0, str(SELF_DIR))

from control_family_dataset import (  # noqa: E402
    FAMILIES,
    FG_DIR,
    build_control_pair_copies_compatible,
    generate_control_pair_descriptor,
)

# FG_genus_2 must be importable and remain the CWD-anchored script dir for
# cache paths. Relative run outputs are re-anchored to this control folder.
for _p in (str(FG_DIR), str(FG_DIR.parent), str(FG_DIR.parents[2])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import DictConfig, OmegaConf, open_dict  # noqa: E402


def _extract_family(argv):
    """Pull control_family=... out of argv (it is not a config.yaml key, so
    Hydra's struct mode would reject it as a plain override)."""
    family = None
    rest = []
    for arg in argv:
        if arg.startswith("control_family="):
            family = arg.split("=", 1)[1]
        elif arg.startswith("+control_family="):
            family = arg.split("=", 1)[1]
        else:
            rest.append(arg)
    return family, rest


def _override_names(overrides):
    return {arg.split("=", 1)[0].lstrip("+") for arg in overrides if "=" in arg}


def _selected_models_from_overrides(overrides):
    for arg in overrides:
        key = arg.split("=", 1)[0].lstrip("+")
        if key != "model_names":
            continue
        value = arg.split("=", 1)[1].strip()
        value = value.strip("[]")
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def _memory_safe_overrides(overrides):
    """Apply FG_genus_2 baseline memory settings without changing the models."""
    selected = _selected_models_from_overrides(overrides)
    smcn_only = selected and set(selected) <= {"smcn", "subcomplex"}
    if smcn_only:
        return overrides

    present = _override_names(overrides)
    extra = []
    defaults = {
        "include_subcomplex_features": "false",
        "prune_to_baseline_fields": "true",
        "compact_subcomplex_cache": "true",
        "cache_as_minibatches": "true",
        "construct_cache_by_phase": "true",
        "pin_memory": "false",
    }
    for key, value in defaults.items():
        if key not in present:
            extra.append(f"{key}={value}")
    return overrides + extra


def _display_path(path: str) -> str:
    value = Path(path)
    try:
        return str(value.relative_to(SELF_DIR))
    except ValueError:
        return value.name


def install_control_family(family: str) -> None:
    """Rebind the pair source to the control generator."""
    if family not in FAMILIES:
        raise SystemExit(
            f"control_family must be one of {FAMILIES}, got {family!r}"
        )
    import fundamental_group_dataset_genus2 as tight

    def _descriptor(*, seed, noise_candidates=(2, 4, 8), subdivision=3,
                    max_attempts=5_000):
        return generate_control_pair_descriptor(
            seed=seed, family=family, noise_candidates=noise_candidates,
            subdivision=subdivision, max_attempts=max_attempts,
        )

    def _copies(g, rng, noise_pairs_per_g, min_subdiv, max_subdiv,
                match_backtracks, max_tries, max_expansions,
                number_of_permuted_copies, low_rk, high_rk,
                include_subcomplex_features=True, binary_marking=False):
        return build_control_pair_copies_compatible(
            g, rng, noise_pairs_per_g, min_subdiv, max_subdiv,
            match_backtracks, max_tries, max_expansions,
            number_of_permuted_copies, low_rk, high_rk,
            family=family,
            include_subcomplex_features=include_subcomplex_features,
            binary_marking=binary_marking,
        )

    tight.generate_tight_pair_descriptor = _descriptor
    tight.build_tight_pair_copies_compatible = _copies

    # record the family in the cache manifest as well
    import utils
    _orig_cache_spec = utils.cache_spec

    def _cache_spec(cfg):
        spec = dict(_orig_cache_spec(cfg))
        spec["control_family"] = family
        spec["is_pi1_matched_control"] = True
        return spec

    utils.cache_spec = _cache_spec
    print(f"[control] pair source = {family} (pi_1-matched control family)")


def main() -> None:
    family, overrides = _extract_family(sys.argv[1:])
    if family is None:
        raise SystemExit(
            "Missing control_family=... \n"
            f"  choices: {', '.join(FAMILIES)}\n"
            "  example: python run_control.py control_family=surface_vs_surface "
            "cache_only=true"
        )

    install_control_family(family)

    # Keep the FG engine's cache paths relative to FG_genus_2.
    os.chdir(FG_DIR)

    overrides = _memory_safe_overrides(overrides)
    with initialize_config_dir(config_dir=str(FG_DIR.parent), version_base=None):
        cfg: DictConfig = compose(config_name="config", overrides=overrides)

    with open_dict(cfg):
        cfg.control_family = family
        cfg.is_pi1_matched_control = True
        base_version = str(cfg.get("cache_version", "v1"))
        if not base_version.endswith(family):
            cfg.cache_version = f"{base_version}_ctrl_{family}"
        # Keep control run artifacts in FG_pie_matched, while the shared cache
        # remains in FG_genus_2/fg_pairs.
        out = str(cfg.get("output_dir", "runs"))
        out_path = Path(out)
        if out_path.is_absolute():
            cfg.output_dir = str(out_path)
        else:
            if "control" not in out:
                out_path = (
                    Path(out).parent / f"control_{family}" / Path(out).name
                    if "/" in out
                    else Path(f"{out}_control_{family}")
                )
            cfg.output_dir = str(SELF_DIR / out_path)

    print(f"[control] cache_version = {cfg.cache_version}")
    print(f"[control] output_dir    = {_display_path(str(cfg.output_dir))}")

    import engine
    engine.run(cfg)


if __name__ == "__main__":
    main()
