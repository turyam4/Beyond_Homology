"""
run_relator.py -- PHASE 4: run the existing ladder on a relator-family pair
===========================================================================
Reuses the FG_genus_2 harness (engine.py / utils.py / config.yaml / every
baseline) unchanged. It rebinds the pair source on the genus-2 generator
module:

    generate_tight_pair_descriptor      -> relator-family descriptor
    build_tight_pair_copies_compatible  -> relator-family copies

Both are imported INSIDE utils.py function bodies, so rebinding the module
attribute takes effect at call time. Nothing in the read-only tree is edited.

Cache safety: `cache_version` is suffixed with the pairing (this flows into
BOTH the cache directory name and the manifest via cache_spec), and `pairing`
is injected into the manifest. validate_cache compares the whole manifest, so
a relator-family cache can never be silently cross-loaded as the
surface-vs-free cache.

USAGE
  python run_relator.py pairing=deep_vs_free cache_only=true
  python run_relator.py pairing=deep_vs_free 'model_names=[smcn]' binary_marking=true model_seed=0
  python run_relator.py pairing=deep_vs_free 'model_names=[esan,homp_cin,gin,hasse_gin,mlp]' model_seed=0
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

SELF = Path(__file__).resolve().parent
if str(SELF) not in sys.path:
    sys.path.insert(0, str(SELF))

from relator_pair_dataset import (  # noqa: E402
    FG_G2, FG_ROOT, PAIRINGS,
    build_relator_pair_copies_compatible, generate_relator_pair_descriptor,
)

for _p in (str(FG_G2), str(FG_ROOT), str(FG_ROOT.parents[1])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import DictConfig, open_dict  # noqa: E402


def _extract(argv):
    pairing, rest = None, []
    for a in argv:
        if a.lstrip("+").startswith("pairing="):
            pairing = a.split("=", 1)[1]
        else:
            rest.append(a)
    return pairing, rest


def install_pairing(pairing: str) -> None:
    if pairing not in PAIRINGS:
        raise SystemExit(f"pairing must be one of {list(PAIRINGS)}, got {pairing!r}")
    import fundamental_group_dataset_genus2 as tight

    def _descriptor(*, seed, noise_candidates=(2, 4, 8), subdivision=3,
                    max_attempts=5_000):
        return generate_relator_pair_descriptor(
            seed=seed, pairing=pairing, noise_candidates=noise_candidates,
            subdivision=subdivision, max_attempts=max_attempts)

    def _copies(g, rng, noise_pairs_per_g, min_subdiv, max_subdiv,
                match_backtracks, max_tries, max_expansions,
                number_of_permuted_copies, low_rk, high_rk,
                include_subcomplex_features=True, binary_marking=False):
        return build_relator_pair_copies_compatible(
            g, rng, noise_pairs_per_g, min_subdiv, max_subdiv,
            match_backtracks, max_tries, max_expansions,
            number_of_permuted_copies, low_rk, high_rk, pairing=pairing,
            include_subcomplex_features=include_subcomplex_features,
            binary_marking=binary_marking)

    tight.generate_tight_pair_descriptor = _descriptor
    tight.build_tight_pair_copies_compatible = _copies

    import utils
    _orig = utils.cache_spec

    def _spec(cfg):
        s = dict(_orig(cfg))
        s["pairing"] = pairing
        s["relator_family"] = True
        return s

    utils.cache_spec = _spec
    a, b = PAIRINGS[pairing]
    print(f"[relator] pair source = {pairing}  ({a} vs {b})")


def main() -> None:
    pairing, overrides = _extract(sys.argv[1:])
    if pairing is None:
        raise SystemExit(
            "Missing pairing=...\n  choices: " + ", ".join(PAIRINGS) +
            "\n  example: python run_relator.py pairing=deep_vs_free cache_only=true")

    install_pairing(pairing)
    os.chdir(FG_G2)  # harness resolves fg_pairs/ runs/ logs/ from here

    with initialize_config_dir(config_dir=str(FG_ROOT), version_base=None):
        cfg: DictConfig = compose(config_name="config", overrides=overrides)

    with open_dict(cfg):
        cfg.pairing = pairing
        base_v = str(cfg.get("cache_version", "v1"))
        if not base_v.endswith(pairing):
            cfg.cache_version = f"{base_v}_rel_{pairing}"
        out = str(cfg.get("output_dir", "runs"))
        if pairing not in out:
            cfg.output_dir = f"{out}_rel_{pairing}"

    print(f"[relator] cache_version={cfg.cache_version}  output_dir={cfg.output_dir}")
    import engine
    engine.run(cfg)


if __name__ == "__main__":
    main()
