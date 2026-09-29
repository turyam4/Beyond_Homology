"""
run_spec_mlp.py -- convenience launcher for the native SPEC_MLP FG baseline
============================================================================
SPEC_MLP is registered in the shared FG configuration and engine, exactly like
the other scalar-feature MLP baselines. This launcher only selects it and keeps
the old short `spec_*` command-line options for convenience.

Because SPEC_MLP is a scalar-feature baseline, the harness materialises its
features ONCE per pair (not once per epoch), and `prune_to_baseline_fields`
keeps exactly the tensors it needs (adjacency + unsigned incidence).

USAGE
  # build/reuse the baseline cache, then run the probe on 1 or 4 GPUs
  ./run_spec_mlp.sh control_family=surface_vs_surface
  ./run_spec_mlp.sh control_family=surface_vs_surface num_gpus=4

  # explicitly choose four model seeds (processed in GPU-sized waves)
  ./run_spec_mlp.sh control_family=surface_vs_surface \
      num_gpus=4 seeds=[4,5,6,7]

Options (Hydra overrides):
  spec_k=16            top-k / bottom-k eigenvalues kept per operator
  spec_moment_L=8      normalised spectral moments tr(M^l)/n, l=1..L
  spec_operators=[A,L,Lnorm,hodge1]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

SELF_DIR = Path(__file__).resolve().parent
if str(SELF_DIR) not in sys.path:
    sys.path.insert(0, str(SELF_DIR))

def _fg_dir() -> Path:
    env = os.environ.get("FG_GENUS2_DIR")
    here = Path(__file__).resolve()
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates += [
        here.parent.parent / "FG_genus_2",
        here.parent.parent.parent / "fundamental_group" / "FG_genus_2",
    ]
    for c in candidates:
        if (c / "engine.py").is_file():
            return c
    raise RuntimeError("Set FG_GENUS2_DIR to the FG_genus_2 directory.")


FG_DIR = _fg_dir()
PROJECT_ROOT = FG_DIR.parents[2]
for _p in (str(FG_DIR), str(FG_DIR.parent), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from baselines.spec_mlp.model import DEFAULT_OPERATORS  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import DictConfig, open_dict  # noqa: E402


def _display_path(path: str) -> str:
    value = Path(path)
    try:
        return str(value.relative_to(SELF_DIR))
    except ValueError:
        return value.name


def main() -> None:
    argv = sys.argv[1:]
    family, overrides = None, []
    spec_opts = {}
    explicit_output_dir = False
    for a in argv:
        if a.startswith(("control_family=", "+control_family=")):
            family = a.split("=", 1)[1]
        elif a.lstrip("+").startswith("spec_"):
            # spec_* are not config.yaml keys; accept them with or without '+'
            k, v = a.lstrip("+").split("=", 1)
            spec_opts[k] = v
        else:
            overrides.append(a)
            if a.split("=", 1)[0].lstrip("+") == "output_dir":
                explicit_output_dir = True

    if family:
        from run_control import install_control_family
        install_control_family(family)

    os.chdir(FG_DIR)
    with initialize_config_dir(config_dir=str(FG_DIR.parent), version_base=None):
        cfg: DictConfig = compose(config_name="config", overrides=overrides)

    with open_dict(cfg):
        model_cfg = cfg.models.spec_mlp
        model_cfg.enabled = True
        model_cfg.hidden_dim = int(
            spec_opts.get("spec_hidden_dim", model_cfg.hidden_dim)
        )
        model_cfg.num_layers = int(
            spec_opts.get("spec_num_layers", model_cfg.num_layers)
        )
        model_cfg.dropout = float(spec_opts.get("spec_dropout", model_cfg.dropout))
        model_cfg.k = int(spec_opts.get("spec_k", model_cfg.k))
        model_cfg.moment_L = int(
            spec_opts.get("spec_moment_L", model_cfg.moment_L)
        )
        if "spec_operators" in spec_opts:
            model_cfg.operators = [
                operator.strip().strip("'\"")
                for operator in spec_opts["spec_operators"].strip("[]").split(",")
                if operator.strip()
            ]
        cfg.model_names = ["spec_mlp"]
        # spectral features need only the pruned baseline tensors
        cfg.prune_to_baseline_fields = True
        cfg.include_subcomplex_features = False
        if family:
            cfg.control_family = family
            base_v = str(cfg.get("cache_version", "v1"))
            if not base_v.endswith(family):
                cfg.cache_version = f"{base_v}_ctrl_{family}"
            if not explicit_output_dir:
                cfg.output_dir = str(
                    SELF_DIR / f"{cfg.get('output_dir','runs')}_control_{family}_spec"
                )
        else:
            out_path = Path(str(cfg.get("output_dir", "runs")))
            if not out_path.is_absolute():
                cfg.output_dir = str(SELF_DIR / out_path)

    print(f"[spec_mlp] cache_version={cfg.cache_version} "
          f"output_dir={_display_path(str(cfg.output_dir))}")

    import engine
    engine.run(cfg)


if __name__ == "__main__":
    main()
