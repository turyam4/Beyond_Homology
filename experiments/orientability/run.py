"""Single baseline evaluator for all orientability benchmark families."""
from __future__ import annotations

from pathlib import Path
import sys

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
EXPERIMENTS_DIR = SCRIPT_DIR.parent
for _path in (PROJECT_ROOT, EXPERIMENTS_DIR, SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from engine import run  # noqa: E402


def _merge_family_cfg(cfg: DictConfig) -> tuple[DictConfig, Path, str]:
    family = str(cfg.family)
    if family not in cfg.families:
        raise ValueError(
            f"Unknown family={family!r}; choose one of {list(cfg.families.keys())}"
        )
    family_cfg = cfg.families[family]
    family_dir = SCRIPT_DIR / str(family_cfg.family_dir)
    merged = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.set_struct(merged, False)
    merged = OmegaConf.merge(merged, family_cfg)
    OmegaConf.set_struct(merged, False)

    # Family-specific model overrides, e.g. dist_mlp.high_rank differs between
    # cylinder (rank 1) and the 2D surface families (rank 2).
    if "models" in family_cfg:
        for model_name, model_overrides in family_cfg.models.items():
            with open_dict(merged):
                merged.models[model_name] = OmegaConf.merge(
                    merged.models[model_name], model_overrides
                )

    with open_dict(merged):
        merged.family = family
        merged.family_label = str(family_cfg.label)
        if "dataset" in family_cfg:
            merged.dataset = family_cfg.dataset
        del merged.families
    return merged, family_dir, str(family_cfg.label)


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    merged, family_dir, label = _merge_family_cfg(cfg)
    run(merged, family_dir, label)


if __name__ == "__main__":
    main()
