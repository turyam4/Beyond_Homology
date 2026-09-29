"""Unified Hydra entrypoint for every genus-2 model, including SPEC_MLP."""
from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

SCRIPT_DIR = Path(__file__).resolve().parent
SMCN_ROOT = SCRIPT_DIR.parents[2]
for path in (SMCN_ROOT.parent, SMCN_ROOT, SCRIPT_DIR.parent, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import engine


@hydra.main(config_path="..", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    engine.run(cfg)


if __name__ == "__main__":
    main()
