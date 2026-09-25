"""Project paths, global seed and YAML configuration loading."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    """Canonical locations for every artifact the project produces."""

    root: Path = PROJECT_ROOT
    config: Path = PROJECT_ROOT / "config"
    data_raw: Path = PROJECT_ROOT / "data" / "raw"
    data_processed: Path = PROJECT_ROOT / "data" / "processed"
    data_synthetic: Path = PROJECT_ROOT / "data" / "synthetic"
    models: Path = PROJECT_ROOT / "models"
    results: Path = PROJECT_ROOT / "results"
    metrics: Path = PROJECT_ROOT / "results" / "metrics"
    figures: Path = PROJECT_ROOT / "results" / "figures"
    examples: Path = PROJECT_ROOT / "results" / "examples"

    def ensure(self) -> None:
        for field_name in (
            "data_raw",
            "data_processed",
            "data_synthetic",
            "models",
            "metrics",
            "figures",
            "examples",
        ):
            getattr(self, field_name).mkdir(parents=True, exist_ok=True)


PATHS = Paths()


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file, resolving bare names against the config directory."""
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = PATHS.config / path
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_global_seed(seed: int = SEED) -> None:
    """Seed every source of randomness the pipeline touches."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:  # torch is optional for quality-only workflows
        pass
