"""Small helpers shared by the training and evaluation scripts."""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np


def repo_root() -> Path:
    """Absolute path to the repository root (the folder holding ``assets/``)."""
    return Path(__file__).resolve().parents[2]


def ensure_src_on_path() -> None:
    """Allow ``python scripts/train_sb3.py`` to work without installing the package."""
    src = repo_root() / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read a YAML config, falling back to an empty dict if the file is absent."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pyyaml is required to read configs: pip install pyyaml") from exc
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def run_dir(algo: str, root: str | os.PathLike[str] | None = None) -> Path:
    """``runs/<algo>/`` inside the repo, created on demand."""
    base = Path(root) if root is not None else repo_root() / "runs"
    out = base / algo
    out.mkdir(parents=True, exist_ok=True)
    return out


def describe_rollout(rewards: list[float], successes: list[bool], distances: list[float]) -> str:
    """One-line summary used by the evaluation script."""
    return (
        f"return {np.mean(rewards):8.2f} +/- {np.std(rewards):5.2f} | "
        f"success {100.0 * np.mean(successes):5.1f}% | "
        f"final distance {np.mean(distances):.4f} m"
    )
