"""Seed control and an environment record for every reported number.

`reproducible_run(seed)` is the single entry point: it seeds every source of
randomness, opens an MLflow run, and logs the seed and the environment
fingerprint to it. A number that comes out of that block can be traced to
the exact code, packages, and platform that produced it.

Two limits that `set_seed` cannot remove, stated here so no caller relies
on them by accident:

- `PYTHONHASHSEED` is read once, when an interpreter starts. Setting it here
  fixes string hashing for processes started *after* the call (subprocesses,
  spawned data-loader workers), not for the current process.
- `np.random.seed` seeds NumPy's legacy global generator only. A
  `np.random.default_rng()` created without a seed is not affected; pass an
  explicit seed to every `default_rng` call.
"""
from __future__ import annotations

import importlib
import importlib.metadata
import os
import platform
import random
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from symbolic_ts.vocabulary import SCHEMA_VERSION

DEFAULT_PACKAGES: tuple[str, ...] = (
    "symbolic-ts", "numpy", "pandas", "scipy", "statsmodels",
    "torch", "transformers", "peft", "mlflow",
)


@dataclass(frozen=True)
class SeedReport:
    seed: int
    torch_seeded: bool
    # True only if this interpreter already started with PYTHONHASHSEED=seed.
    hash_seed_applies_to_current_process: bool


def _optional_module(name: str):
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def set_seed(seed: int) -> SeedReport:
    """Seed `random`, NumPy's global generator, torch (CPU and every CUDA
    device, when torch is installed), and `PYTHONHASHSEED` for child
    processes."""
    already_set = os.environ.get("PYTHONHASHSEED") == str(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch = _optional_module("torch")
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    return SeedReport(seed=seed, torch_seeded=torch is not None, hash_seed_applies_to_current_process=already_set)


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_fingerprint(
    packages: tuple[str, ...] = DEFAULT_PACKAGES, repo_path: str | Path | None = None
) -> dict[str, Any]:
    """Python version, platform, package versions (None = not installed), the
    vocabulary's schema version, and the git commit of `repo_path` (default:
    the current directory). `git_dirty` is True when the working tree has
    uncommitted changes: the commit alone then does not describe the code."""
    repo = Path(repo_path) if repo_path is not None else Path.cwd()
    status = _git(["status", "--porcelain"], repo)
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "schema_version": SCHEMA_VERSION,
        "packages": {name: _package_version(name) for name in packages},
        "git_commit": _git(["rev-parse", "HEAD"], repo),
        "git_dirty": None if status is None else bool(status),
    }


def _flatten(fingerprint: dict[str, Any]) -> dict[str, str]:
    flat = {}
    for key, value in fingerprint.items():
        if isinstance(value, dict):
            flat.update({f"env.{key}.{k}": str(v) for k, v in value.items()})
        else:
            flat[f"env.{key}"] = str(value)
    return flat


@dataclass(frozen=True)
class ReproducibleRun:
    seed: SeedReport
    fingerprint: dict[str, Any]
    mlflow_run_id: str | None  # None when mlflow is not installed


@contextmanager
def reproducible_run(
    seed: int, *, run_name: str | None = None, repo_path: str | Path | None = None
) -> Iterator[ReproducibleRun]:
    """Seed everything, then open an MLflow run that records the seed and the
    environment fingerprint (as params, and in full as
    `environment_fingerprint.json`). Without mlflow installed, the seeding and
    the fingerprint still happen; only the logging is skipped."""
    seed_report = set_seed(seed)
    fingerprint = environment_fingerprint(repo_path=repo_path)
    mlflow = _optional_module("mlflow")
    if mlflow is None:
        yield ReproducibleRun(seed_report, fingerprint, None)
        return

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({"seed": seed, **_flatten(fingerprint)})
        mlflow.log_dict(fingerprint, "environment_fingerprint.json")
        yield ReproducibleRun(seed_report, fingerprint, run.info.run_id)
