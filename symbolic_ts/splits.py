"""Walk-forward splitting with a purge/embargo gap, and the test-set lock (D9).

Phase 0 used a naive first-70%/last-30% holdout per series -- good enough to
prove sigma binning could be fit train-only, explicitly flagged in its own
notes as NOT the real split (symbolic-ts-research's F0-04/F0-10 notes). This
module is the real thing: walk-forward folds with a purge gap sized to the
context window (so no training sequence's input window overlaps a
validation target), plus the mechanism that keeps the actual held-out test
set (D9) locked until the Phase 4 gate.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sized

import numpy as np
import numpy.typing as npt

from symbolic_ts.vocabulary import SCHEMA_VERSION

# WORKING_PLAN.md's locked context window: 50, derived in F0-09 from the real
# PACF cutoff (replacing the earlier placeholder of 30) -- a >=10x margin
# over the largest clean-decay lag measured in any domain. A starting point
# callers can override, not a hardcoded requirement.
DEFAULT_CONTEXT_LEN = 50


class TestSetAccessWarning(UserWarning):
    """Emitted every time the locked test split (D9) is actually read."""


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold. `train_indices` and `validation_indices` are
    positions into the original data, not copies of the data itself --
    callers slice with them (e.g. `data.iloc[fold.train_indices]`)."""

    train_indices: np.ndarray
    validation_indices: np.ndarray


def walk_forward_split(
    data: Sized,
    n_folds: int,
    context_len: int = DEFAULT_CONTEXT_LEN,
    embargo: int = 0,
) -> list[Fold]:
    """Expanding-window walk-forward split into `n_folds` folds.

    Data is divided into `n_folds + 1` equal-sized contiguous blocks in time
    order. Fold `k`'s validation block is block `k + 1`; its training data is
    every earlier index EXCEPT a purge gap of `context_len + embargo`
    immediately before the validation block starts. The purge gap exists
    because a training sample whose target is the last index before that gap
    would otherwise use an input window (`context_len` steps back) that
    overlaps the validation block's own target -- the model would effectively
    train on a preview of what it's being validated on. `embargo` adds extra
    buffer on top of that, for residual serial correlation beyond what a
    single context window captures.

    Raises `ValueError` if any fold's purge gap consumes the entire training
    region -- a fold with zero usable training data is a configuration bug
    (`n_folds` too high, or `context_len + embargo` too large for the data),
    not a valid result to return silently.
    """
    n = len(data)
    if n_folds < 1:
        raise ValueError(f"n_folds must be >= 1, got {n_folds}")
    gap = context_len + embargo

    block_size = n // (n_folds + 1)
    if block_size < 1:
        raise ValueError(f"not enough data ({n} rows) for {n_folds} folds")

    folds = []
    for k in range(n_folds):
        val_start = block_size * (k + 1)
        val_end = n if k == n_folds - 1 else block_size * (k + 2)
        train_end = val_start - gap
        if train_end <= 0:
            raise ValueError(
                f"fold {k}: purge gap ({gap} = context_len + embargo) leaves no "
                f"training data before validation start {val_start}; reduce "
                f"n_folds, context_len, or embargo"
            )
        folds.append(
            Fold(
                train_indices=np.arange(0, train_end),
                validation_indices=np.arange(val_start, val_end),
            )
        )
    return folds


def _hash_array(values: npt.ArrayLike) -> str:
    arr = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _log_test_peek_count_to_mlflow(value: int) -> None:
    """Best-effort MLflow logging -- mlflow is never a hard dependency of
    this library (it isn't installed in every environment that imports
    symbolic-ts, e.g. plain inference), and a missing/unconfigured tracking
    server must never break a test-split read. The warning emitted by
    `TestSetLock.read()` is the enforced record either way."""
    try:
        import mlflow
    except ImportError:
        return
    try:
        mlflow.log_metric("test_peek_count", value)
    except Exception:
        pass


@dataclass
class TestSetLock:
    """The test-set lock (D9): a test split hashed at creation, with every
    subsequent read counted and reported. `read()` doesn't prevent access
    (Claude Code / a training script still needs to eventually run on the
    test set at the Phase 4 gate) -- it makes access impossible to do
    silently."""

    indices: np.ndarray
    data_hash: str
    schema_version: str = SCHEMA_VERSION
    test_peek_count: int = 0

    @classmethod
    def create(cls, data: npt.ArrayLike, test_fraction: float) -> "TestSetLock":
        """Locks the LAST `test_fraction` of `data` (chronological holdout --
        the test set must be the most recent data, never a random slice, or
        the model could be evaluated on the past having trained on the
        future). `test_fraction` has no built-in default: the actual split
        proportion is a decision for whoever calls this with real data, not
        a number this library invents."""
        if not 0 < test_fraction < 1:
            raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
        arr = np.asarray(data)
        n = len(arr)
        test_start = n - int(round(n * test_fraction))
        indices = np.arange(test_start, n)
        return cls(indices=indices, data_hash=_hash_array(arr[indices]))

    def read(self) -> np.ndarray:
        """Returns the locked indices, and logs the access: increments
        `test_peek_count`, emits `TestSetAccessWarning`, and best-effort logs
        `test_peek_count` to MLflow (see `_log_test_peek_count_to_mlflow`).
        The count is in-memory only -- call `save()` again afterward to
        persist it, same as any other mutation to a loaded artifact."""
        self.test_peek_count += 1
        warnings.warn(
            f"test split accessed (test_peek_count={self.test_peek_count}) -- "
            f"per D9, every access must be reported in the thesis",
            TestSetAccessWarning,
            stacklevel=2,
        )
        _log_test_peek_count_to_mlflow(self.test_peek_count)
        return self.indices.copy()

    def save(self, path: str | Path) -> None:
        payload = {
            "schema_version": self.schema_version,
            "data_hash": self.data_hash,
            "indices": self.indices.tolist(),
            "test_peek_count": self.test_peek_count,
        }
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "TestSetLock":
        payload = json.loads(Path(path).read_text())
        return cls(
            indices=np.array(payload["indices"]),
            data_hash=payload["data_hash"],
            schema_version=payload["schema_version"],
            test_peek_count=payload["test_peek_count"],
        )
