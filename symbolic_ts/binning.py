"""Sigma-based and fixed-percentage quantisation, with explicit fit/transform
separation and a persistable calibration artifact.

Sigma is computed from a training slice ONLY (D1) -- the previous code fit
`qcut` over the entire series, leaking future information into every token
derived from it. `fit`/`transform` are separate calls specifically so that
mistake can't happen again: `transform` never re-fits, and raises if called
before `fit`/`load`.

Scope: this module knows nothing about "change" or "volatility" channels,
log returns, or rolling windows -- deciding how a raw reading becomes a
channel worth binning is a projection-level concern (F1-03). SigmaBinner and
FixedPercentBinner are pure numeric binners, parameterised by whatever label
set the caller imports from `vocabulary.py` -- no label strings are
hardcoded here, e.g.:

    from symbolic_ts.vocabulary import NEUTRAL_CHANGE_LABELS
    binner = SigmaBinner(labels=NEUTRAL_CHANGE_LABELS).fit(train_change)

One piece of projection-adjacent behaviour IS carried over from the Phase 0
prototype, because it's genuinely a binning concern, not a projection one:
`log_scale`, for binning a non-negative, right-skewed quantity like a
rolling-std volatility channel. See its docstring below.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import numpy.typing as npt

from symbolic_ts.vocabulary import SCHEMA_VERSION

# D1 (locked design decision, WORKING_PLAN.md): sigma bucket edges, in units
# of training-fit standard deviations either side of the training-fit mean.
# Changing this requires a written rationale.
DEFAULT_SIGMA_EDGES: tuple[float, ...] = (-1.5, -0.5, 0.5, 1.5)

DEFAULT_ZERO_TOLERANCE = 1e-10
DEFAULT_LOG_EPSILON = 1e-8


class NotFittedError(RuntimeError):
    """Raised by transform()/save() when called before fit() or load()."""


@dataclass
class SigmaBinner:
    """Sigma-based quantisation: bucket a value by how many training-fit
    standard deviations it sits from the training-fit mean.

    `log_scale=True` fits and transforms on log(x) instead of x -- for a
    non-negative, right-skewed quantity such as a rolling-std volatility
    channel, symmetric sigma bins on the raw value leave the bottom bucket
    almost empty (see symbolic-ts-research's F0-04/F0-05 notes). Values
    within `zero_tolerance` of zero are excluded from the FIT: a handful of
    exact/near-zero windows would otherwise dominate the variance
    calculation once the log transform blows them up into large negative
    outliers (measured impact in the prototype: a ~4.5x inflation of the
    fitted std from under 2% of rows). They are still classified at
    transform time via `log(x + log_epsilon)`, which correctly places them
    in the lowest bucket rather than crashing on log(0).
    """

    labels: Sequence[str]
    sigma_edges: Sequence[float] = DEFAULT_SIGMA_EDGES
    log_scale: bool = False
    zero_tolerance: float = DEFAULT_ZERO_TOLERANCE
    log_epsilon: float = DEFAULT_LOG_EPSILON
    mean_: float | None = field(default=None, init=False, repr=False)
    std_: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.labels) != len(self.sigma_edges) + 1:
            raise ValueError(
                f"expected {len(self.sigma_edges) + 1} labels for {len(self.sigma_edges)} "
                f"sigma edges, got {len(self.labels)}"
            )

    @property
    def is_fitted(self) -> bool:
        return self.mean_ is not None and self.std_ is not None

    def _log(self, arr: np.ndarray) -> np.ndarray:
        return np.log(arr + self.log_epsilon)

    def fit(self, train_data: npt.ArrayLike) -> "SigmaBinner":
        arr = np.asarray(train_data, dtype=float)
        if self.log_scale:
            arr = arr[~np.isclose(arr, 0.0, atol=self.zero_tolerance)]
            arr = self._log(arr)
        self.mean_ = float(arr.mean())
        self.std_ = float(arr.std())
        return self

    def transform(self, data: npt.ArrayLike) -> list[str]:
        if not self.is_fitted:
            raise NotFittedError("SigmaBinner.transform() called before fit() or load()")
        arr = np.asarray(data, dtype=float)
        if self.log_scale:
            arr = self._log(arr)
        z = (arr - self.mean_) / self.std_
        idx = np.digitize(z, self.sigma_edges)
        return [self.labels[i] for i in idx]

    def save(self, path: str | Path) -> None:
        if not self.is_fitted:
            raise NotFittedError("SigmaBinner.save() called before fit()")
        payload = {
            "schema_version": SCHEMA_VERSION,
            "binner_type": "SigmaBinner",
            "labels": list(self.labels),
            "sigma_edges": list(self.sigma_edges),
            "log_scale": self.log_scale,
            "zero_tolerance": self.zero_tolerance,
            "log_epsilon": self.log_epsilon,
            "mean": self.mean_,
            "std": self.std_,
        }
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "SigmaBinner":
        payload = json.loads(Path(path).read_text())
        binner = cls(
            labels=tuple(payload["labels"]),
            sigma_edges=tuple(payload["sigma_edges"]),
            log_scale=payload["log_scale"],
            zero_tolerance=payload["zero_tolerance"],
            log_epsilon=payload["log_epsilon"],
        )
        binner.mean_ = payload["mean"]
        binner.std_ = payload["std"]
        return binner


@dataclass
class FixedPercentBinner:
    """Legacy fixed-percentage (quantile) binning -- kept for the E1
    ablation, not used by the default (sigma-based) pipeline. Bucket edges
    are the training data's own quantiles, fit once and reused -- not the
    previous behaviour of running `qcut` over the whole series, which
    silently redraws every bucket boundary (and therefore the meaning of
    every already-emitted token) each time new data arrives.
    """

    labels: Sequence[str]
    quantiles: Sequence[float] = (0.2, 0.4, 0.6, 0.8)
    edges_: tuple[float, ...] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.labels) != len(self.quantiles) + 1:
            raise ValueError(
                f"expected {len(self.quantiles) + 1} labels for {len(self.quantiles)} "
                f"quantiles, got {len(self.labels)}"
            )

    @property
    def is_fitted(self) -> bool:
        return self.edges_ is not None

    def fit(self, train_data: npt.ArrayLike) -> "FixedPercentBinner":
        arr = np.asarray(train_data, dtype=float)
        self.edges_ = tuple(float(x) for x in np.quantile(arr, self.quantiles))
        return self

    def transform(self, data: npt.ArrayLike) -> list[str]:
        if not self.is_fitted:
            raise NotFittedError("FixedPercentBinner.transform() called before fit() or load()")
        arr = np.asarray(data, dtype=float)
        idx = np.digitize(arr, self.edges_)
        return [self.labels[i] for i in idx]

    def save(self, path: str | Path) -> None:
        if not self.is_fitted:
            raise NotFittedError("FixedPercentBinner.save() called before fit()")
        payload = {
            "schema_version": SCHEMA_VERSION,
            "binner_type": "FixedPercentBinner",
            "labels": list(self.labels),
            "quantiles": list(self.quantiles),
            "edges": list(self.edges_),
        }
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "FixedPercentBinner":
        payload = json.loads(Path(path).read_text())
        binner = cls(labels=tuple(payload["labels"]), quantiles=tuple(payload["quantiles"]))
        binner.edges_ = tuple(payload["edges"])
        return binner
