"""The full tokenisation path, end to end, for the thesis-invariant tests.

Not a test module (leading underscore): `test_thesis_invariants.py` imports
it, and the cross-process determinism test runs it in fresh interpreters,
so the exact same code is what both sides execute.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from symbolic_ts.binning import SigmaBinner
from symbolic_ts.events import ETTEventAdapter
from symbolic_ts.projection import ETT_ADAPTER, FINANCE_ADAPTER, ProjectionConfig, project
from symbolic_ts.splits import TestSetLock, walk_forward_split
from symbolic_ts.vocabulary import NEUTRAL_CHANGE_LABELS, NEUTRAL_VOLATILITY_LABELS


def synthetic_finance(n: int = 800, seed: int = 0) -> pd.DataFrame:
    """Geometric random walk whose second half is twice as volatile -- a regime
    change, so a sigma fit that sees the future measurably differs from one
    that doesn't."""
    rng = np.random.default_rng(seed)
    scale = np.where(np.arange(n) < n // 2, 0.01, 0.02)
    price = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, scale)))
    return pd.DataFrame({"Adj Close": price}, index=pd.date_range("2020-01-01", periods=n, freq="B"))


def synthetic_ett(n: int = 24 * 120, seed: int = 0) -> pd.DataFrame:
    """Hourly sensor reading with a daily cycle, spikes, and flat stretches --
    the flat stretches give exact-zero rolling volatility, the case
    `log_scale` binning has to survive."""
    rng = np.random.default_rng(seed)
    hours = np.arange(n)
    ot = 20 + 5 * np.sin(2 * np.pi * hours / 24) + rng.normal(0, 0.5, n)
    ot[rng.choice(n, 10, replace=False)] += 25.0
    ot[500:540] = ot[500]
    return pd.DataFrame({"OT": ot}, index=pd.date_range("2016-07-01", periods=n, freq="h"))


def tokenize_split(
    raw: pd.DataFrame,
    config: ProjectionConfig,
    *,
    test_fraction: float = 0.2,
    n_folds: int = 3,
    context_len: int = 50,
) -> dict:
    """project -> lock the test split -> walk-forward on the rest -> fit
    binners on the LAST fold's training indices only -> tokens for that
    fold's validation block. Returns everything a test might inspect."""
    projected = project(raw, config)
    lock = TestSetLock.create(np.arange(len(projected)), test_fraction=test_fraction)
    pre_test = projected.iloc[: lock.indices[0]]
    folds = walk_forward_split(pre_test, n_folds=n_folds, context_len=context_len)
    fold = folds[-1]

    train = pre_test.iloc[fold.train_indices]
    validation = pre_test.iloc[fold.validation_indices]
    change = SigmaBinner(labels=NEUTRAL_CHANGE_LABELS).fit(train["change"])
    volatility = SigmaBinner(labels=NEUTRAL_VOLATILITY_LABELS, log_scale=True).fit(train["volatility"])

    tokens = [
        f"{c}_{v}"
        for c, v in zip(change.transform(validation["change"]), volatility.transform(validation["volatility"]))
    ]
    return {
        "projected": projected,
        "lock": lock,
        "folds": folds,
        "train": train,
        "validation": validation,
        "change_binner": change,
        "volatility_binner": volatility,
        "tokens": tokens,
    }


def tokens_digest() -> str:
    """One hash over both domains' tokens plus ETT event tokens."""
    finance = tokenize_split(synthetic_finance(), FINANCE_ADAPTER)["tokens"]
    ett_result = tokenize_split(synthetic_ett(), ETT_ADAPTER)
    adapter = ETTEventAdapter(peak_hours=frozenset({18, 19, 20}))
    events = [e for ts in ett_result["validation"].index for e in adapter.events_for(ts)]
    joined = "\n".join(finance + ett_result["tokens"] + events)
    return hashlib.sha256(joined.encode()).hexdigest()


if __name__ == "__main__":
    print(tokens_digest())
