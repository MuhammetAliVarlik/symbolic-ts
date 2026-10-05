"""Channel projection: reduce an arbitrary number of input channels down to
the two summary channels the token vocabulary is built on -- change and
volatility. This is what lets one token grammar describe finance (2+
channels), ETT (7 channels), and a many-sensor domain (100+ channels) alike:
every domain is reduced to the same two-channel shape before binning.

There is deliberately no `level` output channel (D2, revised after the F0-04
prototype): a raw level is non-stationary, so a sigma threshold fit on one
region of a growing/drifting series doesn't describe another region of the
same series. Cumulative change already carries the same information a coarse
discretised level would.

Ported from symbolic-ts-research's F0-04 prototype
(scripts/prototype_tokenize.py); see that module's docstring for the
underlying evidence (AAPL log-return scaling, ETT flat-sensor zero windows).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

ChangeType = Literal["diff", "log_return"]
DailyAggregation = Literal["last", "mean"]

# Rolling window for the volatility channel. Matches the short-window
# realised-volatility figure validated in Phase 0 (F0-04); revisit only with
# evidence from the real ACF/PACF lag structure (F1's analogue of F0-09).
DEFAULT_VOLATILITY_WINDOW = 20

# Default daily aggregation for sub-daily readings (decision B1, Muhammet,
# 2026-10-05): the LAST reading of each calendar day. Finance's daily bar is
# an end-of-day close, so `.last()` gives a sub-daily sensor the same
# meaning: a snapshot at day end, not an average over the day. This is a
# recorded design decision, not a neutral detail: in symbolic-ts-research,
# F0-11 and F1-11 showed that `.last()` and `.mean()` change the cross-domain
# similarity results materially. Report both where that comparison matters.
DEFAULT_DAILY_AGGREGATION: DailyAggregation = "last"


@dataclass(frozen=True)
class ProjectionConfig:
    """A domain adapter, expressed as data rather than a code branch: which
    column of the input DataFrame is the base reading each channel is
    derived from, and how its change should be computed. A new domain is a
    new `ProjectionConfig` value, not a new `if domain == ...` branch inside
    `project()`.
    """

    base_reading: str
    change_type: ChangeType = "diff"
    volatility_window: int = DEFAULT_VOLATILITY_WINDOW


# Phase 0 domain adapters (F0-04): finance's level grows multiplicatively
# over a multi-year window, so a raw dollar difference is calibrated to
# whichever price regime dominates the training slice -- its change channel
# is a log return instead. ETT's sensor reading has no equivalent growth
# trend, so a plain difference holds up.
FINANCE_ADAPTER = ProjectionConfig(base_reading="Adj Close", change_type="log_return")
ETT_ADAPTER = ProjectionConfig(base_reading="OT", change_type="diff")


def _compute_change(reading: pd.Series, change_type: ChangeType) -> pd.Series:
    if change_type == "log_return":
        return np.log(reading).diff()
    if change_type == "diff":
        return reading.diff()
    raise ValueError(f"unknown change_type {change_type!r}, expected 'diff' or 'log_return'")


def project(df: pd.DataFrame, config: ProjectionConfig) -> pd.DataFrame:
    """Reduce `df` (any number of input channels) to two summary channels:
    `change`, `volatility`. Only `config.base_reading` is read -- every
    other input column is ignored, which is what makes the same function
    work unchanged whether `df` has 1 column or 100.

    Missing values: a NaN in the base reading propagates through `change`
    (one row) and `volatility` (up to `volatility_window` rows, since
    pandas' rolling std treats any window containing a NaN as
    under-observed), and every resulting NaN row is dropped from the
    output -- the same warmup cost already paid at the start of every
    series. The output is therefore guaranteed to never contain NaN.
    """
    if config.base_reading not in df.columns:
        raise KeyError(
            f"base_reading {config.base_reading!r} not found in input columns: {list(df.columns)}"
        )
    reading = df[config.base_reading]
    change = _compute_change(reading, config.change_type)
    volatility = change.rolling(config.volatility_window).std()
    return pd.DataFrame({"change": change, "volatility": volatility}).dropna()


def to_daily(df: pd.DataFrame, how: DailyAggregation = DEFAULT_DAILY_AGGREGATION) -> pd.DataFrame:
    """Resample a sub-daily DataFrame (DatetimeIndex) to one row per calendar
    day, before `project()`. `how="last"` (the default, see
    `DEFAULT_DAILY_AGGREGATION`) keeps each column's last non-missing value of
    the day; `how="mean"` averages the day. Days with a missing value after
    aggregation are dropped, the same rule `project()` applies.

    Use this when a sub-daily domain is compared with a daily one: a lag-1
    transition must cover comparable elapsed time in both (F0-07's sampling
    frequency finding)."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"to_daily needs a DatetimeIndex, got {type(df.index).__name__}")
    daily = df.resample("D")
    if how == "last":
        out = daily.last()
    elif how == "mean":
        out = daily.mean()
    else:
        raise ValueError(f"unknown how {how!r}, expected 'last' or 'mean'")
    return out.dropna()
