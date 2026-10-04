"""Stationarity tests and correlograms, one implementation for the whole thesis.

Phase 0 ran ADF/KPSS (F0-03) and ACF/PACF (F0-09) from throwaway research
scripts; the thesis pipeline needs the same numbers, so both now call this
module instead of each wrapping `statsmodels` their own way. The arguments
default to exactly what those scripts used, so porting them changes no
result.

Two return types, one per kind of question, with matching field names:

- `StationarityTestResult` (ADF, KPSS): statistic, p-value, lags used, and the
  test's rejection bounds (its critical values).
- `Correlogram` (ACF, PACF): per-lag values and the per-lag confidence bound
  each value is compared against.

ADF and KPSS have opposite nulls (ADF: unit root; KPSS: stationary), so a raw
"rejected" flag means opposite things for the two. `is_stationary` resolves
that once, here, rather than in every caller.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf


@dataclass(frozen=True)
class StationarityTestResult:
    test: str  # "ADF" or "KPSS"
    statistic: float
    p_value: float
    lags: int
    n_obs: int
    critical_values: dict[str, float]  # the test's bounds, keyed by significance level ("5%")
    alpha: float
    # KPSS p-values are interpolated from a table covering only [0.01, 0.1];
    # outside it statsmodels clips to the edge. True means `p_value` is that
    # edge, not the real p-value (the real one is further out).
    p_value_clipped: bool = False

    @property
    def reject_null(self) -> bool:
        return self.p_value < self.alpha

    @property
    def is_stationary(self) -> bool:
        return self.reject_null if self.test == "ADF" else not self.reject_null


@dataclass(frozen=True, eq=False)
class Correlogram:
    kind: str  # "ACF" or "PACF"
    lags: np.ndarray  # 1..nlags; lag 0 is always 1 by definition and is dropped
    values: np.ndarray
    bounds: np.ndarray  # half-width of the confidence band at each lag
    n_obs: int
    alpha: float
    bound_method: str  # "bartlett" (ACF) or "white-noise" (PACF)

    def outside_bounds(self) -> np.ndarray:
        return np.abs(self.values) >= self.bounds

    def first_lag_within_bounds(self, consecutive: int = 3) -> int | None:
        """First lag k where the value stays inside its bound for `consecutive`
        lags in a row, so a single noisy crossing isn't reported as the
        decorrelation point. None if that never happens up to `nlags`.

        A raw signal only: a periodic series can dip inside the band for a
        few lags without having decayed. Telling those apart needs the curve
        itself (F0-09 tried two automatic rules; both failed)."""
        within = ~self.outside_bounds()
        for i in range(len(within) - consecutive + 1):
            if within[i : i + consecutive].all():
                return int(self.lags[i])
        return None


def _as_clean_array(series: npt.ArrayLike) -> np.ndarray:
    values = np.asarray(series, dtype=float)
    if np.isnan(values).any():
        raise ValueError("series contains NaN; drop or fill missing values before testing")
    return values


def adf_test(
    series: npt.ArrayLike, regression: str = "c", autolag: str | None = "AIC", alpha: float = 0.05
) -> StationarityTestResult:
    """Augmented Dickey-Fuller test. H0: the series has a unit root.

    Worked example (statsmodels' "Stationarity and detrending (ADF/KPSS)"
    notebook, https://www.statsmodels.org/stable/examples/notebooks/generated/stationarity_detrending_adf_kpss.html):
    the bundled sunspots `SUNACTIVITY` series, autolag="AIC" -> statistic
    -2.837781, p-value 0.053076, 8 lags, 300 observations, critical values
    1% -3.452337 / 5% -2.871223 / 10% -2.571929.
    """
    values = _as_clean_array(series)
    result = adfuller(values, regression=regression, autolag=autolag)
    statistic, p_value, used_lag, n_obs, critical_values = result[:5]
    return StationarityTestResult(
        test="ADF",
        statistic=float(statistic),
        p_value=float(p_value),
        lags=int(used_lag),
        n_obs=int(n_obs),
        critical_values={k: float(v) for k, v in critical_values.items()},
        alpha=alpha,
    )


def kpss_test(
    series: npt.ArrayLike, regression: str = "c", nlags: str | int = "auto", alpha: float = 0.05
) -> StationarityTestResult:
    """KPSS test. H0: the series is (level- or trend-) stationary.

    Worked example (same statsmodels notebook as `adf_test`): sunspots
    `SUNACTIVITY`, regression="c", nlags="auto" -> statistic 0.669866,
    p-value 0.016285, 7 lags, critical values 10% 0.347 / 5% 0.463 /
    2.5% 0.574 / 1% 0.739.
    """
    values = _as_clean_array(series)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", InterpolationWarning)
        statistic, p_value, lags, critical_values = kpss(values, regression=regression, nlags=nlags)
    clipped = any(issubclass(w.category, InterpolationWarning) for w in caught)
    for w in caught:
        if not issubclass(w.category, InterpolationWarning):
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    return StationarityTestResult(
        test="KPSS",
        statistic=float(statistic),
        p_value=float(p_value),
        lags=int(lags),
        n_obs=len(values),
        critical_values={k: float(v) for k, v in critical_values.items()},
        alpha=alpha,
        p_value_clipped=clipped,
    )


def acf_with_bounds(series: npt.ArrayLike, nlags: int, alpha: float = 0.05) -> Correlogram:
    """Sample ACF with Bartlett confidence bounds.

    Bartlett's bound at lag k widens with every autocorrelation already
    estimated below it, so a slowly decaying series isn't declared
    "significant" at lag 40 just because lag 1 was (NIST handbook,
    autocorrelation plot page, the MA-identification band,
    https://www.itl.nist.gov/div898/handbook/eda/section3/autocopl.htm):

        bound_k = z_{1-alpha/2} * sqrt((1 + 2 * sum_{i=1}^{k-1} r_i^2) / N)
    """
    values = _as_clean_array(series)
    acf_vals, confint = acf(values, nlags=nlags, alpha=alpha, bartlett_confint=True, fft=True)
    return Correlogram(
        kind="ACF",
        lags=np.arange(1, nlags + 1),
        values=acf_vals[1:],
        bounds=(confint[1:, 1] - confint[1:, 0]) / 2,
        n_obs=len(values),
        alpha=alpha,
        bound_method="bartlett",
    )


def pacf_with_bounds(
    series: npt.ArrayLike, nlags: int, alpha: float = 0.05, method: str = "ywadjusted"
) -> Correlogram:
    """Sample PACF with the white-noise bound z_{1-alpha/2} / sqrt(N) at every
    lag (same NIST section as `acf_with_bounds`). Bartlett's formula is an ACF
    result; the constant band is the standard PACF pairing."""
    values = _as_clean_array(series)
    pacf_vals, confint = pacf(values, nlags=nlags, alpha=alpha, method=method)
    return Correlogram(
        kind="PACF",
        lags=np.arange(1, nlags + 1),
        values=pacf_vals[1:],
        bounds=(confint[1:, 1] - confint[1:, 0]) / 2,
        n_obs=len(values),
        alpha=alpha,
        bound_method="white-noise",
    )
