import warnings

import numpy as np
import pytest
import statsmodels.api as sm
from scipy import stats

from symbolic_ts import stationarity as st


@pytest.fixture(scope="module")
def sunspots():
    return sm.datasets.sunspots.load_pandas().data["SUNACTIVITY"]


def test_adf_reproduces_statsmodels_documented_sunspots_example(sunspots):
    # statsmodels "Stationarity and detrending (ADF/KPSS)" notebook, printed to 6 dp
    result = st.adf_test(sunspots)
    assert result.statistic == pytest.approx(-2.837781, abs=1e-6)
    assert result.p_value == pytest.approx(0.053076, abs=1e-6)
    assert result.lags == 8
    assert result.n_obs == 300
    assert result.critical_values["1%"] == pytest.approx(-3.452337, abs=1e-6)
    assert result.critical_values["5%"] == pytest.approx(-2.871223, abs=1e-6)
    assert result.critical_values["10%"] == pytest.approx(-2.571929, abs=1e-6)
    assert not result.reject_null  # p=0.053 > 0.05: unit root not rejected
    assert not result.is_stationary


def test_kpss_reproduces_statsmodels_documented_sunspots_example(sunspots):
    result = st.kpss_test(sunspots)
    assert result.statistic == pytest.approx(0.669866, abs=1e-6)
    assert result.p_value == pytest.approx(0.016285, abs=1e-6)
    assert result.lags == 7
    assert result.critical_values == pytest.approx({"10%": 0.347, "5%": 0.463, "2.5%": 0.574, "1%": 0.739})
    assert not result.p_value_clipped
    assert result.reject_null  # p=0.016 < 0.05: stationarity rejected
    assert not result.is_stationary


def test_adf_and_kpss_conclusions_point_the_same_way_despite_opposite_nulls():
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(1000)
    walk = np.cumsum(noise)

    assert st.adf_test(noise).is_stationary
    assert st.kpss_test(noise).is_stationary
    assert not st.adf_test(walk).is_stationary
    assert not st.kpss_test(walk).is_stationary


def test_kpss_flags_a_clipped_p_value_instead_of_hiding_it():
    walk = np.cumsum(np.random.default_rng(0).standard_normal(1000))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the InterpolationWarning must not leak out
        result = st.kpss_test(walk)
    assert result.p_value_clipped
    assert result.p_value == pytest.approx(0.01)  # the table's edge, not the real p-value


def test_acf_bounds_follow_nists_bartlett_formula():
    x = np.random.default_rng(1).standard_normal(500)
    result = st.acf_with_bounds(x, nlags=10)
    z = stats.norm.ppf(0.975)
    expected = [z * np.sqrt((1 + 2 * np.sum(result.values[: k - 1] ** 2)) / len(x)) for k in range(1, 11)]
    assert result.bound_method == "bartlett"
    assert result.lags.tolist() == list(range(1, 11))
    assert result.bounds == pytest.approx(expected)
    assert result.bounds[0] == pytest.approx(z / np.sqrt(len(x)))  # lag 1: nothing below it yet


def test_pacf_bounds_are_the_constant_white_noise_band():
    x = np.random.default_rng(2).standard_normal(400)
    result = st.pacf_with_bounds(x, nlags=8)
    assert result.bound_method == "white-noise"
    assert result.bounds == pytest.approx(np.full(8, stats.norm.ppf(0.975) / np.sqrt(400)))


def test_pacf_of_ar1_cuts_off_after_lag_1():
    rng = np.random.default_rng(3)
    x = np.zeros(5000)
    for t in range(1, len(x)):
        x[t] = 0.7 * x[t - 1] + rng.standard_normal()
    result = st.pacf_with_bounds(x, nlags=5)
    assert result.values[0] == pytest.approx(0.7, abs=0.03)
    assert result.outside_bounds()[0]
    assert not result.outside_bounds()[1:].any()


def _correlogram(values, bound=0.1):
    values = np.asarray(values, dtype=float)
    return st.Correlogram(
        kind="ACF", lags=np.arange(1, len(values) + 1), values=values,
        bounds=np.full(len(values), bound), n_obs=100, alpha=0.05, bound_method="bartlett",
    )


def test_first_lag_within_bounds_ignores_a_single_noisy_crossing():
    # inside at lag 3 only, then outside again; the first 3-in-a-row run starts at lag 5
    c = _correlogram([0.5, 0.3, 0.01, 0.3, 0.01, -0.02, 0.05, 0.2])
    assert c.first_lag_within_bounds(consecutive=3) == 5
    assert c.first_lag_within_bounds(consecutive=1) == 3


def test_first_lag_within_bounds_is_none_when_never_reached():
    assert _correlogram([0.5, 0.4, 0.3]).first_lag_within_bounds() is None


@pytest.mark.parametrize("fn", [st.adf_test, st.kpss_test])
def test_tests_refuse_nan_rather_than_silently_dropping_it(fn):
    with pytest.raises(ValueError, match="NaN"):
        fn([1.0, 2.0, np.nan, 3.0] * 20)
