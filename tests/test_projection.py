import numpy as np
import pandas as pd
import pytest

from symbolic_ts.projection import (
    DEFAULT_VOLATILITY_WINDOW,
    ETT_ADAPTER,
    FINANCE_ADAPTER,
    ProjectionConfig,
    project,
)

RNG = np.random.default_rng(42)


def _make_df(n_channels: int, n_rows: int = 200, base_col: str = "OT") -> pd.DataFrame:
    data = {f"chan_{i}": RNG.normal(size=n_rows) for i in range(max(n_channels - 1, 0))}
    data[base_col] = np.cumsum(RNG.normal(size=n_rows)) + 100
    return pd.DataFrame(data)


@pytest.mark.parametrize("n_channels", [1, 2, 7, 100])
def test_project_reduces_arbitrary_channel_counts_to_two(n_channels):
    df = _make_df(n_channels)
    config = ProjectionConfig(base_reading="OT")
    out = project(df, config)
    assert list(out.columns) == ["change", "volatility"]
    assert len(out) > 0


def test_project_raises_on_missing_base_reading_column():
    df = _make_df(3)
    config = ProjectionConfig(base_reading="does_not_exist")
    with pytest.raises(KeyError):
        project(df, config)


def test_default_volatility_window_is_documented_value():
    assert DEFAULT_VOLATILITY_WINDOW == 20


def test_volatility_window_is_configurable():
    df = _make_df(2, n_rows=200)
    small = project(df, ProjectionConfig(base_reading="OT", volatility_window=5))
    big = project(df, ProjectionConfig(base_reading="OT", volatility_window=50))
    assert len(small) > len(big)  # a bigger window costs more warmup rows


def test_output_never_contains_nan_despite_missing_input_values():
    df = _make_df(2, n_rows=200)
    df.loc[df.index[50], "OT"] = np.nan  # inject a missing reading mid-series
    out = project(df, ProjectionConfig(base_reading="OT"))
    assert not out.isna().any().any()
    assert len(out) > 0


def test_missing_values_in_non_base_reading_columns_are_ignored():
    df = _make_df(5, n_rows=100)
    df["chan_0"] = np.nan  # entirely NaN, but not the base reading
    out = project(df, ProjectionConfig(base_reading="OT"))
    assert not out.isna().any().any()
    assert len(out) > 0


def test_log_return_change_type_matches_log_diff():
    values = np.abs(RNG.normal(loc=100, scale=1, size=100)) + 50
    df = pd.DataFrame({"Adj Close": values})
    config = ProjectionConfig(base_reading="Adj Close", change_type="log_return", volatility_window=5)
    out = project(df, config)
    expected = np.log(df["Adj Close"]).diff().reindex(out.index)
    pd.testing.assert_series_equal(out["change"], expected, check_names=False)


def test_diff_change_type_matches_plain_diff():
    df = pd.DataFrame({"OT": RNG.normal(size=100).cumsum()})
    config = ProjectionConfig(base_reading="OT", change_type="diff", volatility_window=5)
    out = project(df, config)
    expected = df["OT"].diff().reindex(out.index)
    pd.testing.assert_series_equal(out["change"], expected, check_names=False)


def test_unknown_change_type_raises():
    df = _make_df(2)
    config = ProjectionConfig(base_reading="OT", change_type="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        project(df, config)


def test_finance_adapter_uses_log_return_on_adj_close():
    assert FINANCE_ADAPTER.change_type == "log_return"
    assert FINANCE_ADAPTER.base_reading == "Adj Close"


def test_ett_adapter_uses_plain_diff_on_ot():
    assert ETT_ADAPTER.change_type == "diff"
    assert ETT_ADAPTER.base_reading == "OT"


def test_project_is_deterministic():
    df = _make_df(7, n_rows=150)
    config = ProjectionConfig(base_reading="OT")
    out_a = project(df, config)
    out_b = project(df, config)
    pd.testing.assert_frame_equal(out_a, out_b)
