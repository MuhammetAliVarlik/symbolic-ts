import json

import numpy as np
import pytest

from symbolic_ts.binning import (
    DEFAULT_SIGMA_EDGES,
    FixedPercentBinner,
    NotFittedError,
    SigmaBinner,
)
from symbolic_ts.vocabulary import NEUTRAL_CHANGE_LABELS, SCHEMA_VERSION

RNG = np.random.default_rng(42)


def _fresh_binner(**kwargs):
    return SigmaBinner(labels=NEUTRAL_CHANGE_LABELS, **kwargs)


def test_post_init_validates_label_count():
    with pytest.raises(ValueError):
        SigmaBinner(labels=("A", "B"))  # 2 labels, needs len(sigma_edges)+1 == 5


def test_transform_raises_before_fit():
    binner = _fresh_binner()
    with pytest.raises(NotFittedError):
        binner.transform([0.0, 1.0])


def test_save_raises_before_fit(tmp_path):
    binner = _fresh_binner()
    with pytest.raises(NotFittedError):
        binner.save(tmp_path / "calib.json")


def test_fit_then_transform_produces_known_labels():
    train = RNG.normal(loc=0.0, scale=1.0, size=1000)
    binner = _fresh_binner().fit(train)
    tokens = binner.transform([0.0, -10.0, 10.0])
    assert tokens[0] == "C2"  # near mean -> middle bucket
    assert tokens[1] == "C0"  # far below mean -> lowest bucket
    assert tokens[2] == "C4"  # far above mean -> highest bucket


def test_sigma_differs_when_fit_on_train_vs_full_data():
    """Leakage guard: fitting on a train-only slice must NOT reproduce the
    same mean/std as fitting on the full series, when the held-out portion
    has different statistics. This is the direct analogue of F0-04's
    original leakage-guard finding.
    """
    train_only = RNG.normal(loc=0.0, scale=1.0, size=700)
    held_out = RNG.normal(loc=5.0, scale=3.0, size=300)
    full = np.concatenate([train_only, held_out])

    binner_train = _fresh_binner().fit(train_only)
    binner_full = _fresh_binner().fit(full)

    assert binner_train.mean_ != pytest.approx(binner_full.mean_)
    assert binner_train.std_ != pytest.approx(binner_full.std_)


def test_transform_is_deterministic_across_repeated_calls():
    train = RNG.normal(size=500)
    data = RNG.normal(size=200)
    binner = _fresh_binner().fit(train)
    first = binner.transform(data)
    second = binner.transform(data)
    third = binner.transform(data)
    assert first == second == third


def test_log_scale_excludes_near_zero_from_fit_but_not_transform():
    """Mirrors the F0-04 fix: near-zero volatility windows must be excluded
    from the FIT (they'd otherwise blow up the log-scale variance), but must
    still be classifiable at transform time (lowest bucket), not raise.
    """
    nonzero = np.abs(RNG.normal(loc=1.0, scale=0.3, size=980)) + 0.1
    near_zero = np.zeros(20)
    train_with_zeros = np.concatenate([nonzero, near_zero])
    train_without_zeros = nonzero

    binner_with = SigmaBinner(labels=NEUTRAL_CHANGE_LABELS, log_scale=True).fit(train_with_zeros)
    binner_without = SigmaBinner(labels=NEUTRAL_CHANGE_LABELS, log_scale=True).fit(train_without_zeros)

    # excluding the near-zero windows from the fit should leave mean/std unchanged
    assert binner_with.mean_ == pytest.approx(binner_without.mean_)
    assert binner_with.std_ == pytest.approx(binner_without.std_)

    # but transform must still classify a zero without raising, landing in the lowest bucket
    tokens = binner_with.transform([0.0])
    assert tokens[0] == "C0"


def test_log_scale_inflates_std_when_zeros_are_not_excluded_from_fit():
    """Companion negative check: without exclusion (zero_tolerance=0 and no
    filtering), a naive implementation would let zeros dominate the log-scale
    variance. This test locks in that the exclusion path is actually doing
    something, by comparing against manually including the zeros unfiltered.
    """
    nonzero = np.abs(RNG.normal(loc=1.0, scale=0.3, size=980)) + 0.1
    near_zero = np.zeros(20)
    train_with_zeros = np.concatenate([nonzero, near_zero])

    binner_excludes = SigmaBinner(labels=NEUTRAL_CHANGE_LABELS, log_scale=True).fit(train_with_zeros)

    naive_log = np.log(train_with_zeros + 1e-8)
    naive_std = float(naive_log.std())

    assert binner_excludes.std_ < naive_std


def test_save_load_round_trip_sigma_binner(tmp_path):
    train = RNG.normal(size=500)
    binner = _fresh_binner(log_scale=False).fit(train)
    path = tmp_path / "sigma_calib.json"
    binner.save(path)

    loaded = SigmaBinner.load(path)
    assert loaded.mean_ == pytest.approx(binner.mean_)
    assert loaded.std_ == pytest.approx(binner.std_)
    assert loaded.labels == binner.labels
    assert loaded.sigma_edges == binner.sigma_edges

    data = RNG.normal(size=50)
    assert loaded.transform(data) == binner.transform(data)


def test_save_writes_schema_version(tmp_path):
    train = RNG.normal(size=200)
    binner = _fresh_binner().fit(train)
    path = tmp_path / "calib.json"
    binner.save(path)
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["binner_type"] == "SigmaBinner"


def test_default_sigma_edges_match_d1():
    assert DEFAULT_SIGMA_EDGES == (-1.5, -0.5, 0.5, 1.5)


def test_load_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        SigmaBinner.load(tmp_path / "does_not_exist.json")


# ---- FixedPercentBinner -----------------------------------------------


def _fresh_percent_binner(**kwargs):
    return FixedPercentBinner(labels=NEUTRAL_CHANGE_LABELS, **kwargs)


def test_percent_binner_post_init_validates_label_count():
    with pytest.raises(ValueError):
        FixedPercentBinner(labels=("A", "B", "C"))  # needs len(quantiles)+1 == 5


def test_percent_binner_transform_raises_before_fit():
    binner = _fresh_percent_binner()
    with pytest.raises(NotFittedError):
        binner.transform([0.0])


def test_percent_binner_fit_transform_matches_quantile_edges():
    train = RNG.uniform(0, 100, size=1000)
    binner = _fresh_percent_binner().fit(train)
    expected_edges = tuple(float(x) for x in np.quantile(train, (0.2, 0.4, 0.6, 0.8)))
    assert binner.edges_ == pytest.approx(expected_edges)

    below_all = binner.transform([expected_edges[0] - 1])
    above_all = binner.transform([expected_edges[-1] + 1])
    assert below_all[0] == "C0"
    assert above_all[0] == "C4"


def test_percent_binner_is_deterministic():
    train = RNG.uniform(0, 100, size=500)
    data = RNG.uniform(0, 100, size=100)
    binner = _fresh_percent_binner().fit(train)
    assert binner.transform(data) == binner.transform(data)


def test_percent_binner_save_load_round_trip(tmp_path):
    train = RNG.uniform(0, 100, size=500)
    binner = _fresh_percent_binner().fit(train)
    path = tmp_path / "percent_calib.json"
    binner.save(path)

    loaded = FixedPercentBinner.load(path)
    assert loaded.edges_ == binner.edges_
    data = RNG.uniform(0, 100, size=50)
    assert loaded.transform(data) == binner.transform(data)


def test_percent_binner_save_writes_schema_version(tmp_path):
    train = RNG.uniform(0, 100, size=200)
    binner = _fresh_percent_binner().fit(train)
    path = tmp_path / "calib.json"
    binner.save(path)
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["binner_type"] == "FixedPercentBinner"
