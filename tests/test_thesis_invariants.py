"""The four failure modes that would invalidate a thesis result, each tested
end to end through the real pipeline (`_pipeline.tokenize_split`) rather than
one module at a time: determinism, leakage, schema conformance, and split
disjointness. Unit tests for each module live in their own files."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from _pipeline import synthetic_ett, synthetic_finance, tokenize_split, tokens_digest
from symbolic_ts.binning import FixedPercentBinner, SigmaBinner
from symbolic_ts.events import ETTEventAdapter
from symbolic_ts.projection import ETT_ADAPTER, FINANCE_ADAPTER
from symbolic_ts.vocabulary import (
    NEUTRAL_CHANGE_LABELS,
    NEUTRAL_TOKENS,
    NEUTRAL_VOLATILITY_LABELS,
    validate_sequence,
)

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent


# --- Determinism -------------------------------------------------------------


def test_identical_input_yields_identical_tokens_across_runs():
    assert tokens_digest() == tokens_digest()


def test_identical_input_yields_identical_tokens_across_processes():
    # Fresh interpreters with different hash seeds: any dependence on set/dict
    # iteration order or other per-process state would change the digest.
    def digest_in_subprocess(hash_seed: str) -> str:
        env = {**os.environ, "PYTHONHASHSEED": hash_seed,
               "PYTHONPATH": os.pathsep.join([str(TESTS_DIR), str(REPO_ROOT)])}
        out = subprocess.run([sys.executable, str(TESTS_DIR / "_pipeline.py")],
                             env=env, capture_output=True, text=True, check=True)
        return out.stdout.strip()

    in_process = tokens_digest()
    assert digest_in_subprocess("0") == in_process
    assert digest_in_subprocess("12345") == in_process


# --- Leakage -----------------------------------------------------------------


@pytest.mark.parametrize("make_raw, config", [(synthetic_finance, FINANCE_ADAPTER), (synthetic_ett, ETT_ADAPTER)])
def test_sigma_fit_on_train_differs_from_sigma_fit_on_full_data(make_raw, config):
    result = tokenize_split(make_raw(), config)
    full = SigmaBinner(labels=NEUTRAL_CHANGE_LABELS).fit(result["projected"]["change"])
    train_only = result["change_binner"]
    assert train_only.std_ != pytest.approx(full.std_, rel=1e-3)
    # and the train fit really is the train slice's own statistic, nothing more
    assert train_only.std_ == pytest.approx(float(result["train"]["change"].std(ddof=0)))


def test_sigma_from_train_excludes_the_more_volatile_future_regime():
    # synthetic_finance doubles its volatility halfway; the last fold's train
    # slice ends before the test lock, so it sees less of the volatile half.
    result = tokenize_split(synthetic_finance(), FINANCE_ADAPTER)
    full = SigmaBinner(labels=NEUTRAL_CHANGE_LABELS).fit(result["projected"]["change"])
    assert result["change_binner"].std_ < full.std_


@pytest.mark.parametrize(
    "binner",
    [SigmaBinner(labels=NEUTRAL_CHANGE_LABELS), SigmaBinner(labels=NEUTRAL_VOLATILITY_LABELS, log_scale=True),
     FixedPercentBinner(labels=NEUTRAL_CHANGE_LABELS)],
    ids=["sigma", "sigma-log", "percent"],
)
def test_transform_never_triggers_a_refit(binner, monkeypatch):
    rng = np.random.default_rng(0)
    binner.fit(np.abs(rng.normal(1.0, 0.2, 500)))
    fitted = {k: v for k, v in vars(binner).items() if k.endswith("_")}

    def refit_forbidden(*args, **kwargs):
        raise AssertionError("transform() called fit()")

    monkeypatch.setattr(type(binner), "fit", refit_forbidden)
    first = binner.transform(np.abs(rng.normal(50.0, 30.0, 500)))  # wildly different data
    second = binner.transform(np.abs(rng.normal(50.0, 30.0, 500)))

    assert {k: v for k, v in vars(binner).items() if k.endswith("_")} == fitted
    assert len(first) == len(second) == 500


def test_a_reloaded_calibration_reproduces_tokens_without_fitting(tmp_path):
    result = tokenize_split(synthetic_finance(), FINANCE_ADAPTER)
    result["change_binner"].save(tmp_path / "change.json")
    reloaded = SigmaBinner.load(tmp_path / "change.json")
    validation_change = result["validation"]["change"]
    assert reloaded.transform(validation_change) == result["change_binner"].transform(validation_change)


# --- Schema conformance ------------------------------------------------------


@pytest.mark.parametrize("make_raw, config", [(synthetic_finance, FINANCE_ADAPTER), (synthetic_ett, ETT_ADAPTER)])
def test_every_emitted_token_validates(make_raw, config):
    result = tokenize_split(make_raw(), config)
    report = validate_sequence(result["tokens"], allowed=NEUTRAL_TOKENS)
    assert report.is_valid, report.invalid_tokens[:5]
    assert report.total == len(result["validation"])


def test_extreme_values_still_map_to_valid_tokens():
    # spikes in synthetic_ett land far beyond +-1.5 sigma; flat stretches give
    # zero volatility -- both must still produce in-vocabulary tokens
    result = tokenize_split(synthetic_ett(), ETT_ADAPTER, test_fraction=0.1, n_folds=1)
    projected = result["projected"]
    change = result["change_binner"].transform(projected["change"])
    volatility = result["volatility_binner"].transform(projected["volatility"])
    tokens = [f"{c}_{v}" for c, v in zip(change, volatility)]
    assert validate_sequence(tokens, allowed=NEUTRAL_TOKENS).is_valid
    assert {"C0", "C4"} <= set(change)  # the extremes really were exercised
    assert "V0" in set(volatility)


def test_event_tokens_interleaved_with_value_tokens_validate():
    result = tokenize_split(synthetic_ett(), ETT_ADAPTER)
    adapter = ETTEventAdapter(peak_hours=frozenset({18, 19, 20}))
    sequence = []
    for ts, token in zip(result["validation"].index, result["tokens"]):
        sequence.append(token)
        sequence.extend(adapter.events_for(ts))
    report = validate_sequence(sequence)
    assert report.is_valid, report.invalid_tokens[:5]


@pytest.mark.parametrize("binner_cls", [SigmaBinner, FixedPercentBinner])
def test_nan_is_refused_instead_of_becoming_the_top_bucket(binner_cls):
    binner = binner_cls(labels=NEUTRAL_CHANGE_LABELS).fit([1.0, 2.0, 3.0, 4.0, 5.0])
    with pytest.raises(ValueError, match="NaN"):
        binner.transform([3.0, float("nan")])
    with pytest.raises(ValueError, match="NaN"):
        binner_cls(labels=NEUTRAL_CHANGE_LABELS).fit([1.0, float("nan"), 3.0])


def test_infinities_keep_their_correct_extreme_bucket():
    binner = SigmaBinner(labels=NEUTRAL_CHANGE_LABELS).fit([1.0, 2.0, 3.0, 4.0, 5.0])
    assert binner.transform([float("-inf"), float("inf")]) == ["C0", "C4"]


def test_log_scale_refuses_negative_input():
    binner = SigmaBinner(labels=NEUTRAL_VOLATILITY_LABELS, log_scale=True).fit([0.5, 1.0, 1.5])
    with pytest.raises(ValueError, match="non-negative"):
        binner.transform([1.0, -0.1])


# --- Split disjointness ------------------------------------------------------


@pytest.mark.parametrize("n_folds", [1, 3, 5])
def test_train_validation_test_are_pairwise_disjoint_after_purging(n_folds):
    result = tokenize_split(synthetic_finance(n=1500), FINANCE_ADAPTER, n_folds=n_folds)
    test = set(result["lock"].indices.tolist())
    assert len(result["folds"]) == n_folds
    for fold in result["folds"]:
        train, validation = set(fold.train_indices.tolist()), set(fold.validation_indices.tolist())
        assert not train & validation
        assert not train & test
        assert not validation & test
        # purge: the last training target sits at least a context window before validation
        assert fold.validation_indices.min() - fold.train_indices.max() > 50
        # chronology: everything the model learns from or is tuned on precedes the test split
        assert max(fold.validation_indices.max(), fold.train_indices.max()) < min(test)
