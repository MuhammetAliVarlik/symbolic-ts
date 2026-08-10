import numpy as np
import pytest

from symbolic_ts import metrics as m


def test_rmse_matches_sklearn_documented_example():
    # sklearn.metrics.mean_squared_error docs: MSE=0.375 -> RMSE=sqrt(0.375)
    assert m.rmse([3, -0.5, 2, 7], [2.5, 0.0, 2, 8]) == pytest.approx(0.6123724356957945)


def test_mae_matches_sklearn_documented_example():
    # sklearn.metrics.mean_absolute_error docs: MAE=0.5
    assert m.mae([3, -0.5, 2, 7], [2.5, 0.0, 2, 8]) == pytest.approx(0.5)


def test_directional_accuracy_hand_computed_example():
    true_change = [1, -2, 3, -4, 0]
    pred_change = [2, -1, -3, -4, 0]
    assert m.directional_accuracy(true_change, pred_change) == pytest.approx(0.8)


def test_directional_accuracy_zero_only_matches_exact_zero():
    assert m.directional_accuracy([0], [0.001]) == pytest.approx(0.0)
    assert m.directional_accuracy([0], [0]) == pytest.approx(1.0)


TRUE_3CLASS = ["A"] * 4 + ["B"] * 5 + ["C"] * 5
PRED_3CLASS = ["A", "A", "A", "B"] + ["B", "B", "B", "B", "C"] + ["A", "A", "C", "C", "C"]


def test_confusion_matrix_matches_hand_built_example():
    matrix, labels = m.confusion_matrix(TRUE_3CLASS, PRED_3CLASS)
    assert labels == ("A", "B", "C")
    assert matrix.tolist() == [[3, 1, 0], [0, 4, 1], [2, 0, 3]]


def test_accuracy_matches_hand_computed_example():
    assert m.accuracy(TRUE_3CLASS, PRED_3CLASS) == pytest.approx(10 / 14)


def test_per_class_f1_matches_hand_computed_example():
    result = m.per_class_f1(TRUE_3CLASS, PRED_3CLASS)
    assert result["A"] == pytest.approx(2 / 3)
    assert result["B"] == pytest.approx(0.8)
    assert result["C"] == pytest.approx(2 / 3)


def test_macro_f1_matches_hand_computed_example():
    assert m.macro_f1(TRUE_3CLASS, PRED_3CLASS) == pytest.approx(32 / 45)


def test_mcc_matches_wikipedia_cat_dog_example():
    # TP=6, TN=3, FP=1, FN=2 -> MCC = 16/sqrt(1120) = 0.4780914...
    y_true = ["pos"] * 6 + ["pos"] * 2 + ["neg"] * 1 + ["neg"] * 3
    y_pred = ["pos"] * 6 + ["neg"] * 2 + ["pos"] * 1 + ["neg"] * 3
    assert m.mcc(y_true, y_pred, labels=["pos", "neg"]) == pytest.approx(16 / np.sqrt(1120))


def test_mcc_perfect_classifier_is_one():
    y_true = ["A", "B", "A", "B"]
    assert m.mcc(y_true, y_true) == pytest.approx(1.0)


def test_wilson_score_interval_matches_published_example():
    # Mikulski's worked example: k=40, n=100, 95% -> [0.3094, 0.4980]
    lower, upper = m.wilson_score_interval(40, 100, confidence=0.95)
    assert lower == pytest.approx(0.3094, abs=1e-3)
    assert upper == pytest.approx(0.4980, abs=1e-3)


def test_wilson_score_interval_contains_point_estimate():
    lower, upper = m.wilson_score_interval(50, 100, confidence=0.95)
    assert lower < 0.5 < upper


def test_wilson_score_interval_rejects_invalid_input():
    with pytest.raises(ValueError):
        m.wilson_score_interval(150, 100)
    with pytest.raises(ValueError):
        m.wilson_score_interval(5, 0)


def test_mcnemar_uncorrected_matches_wikipedia_314_patient_example():
    result = m.mcnemar_test([[101, 121], [59, 33]], exact=False, continuity_correction=False)
    assert result.statistic == pytest.approx(21.355555555555554)
    assert result.p_value < 0.001


def test_mcnemar_continuity_corrected_hand_verified():
    result = m.mcnemar_test([[101, 121], [59, 33]], exact=False, continuity_correction=True)
    assert result.statistic == pytest.approx((abs(121 - 59) - 1) ** 2 / (121 + 59))
    assert result.statistic == pytest.approx(20.67222222222222)


def test_mcnemar_exact_matches_independent_binomial_pmf_cross_check():
    from scipy.stats import binomtest

    table = [[50, 2], [8, 40]]  # b=2, c=8, b+c=10 < 25 -> exact path
    result = m.mcnemar_test(table)
    assert result.exact is True
    expected = binomtest(2, 10, 0.5, alternative="two-sided").pvalue
    assert result.p_value == pytest.approx(expected)


def test_mcnemar_auto_selects_exact_below_25_discordant_pairs():
    below_threshold = m.mcnemar_test([[10, 5], [3, 2]])  # b+c=8 < 25
    assert below_threshold.exact is True
    above_threshold = m.mcnemar_test([[10, 20], [10, 2]])  # b+c=30 >= 25
    assert above_threshold.exact is False


def test_diebold_mariano_h1_reduces_to_nist_one_sample_t_test_example():
    # NIST e-Handbook ZARR13.DAT one-sample t-test: N=195, mean=9.261460,
    # std=0.022789, tested against mu=5 -> t=2611.284, df=194.
    rng = np.random.default_rng(0)
    n = 195
    base = rng.normal(size=n)
    x = (base - base.mean()) / base.std(ddof=1) * 0.022789 + 9.261460
    d = x - 5.0

    result = m.diebold_mariano_test(np.sqrt(d), np.zeros(n), h=1, loss="squared", correction=True)
    assert result.statistic == pytest.approx(2611.284, abs=0.5)
    assert result.p_value < 1e-6


def test_diebold_mariano_identical_forecasts_gives_zero_statistic():
    errors = np.array([1.0, -2.0, 3.0, -1.5, 2.5])
    result = m.diebold_mariano_test(errors, errors, h=1)
    assert result.statistic == pytest.approx(0.0)
    assert result.p_value == pytest.approx(1.0)


def test_diebold_mariano_rejects_unknown_loss():
    with pytest.raises(ValueError):
        m.diebold_mariano_test([1.0], [1.0], loss="huber")


def test_holm_correction_matches_wikipedia_four_hypothesis_example():
    p_values = [0.01, 0.04, 0.03, 0.005]
    reject = m.holm_correction(p_values, alpha=0.05)
    assert reject.tolist() == [True, False, False, True]


def test_holm_correction_all_reject_when_all_p_values_tiny():
    reject = m.holm_correction([1e-10, 1e-9, 1e-8], alpha=0.05)
    assert reject.tolist() == [True, True, True]


def test_holm_correction_none_reject_when_all_p_values_large():
    reject = m.holm_correction([0.9, 0.8, 0.7], alpha=0.05)
    assert reject.tolist() == [False, False, False]
