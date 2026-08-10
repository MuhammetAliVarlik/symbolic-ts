"""Every metric and statistical test this thesis reports, in one place.

RMSE alone can't evaluate a symbolic classification task, and neither
"looked better on a plot" nor an unpaired significance test is good enough
for a headline transfer claim -- this module covers point-forecast metrics,
classification metrics, an interval estimator, and the two paired
significance tests the literature in this area actually uses (McNemar for
classification comparisons, Diebold-Mariano for forecast-accuracy
comparisons), plus the multiple-comparison correction needed once more than
one pair gets compared.

Every implementation below is checked against a worked numeric example with
a cited source in its own docstring -- most from an external, independently
publishable reference (Wikipedia's cited primary sources, NIST's Statistical
Engineering Handbook, scikit-learn's documented examples); a few
(directional accuracy, the multi-class F1 example) are hand-constructed and
verified by direct arithmetic here rather than lifted from an external
citation, and are labelled as such rather than misattributed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt
from scipy import stats

# ---------------------------------------------------------------------------
# Point metrics
# ---------------------------------------------------------------------------


def rmse(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """Root mean squared error. Definition: Hyndman & Koehler (2006),
    "Another look at measures of forecast accuracy," Int. J. Forecasting.

    Worked example (scikit-learn's `mean_squared_error` documentation,
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html):
    y_true=[3, -0.5, 2, 7], y_pred=[2.5, 0.0, 2, 8] -> MSE=0.375, so
    RMSE=sqrt(0.375)=0.6123724...
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """Mean absolute error. Definition: Hyndman & Koehler (2006), as above.

    Worked example (scikit-learn's `mean_absolute_error` documentation,
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html):
    y_true=[3, -0.5, 2, 7], y_pred=[2.5, 0.0, 2, 8] -> MAE=0.5.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def directional_accuracy(true_change: npt.ArrayLike, pred_change: npt.ArrayLike) -> float:
    """Fraction of timesteps where the predicted change has the same sign as
    the true change (both operate on already-computed `change` values, e.g.
    from `projection.py` -- not raw levels, so there is no ambiguity about
    which prior level a "direction" is measured against). Zero counts as its
    own sign: a true or predicted change of exactly 0 only matches another
    exact 0.

    This is a hand-constructed, hand-verified example (not lifted from an
    external publication -- it is a direct sign-match rate, not a procedure
    with room for a subtle implementation bug): true=[1,-2,3,-4,0],
    pred=[2,-1,-3,-4,0] -> signs [+,-,+,-,0] vs [+,-,-,-,0] -> agree at
    indices 0,1,3,4 -> 4/5 = 0.8.
    """
    true_sign = np.sign(np.asarray(true_change, dtype=float))
    pred_sign = np.sign(np.asarray(pred_change, dtype=float))
    return float(np.mean(true_sign == pred_sign))


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------


def confusion_matrix(
    y_true: Sequence, y_pred: Sequence, labels: Sequence | None = None
) -> tuple[np.ndarray, tuple]:
    """Row-normalised... no -- raw counts. Rows are true labels, columns are
    predicted labels, `matrix[i, j]` = number of samples with true label
    `labels[i]` predicted as `labels[j]`. `labels` defaults to the sorted
    union of values seen in `y_true`/`y_pred`."""
    y_true = list(y_true)
    y_pred = list(y_pred)
    if labels is None:
        labels = tuple(sorted(set(y_true) | set(y_pred)))
    else:
        labels = tuple(labels)
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[index[t], index[p]] += 1
    return matrix, labels


def accuracy(y_true: Sequence, y_pred: Sequence) -> float:
    """Overall accuracy: fraction of exact matches.

    Worked example: see the shared confusion matrix in `per_class_f1`'s
    docstring -- accuracy = (3 + 4 + 3) / 14 = 10/14 = 0.7142857...
    """
    matrix, _ = confusion_matrix(y_true, y_pred)
    return float(np.trace(matrix) / matrix.sum())


def per_class_f1(y_true: Sequence, y_pred: Sequence, labels: Sequence | None = None) -> dict:
    """Per-class F1 (one-vs-rest precision/recall from the confusion matrix).

    Hand-constructed, hand-verified worked example (3 classes A, B, C):

        true = [A,A,A,A, B,B,B,B,B, C,C,C,C,C]
        pred = [A,A,A,B, B,B,B,B,C, A,A,C,C,C]

    gives confusion matrix (rows=true, cols=pred, order A,B,C):
        [[3, 1, 0],
         [0, 4, 1],
         [2, 0, 3]]

    Class A: TP=3, predicted-A total=5 -> FP=2, true-A total=4 -> FN=1;
      precision=3/5=0.6, recall=3/4=0.75, F1=2*0.6*0.75/1.35=0.6666...
    Class B: TP=4, predicted-B total=5 -> FP=1, true-B total=5 -> FN=1;
      precision=recall=0.8, F1=0.8
    Class C: TP=3, predicted-C total=4 -> FP=1, true-C total=5 -> FN=2;
      precision=0.75, recall=0.6, F1=2*0.75*0.6/1.35=0.6666...
    """
    matrix, resolved_labels = confusion_matrix(y_true, y_pred, labels)
    result = {}
    for i, label in enumerate(resolved_labels):
        tp = matrix[i, i]
        fp = matrix[:, i].sum() - tp
        fn = matrix[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        result[label] = float(f1)
    return result


def macro_f1(y_true: Sequence, y_pred: Sequence, labels: Sequence | None = None) -> float:
    """Unweighted mean of `per_class_f1` -- every class counts equally
    regardless of support, so a model that ignores a rare class is punished
    even if overall accuracy looks fine.

    Worked example: the same confusion matrix as `per_class_f1` ->
    macro F1 = (0.6666... + 0.8 + 0.6666...) / 3 = 32/45 = 0.71111...
    """
    per_class = per_class_f1(y_true, y_pred, labels)
    return float(np.mean(list(per_class.values())))


def mcc(y_true: Sequence, y_pred: Sequence, labels: Sequence | None = None) -> float:
    """Matthews correlation coefficient, multi-class generalisation
    (Gorodkin, 2004, "Comparing two K-category assignments by a K-category
    correlation coefficient," Computational Biology and Chemistry). Reduces
    exactly to the classic binary MCC for 2 classes.

    Worked example (binary case; Wikipedia's Matthews correlation
    coefficient article, cat-vs-dog classifier): TP=6, TN=3, FP=1, FN=2 ->
    MCC = (6*3 - 1*2) / sqrt(7*8*4*5) = 16 / sqrt(1120) = 0.4780914...
    Confusion matrix here (rows=true [pos,neg], cols=pred [pos,neg]):
    [[TP, FN], [FP, TN]] = [[6, 2], [1, 3]].
    """
    matrix, _ = confusion_matrix(y_true, y_pred, labels)
    matrix = matrix.astype(float)
    s = matrix.sum()
    c = np.trace(matrix)
    predicted_totals = matrix.sum(axis=0)  # p_k
    true_totals = matrix.sum(axis=1)  # t_k

    numerator = c * s - np.dot(predicted_totals, true_totals)
    denominator = np.sqrt((s**2 - np.dot(predicted_totals, predicted_totals)) * (s**2 - np.dot(true_totals, true_totals)))
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


# ---------------------------------------------------------------------------
# Intervals
# ---------------------------------------------------------------------------


def wilson_score_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion --
    substantially better small-sample and near-0/1 coverage than the naive
    normal (Wald) interval. Formula: Wilson (1927), "Probable inference, the
    law of succession, and statistical inference," J. American Statistical
    Association.

    Worked example (Bartosz Mikulski, "Wilson score in Python - example,"
    https://mikulskibartosz.name/wilson-score-in-python-example, cross-checked
    against the `wilson-score-interval` reference implementation cited
    there): k=40 successes, n=100, 95% confidence (z=1.96) ->
    interval approximately [0.3094, 0.4980].
    """
    if not 0 <= successes <= n:
        raise ValueError(f"successes must be in [0, n], got successes={successes}, n={n}")
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    z = stats.norm.ppf(0.5 + confidence / 2)
    p_hat = successes / n
    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denominator
    margin = (z / denominator) * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
    return float(center - margin), float(center + margin)


# ---------------------------------------------------------------------------
# Paired tests
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class McNemarResult:
    statistic: float | None
    p_value: float
    exact: bool


def mcnemar_test(
    table: npt.ArrayLike, exact: bool | None = None, continuity_correction: bool = True
) -> McNemarResult:
    """McNemar's test for paired nominal data: are two classifiers' errors
    on the SAME test items distributed differently? Only the discordant
    cells matter -- `table` is `[[a, b], [c, d]]` where `b`/`c` are the
    off-diagonal (disagreement) counts.

    `exact=None` (default) follows the standard convention -- use the exact
    binomial test when `b + c < 25`, chi-square otherwise (Wikipedia's
    McNemar's test article; see also Fagerland, Lydersen & Laake (2013),
    "The McNemar test for binary matched-pairs data," BMC Medical Research
    Methodology, for when the exact test is preferable).

    Worked example, UNCORRECTED chi-square (Wikipedia's McNemar's test
    article, 314-patient before/after example -- note this cited example
    uses the plain formula, not continuity-corrected): b=121, c=59 ->
    chi2 = (121-59)^2 / (121+59) = 3844/180 = 21.3555..., p < 0.001
    (`continuity_correction=False` reproduces this exactly). The
    continuity-corrected value for the same table, (|121-59|-1)^2/180 =
    20.6722..., is hand-verified here rather than lifted from that
    citation, since the source doesn't give the corrected number for this
    example.

    The exact variant has no external worked example cited here; it is
    cross-checked in this module's tests directly against the binomial PMF
    it is defined from (`scipy.stats.binomtest`), which is the standard
    implementation of the same textbook formula.
    """
    table = np.asarray(table, dtype=float)
    b, c = table[0, 1], table[1, 0]
    n_discordant = b + c
    if exact is None:
        exact = n_discordant < 25

    if exact:
        k = int(min(b, c))
        result = stats.binomtest(k, int(n_discordant), 0.5, alternative="two-sided")
        return McNemarResult(statistic=None, p_value=float(result.pvalue), exact=True)

    if continuity_correction:
        statistic = (abs(b - c) - 1) ** 2 / n_discordant
    else:
        statistic = (b - c) ** 2 / n_discordant
    p_value = float(stats.chi2.sf(statistic, df=1))
    return McNemarResult(statistic=float(statistic), p_value=p_value, exact=False)


@dataclass(frozen=True)
class DieboldMarianoResult:
    statistic: float
    p_value: float


def diebold_mariano_test(
    errors_a: npt.ArrayLike, errors_b: npt.ArrayLike, h: int = 1, loss: str = "squared", correction: bool = True
) -> DieboldMarianoResult:
    """Diebold-Mariano test for equal forecast accuracy (Diebold & Mariano,
    1995, "Comparing predictive accuracy," J. Business & Economic
    Statistics), with the small-sample correction of Harvey, Leybourne &
    Newbold (1997), "Testing the equality of prediction mean squared
    errors," Int. J. Forecasting -- applied by default (`correction=True`),
    since the uncorrected test over-rejects in finite samples.

    The loss differential is `d_t = L(errors_a[t]) - L(errors_b[t])` with
    `L` squared or absolute error (`loss="squared"`/`"absolute"`). For
    `h`-step-ahead forecasts, `d_t` can be autocorrelated up to lag `h - 1`
    under the null, so the long-run variance includes those autocovariance
    terms (Newey-West-style, unweighted, matching Diebold & Mariano's
    original specification); for `h=1` (this project's case: one-step
    change/volatility forecasts) it reduces to the plain sample variance,
    and the whole statistic reduces to a one-sample t-statistic on `d_t`.

    That `h=1` reduction is what's validated here, against NIST's
    Statistical Engineering Handbook one-sample t-test worked example
    (https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm,
    ZARR13.DAT): a sample of N=195 with mean=9.261460, standard
    deviation=0.022789, tested against mu=5, gives t=2611.284. This module's
    test suite constructs a synthetic loss-differential array with exactly
    that sample mean and standard deviation (via an affine rescaling of any
    base sample) and confirms the same t-statistic comes out of the h=1
    formula below. The general h>1 long-run-variance case is implemented
    per the cited papers but has no independent published worked example
    checked against it here.
    """
    errors_a = np.asarray(errors_a, dtype=float)
    errors_b = np.asarray(errors_b, dtype=float)
    if loss == "squared":
        loss_a, loss_b = errors_a**2, errors_b**2
    elif loss == "absolute":
        loss_a, loss_b = np.abs(errors_a), np.abs(errors_b)
    else:
        raise ValueError(f"unknown loss {loss!r}, expected 'squared' or 'absolute'")

    d = loss_a - loss_b
    t = len(d)
    d_mean = d.mean()

    autocovariance_sum = 0.0
    for lag in range(1, h):
        cov = np.mean((d[lag:] - d_mean) * (d[:-lag] - d_mean))
        autocovariance_sum += 2 * cov
    long_run_variance = d.var(ddof=0) + autocovariance_sum
    long_run_variance = max(long_run_variance, 0.0)

    if long_run_variance == 0.0:
        # No variance in the loss differential -- the two forecasts are
        # either identical (d_mean == 0, no evidence of a difference) or
        # differ by an exact constant (perfect, degenerate "significance").
        statistic = 0.0 if d_mean == 0.0 else np.sign(d_mean) * np.inf
    else:
        statistic = d_mean / np.sqrt(long_run_variance / t)

    if correction:
        adjustment = np.sqrt((t + 1 - 2 * h + h * (h - 1) / t) / t)
        statistic = statistic * adjustment
        p_value = float(2 * stats.t.sf(abs(statistic), df=t - 1))
    else:
        p_value = float(2 * stats.norm.sf(abs(statistic)))

    return DieboldMarianoResult(statistic=float(statistic), p_value=p_value)


# ---------------------------------------------------------------------------
# Multiple-comparison correction
# ---------------------------------------------------------------------------


def holm_correction(p_values: npt.ArrayLike, alpha: float = 0.05) -> np.ndarray:
    """Holm-Bonferroni step-down procedure: controls the family-wise error
    rate without the Bonferroni correction's full conservatism. Returns a
    boolean array, in the SAME order as the input, of which hypotheses are
    rejected at level `alpha`.

    Algorithm (Holm, 1979, "A simple sequentially rejective multiple test
    procedure," Scandinavian Journal of Statistics): sort p-values
    ascending; reject p_(k) if p_(k) <= alpha / (m + 1 - k) AND every
    earlier (smaller) p-value was also rejected; stop at the first failure.

    Worked example (Wikipedia's Holm-Bonferroni method article): p =
    [0.01, 0.04, 0.03, 0.005], alpha=0.05, m=4. Sorted: 0.005, 0.01, 0.03,
    0.04. Thresholds: alpha/4=0.0125, alpha/3=0.01667, alpha/2=0.025,
    alpha/1=0.05. 0.005<=0.0125 reject; 0.01<=0.01667 reject;
    0.03<=0.025 fails -> stop. Result: reject p1 (0.01) and p4 (0.005) only.
    """
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    order = np.argsort(p_values)
    reject_sorted = np.zeros(m, dtype=bool)
    for rank, idx in enumerate(order):
        threshold = alpha / (m - rank)
        if p_values[idx] <= threshold:
            reject_sorted[rank] = True
        else:
            break
    reject = np.zeros(m, dtype=bool)
    reject[order] = reject_sorted
    return reject
