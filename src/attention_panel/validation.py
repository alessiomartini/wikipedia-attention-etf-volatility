"""Walk-forward validation, the HAR baseline, and nested-model comparison.

WHY NUMPY AND NOT SCIKIT-LEARN

Ridge regression has a closed form, Newey-West is thirty lines, and the normal
CDF is `math.erf`. Depending on scikit-learn, scipy and statsmodels for that
would make the core of the study unavailable on any Python new enough that those
three have not shipped wheels yet -- which is exactly the Python this project is
being run on. They stay in the optional `model` extra for anything genuinely
hard; nothing here needs them.

WHAT IS BEING GUARDED AGAINST

Ordinary k-fold shuffles time and lets a model train on the future to predict
the past. Walk-forward fixes that, but on a PANEL it is not enough on its own,
for two reasons that ordinary time-series validation does not face:

1. SPLIT BY DATE, NEVER BY ROW. A panel has many rows per date. Splitting rows
   at random, or even sequentially, puts LVMH's Tuesday in training and
   Kering's Tuesday in test -- and attention shocks are correlated across firms
   on the same day, which is why the standard errors are clustered by date in
   the first place. The same day must be wholly in one side.

2. PURGE THE LABEL OVERLAP. The target on train day `t` is realised over
   `t+1..t+h`. If that window reaches into the test period, the training row saw
   test-period outcomes. The last `h` training days are therefore dropped -- a
   purge in the sense of Lopez de Prado.

WHY THE FEATURES CAN SAFELY BE BUILT BEFORE SPLITTING

Every transform in this project is strictly trailing: HAR components are shifted
rolling means, abnormal attention uses a trailing median and MAD, the weekday
adjustment looks only at past instances of that weekday. A trailing window on a
test row reaches back into training data, which is the past and is allowed; it
never reaches forward. That property is what makes it correct to compute
features on the whole series and split afterwards -- and it is a property of the
feature code, not an assumption, which is why features.py is written the way it
is.

SCALING IS THE EXCEPTION. A scaler fitted before splitting leaks the test set's
mean and variance into training. It is fitted per fold, on the training slice
only, inside `fit_fold`.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fold:
    """One walk-forward split, expressed as dates rather than row positions."""

    index: int
    train_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex

    def describe(self) -> str:
        return (
            f"fold {self.index}: train {self.train_dates.min().date()}"
            f"..{self.train_dates.max().date()} ({len(self.train_dates)}d) -> "
            f"test {self.test_dates.min().date()}..{self.test_dates.max().date()} "
            f"({len(self.test_dates)}d)"
        )


def purged_walk_forward(
    dates: pd.DatetimeIndex,
    n_splits: int = 5,
    horizon: int = 1,
    embargo_days: int = 5,
    min_train_days: int = 250,
) -> list[Fold]:
    """Expanding-window folds over unique dates, with the label overlap purged.

    Args:
        horizon: how many days ahead the target is realised. The final `horizon`
            training days are dropped, because their targets are realised inside
            the test window.
        embargo_days: additional training days dropped before each test window.
            The purge alone handles the target overlap; the embargo covers the
            residual worry that a feature's trailing window ends close enough to
            the boundary for the two periods to share an unusual event.

    Folds are returned only when the training side still has `min_train_days`
    after purging. A fold trained on eighty days of a twenty-two-day HAR
    component is not a fold, it is noise with a confidence interval.
    """
    unique = pd.DatetimeIndex(sorted(pd.DatetimeIndex(dates).unique()))
    if len(unique) < min_train_days + n_splits:
        return []

    # Test blocks of equal length tiling the tail of the sample.
    testable = len(unique) - min_train_days
    block = testable // n_splits
    if block < 1:
        return []

    folds: list[Fold] = []
    for k in range(n_splits):
        test_start = min_train_days + k * block
        test_end = test_start + block if k < n_splits - 1 else len(unique)
        test_dates = unique[test_start:test_end]

        # Purge + embargo: drop the training days whose target windows reach
        # into the test period, and a margin beyond them.
        cut = test_start - horizon - embargo_days
        if cut < min_train_days // 2:
            log.info("fold %d skipped: purge leaves too little training data", k)
            continue
        train_dates = unique[:cut]
        folds.append(Fold(index=k, train_dates=train_dates, test_dates=test_dates))

    return folds


# ---------------------------------------------------------------------------
# Ridge, in closed form
# ---------------------------------------------------------------------------


@dataclass
class RidgeFit:
    """A fitted ridge model, carrying the scaling it was fitted with."""

    coefficients: np.ndarray
    #: The training mean of the target. Because the features are centred, this
    #: is the prediction at the average feature vector -- an intercept in the
    #: standardised space, not a coefficient in the original units.
    y_centre: float
    centre: np.ndarray
    scale: np.ndarray
    columns: list[str] = field(default_factory=list)

    def predict(self, X: np.ndarray) -> np.ndarray:
        standardised = (X - self.centre) / self.scale
        return standardised @ self.coefficients + self.y_centre


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float = 1.0, columns=None) -> RidgeFit:
    """Ridge by the normal equations, with standardisation fitted here.

    The centring and scaling are computed from THIS X only, which is what makes
    it safe to call per fold: a scaler fitted before splitting would carry the
    test set's mean and variance into training, the most common way a
    walk-forward backtest is quietly invalidated.

    The intercept is not penalised -- it is handled by centring `y` rather than
    by adding a column to the design matrix, since shrinking an intercept
    towards zero would bias every prediction toward the origin.
    """
    X = np.asarray(X, dtype="float64")
    y = np.asarray(y, dtype="float64")

    centre = X.mean(axis=0)
    scale = X.std(axis=0)
    # A constant column has zero spread and cannot be standardised. Dividing by
    # one leaves it at zero after centring, so it contributes nothing rather
    # than producing NaN across the whole design matrix.
    scale = np.where(scale > 1e-12, scale, 1.0)
    Z = (X - centre) / scale

    y_mean = y.mean()
    penalty = alpha * np.eye(Z.shape[1])
    coefficients = np.linalg.solve(Z.T @ Z + penalty, Z.T @ (y - y_mean))
    return RidgeFit(coefficients, float(y_mean), centre, scale, list(columns or []))


# ---------------------------------------------------------------------------
# Out-of-sample comparison
# ---------------------------------------------------------------------------


def newey_west_se(series: np.ndarray, lags: int | None = None) -> float:
    """Standard error of a mean under serial correlation (Bartlett kernel).

    The loss differentials compared below are autocorrelated -- volatility
    errors cluster in time -- and a naive standard error would understate the
    uncertainty, turning noise into significance. The default lag follows the
    usual n^(1/3) rule.
    """
    x = np.asarray(series, dtype="float64")
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    if lags is None:
        lags = max(1, int(np.floor(n ** (1 / 3))))

    deviations = x - x.mean()
    variance = float(deviations @ deviations / n)
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - lag / (lags + 1)
        covariance = float(deviations[lag:] @ deviations[:-lag] / n)
        variance += 2.0 * weight * covariance
    if variance <= 0:
        return float("nan")
    return math.sqrt(variance / n)


def _normal_sf(z: float) -> float:
    """One-sided upper-tail probability, without scipy."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def out_of_sample_r2(y: np.ndarray, prediction: np.ndarray, benchmark: np.ndarray) -> float:
    """R^2 of a model against a benchmark, out of sample.

    `1 - SSE(model) / SSE(benchmark)`. Negative means the model is worse than
    the benchmark, which is a perfectly ordinary and informative outcome here:
    the benchmark is HAR, and beating it is the entire claim under test.

    THIS, NOT RAW R^2, IS THE NUMBER THE STUDY REPORTS (DESIGN.md section 5).
    A raw R^2 against a zero model would sit around 0.5 purely because realized
    volatility predicts itself.
    """
    y = np.asarray(y, dtype="float64")
    mask = np.isfinite(y) & np.isfinite(prediction) & np.isfinite(benchmark)
    if mask.sum() < 2:
        return float("nan")
    sse_model = float(((y[mask] - prediction[mask]) ** 2).sum())
    sse_benchmark = float(((y[mask] - benchmark[mask]) ** 2).sum())
    if sse_benchmark <= 0:
        return float("nan")
    return 1.0 - sse_model / sse_benchmark


@dataclass
class TestResult:
    statistic: float
    p_value: float
    note: str = ""


def clark_west(y: np.ndarray, benchmark: np.ndarray, model: np.ndarray) -> TestResult:
    """Clark-West test that a NESTED model beats its benchmark out of sample.

    THIS FUNCTION IS ONLY VALID FOR NESTED FORECASTS, and misusing it does not
    fail quietly -- it fails loudly in the wrong direction. For two INDEPENDENT
    forecasts the adjustment term has expectation `2 * var(y)` rather than
    zero, so the test rejects essentially always. `run_walk_forward` therefore
    checks that the feature sets nest before calling this; a caller invoking it
    directly is responsible for the same check. For non-nested comparisons use
    `diebold_mariano`.

    WHY NOT DIEBOLD-MARIANO HERE. When one model nests the other, the larger
    model estimates extra parameters whose true value may be zero. That
    estimation noise inflates its out-of-sample squared error even under the
    null, so a straight comparison of squared errors is biased AGAINST the
    larger model, and Diebold-Mariano under-rejects. Clark-West adds back the
    adjustment term `(benchmark - model)^2`, which corrects exactly that.

    HAR is nested inside HAR-plus-attention, so this is the right test for the
    study's primary claim. One-sided: the alternative is that adding attention
    helps, and "attention makes the forecast worse" is not a finding anyone
    would act on.
    """
    y = np.asarray(y, dtype="float64")
    benchmark = np.asarray(benchmark, dtype="float64")
    model = np.asarray(model, dtype="float64")
    mask = np.isfinite(y) & np.isfinite(benchmark) & np.isfinite(model)
    if mask.sum() < 30:
        return TestResult(float("nan"), float("nan"), "fewer than 30 comparable points")

    y, benchmark, model = y[mask], benchmark[mask], model[mask]
    adjusted = (y - benchmark) ** 2 - ((y - model) ** 2 - (benchmark - model) ** 2)

    se = newey_west_se(adjusted)
    if not math.isfinite(se) or se <= 0:
        return TestResult(float("nan"), float("nan"), "degenerate variance")

    statistic = float(adjusted.mean() / se)
    return TestResult(statistic, _normal_sf(statistic), "one-sided; H0: attention adds nothing")


def diebold_mariano(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> TestResult:
    """Equal predictive accuracy of two NON-nested forecasts.

    Kept for comparisons the Clark-West correction does not apply to, such as
    two different attention specifications neither of which nests the other.
    Using it on nested models would under-reject; that is what `clark_west` is
    for.
    """
    y = np.asarray(y, dtype="float64")
    mask = np.isfinite(y) & np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 30:
        return TestResult(float("nan"), float("nan"), "fewer than 30 comparable points")

    differential = (y[mask] - a[mask]) ** 2 - (y[mask] - b[mask]) ** 2
    se = newey_west_se(differential)
    if not math.isfinite(se) or se <= 0:
        return TestResult(float("nan"), float("nan"), "degenerate variance")

    statistic = float(differential.mean() / se)
    # Two-sided: neither direction is the designated alternative.
    return TestResult(statistic, 2.0 * _normal_sf(abs(statistic)), "two-sided; H0: equal accuracy")


# ---------------------------------------------------------------------------
# Running a comparison over folds
# ---------------------------------------------------------------------------


@dataclass
class WalkForwardResult:
    """Everything one baseline-versus-model comparison produced."""

    predictions: pd.DataFrame
    folds: list[Fold]
    incremental_r2: float = float("nan")
    baseline_r2_vs_mean: float = float("nan")
    clark_west: TestResult | None = None

    def summary(self) -> str:
        lines = [
            f"{len(self.folds)} folds, {len(self.predictions)} out-of-sample rows",
            f"baseline R^2 vs the sample mean : {self.baseline_r2_vs_mean:+.4f}",
            f"INCREMENTAL R^2 over baseline   : {self.incremental_r2:+.4f}",
        ]
        if self.clark_west:
            lines.append(
                f"Clark-West                      : t={self.clark_west.statistic:+.3f}, "
                f"p={self.clark_west.p_value:.4f}  ({self.clark_west.note})"
            )
        return "\n".join(lines)


def run_walk_forward(
    panel: pd.DataFrame,
    baseline_features: list[str],
    model_features: list[str],
    target: str = "target",
    date_column: str = "date",
    alpha: float = 1.0,
    n_splits: int = 5,
    horizon: int = 1,
    embargo_days: int = 5,
    min_train_days: int = 250,
) -> WalkForwardResult:
    """Fit baseline and model fold by fold, and compare them out of sample.

    `model_features` must be a superset of `baseline_features` for the
    Clark-West test to be the right one -- that is what "nested" means, and the
    function checks rather than assumes it.
    """
    if not set(baseline_features).issubset(model_features):
        raise ValueError(
            "model_features must contain baseline_features: Clark-West applies to "
            "nested models, and reporting it for non-nested ones would overstate "
            "significance. Use diebold_mariano for a non-nested comparison."
        )

    needed = sorted({*baseline_features, *model_features, target, date_column})
    frame = panel[needed].dropna()
    if frame.empty:
        return WalkForwardResult(pd.DataFrame(), [])

    dates = pd.DatetimeIndex(frame[date_column])
    folds = purged_walk_forward(dates, n_splits, horizon, embargo_days, min_train_days)
    if not folds:
        return WalkForwardResult(pd.DataFrame(), [])

    collected = []
    for fold in folds:
        train = frame[frame[date_column].isin(fold.train_dates)]
        test = frame[frame[date_column].isin(fold.test_dates)]
        if train.empty or test.empty:
            continue

        y_train = train[target].to_numpy()
        y_test = test[target].to_numpy()

        baseline = fit_ridge(train[baseline_features].to_numpy(), y_train, alpha, baseline_features)
        model = fit_ridge(train[model_features].to_numpy(), y_train, alpha, model_features)

        collected.append(
            pd.DataFrame(
                {
                    date_column: test[date_column].to_numpy(),
                    "fold": fold.index,
                    "y": y_test,
                    "baseline": baseline.predict(test[baseline_features].to_numpy()),
                    "model": model.predict(test[model_features].to_numpy()),
                    # The training mean is the honest null forecast: it uses no
                    # information beyond the level of the target.
                    "train_mean": float(y_train.mean()),
                }
            )
        )

    if not collected:
        return WalkForwardResult(pd.DataFrame(), folds)

    predictions = pd.concat(collected, ignore_index=True)
    y = predictions["y"].to_numpy()

    return WalkForwardResult(
        predictions=predictions,
        folds=folds,
        incremental_r2=out_of_sample_r2(y, predictions["model"].to_numpy(), predictions["baseline"].to_numpy()),
        baseline_r2_vs_mean=out_of_sample_r2(y, predictions["baseline"].to_numpy(), predictions["train_mean"].to_numpy()),
        clark_west=clark_west(y, predictions["baseline"].to_numpy(), predictions["model"].to_numpy()),
    )
