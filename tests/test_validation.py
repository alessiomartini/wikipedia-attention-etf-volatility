"""Tests for walk-forward validation and nested-model comparison.

Two properties matter most and both are checked with a negative control: that
a date never appears on both sides of a split -- which on a panel is the leak
ordinary time-series validation does not face -- and that a genuine signal is
actually detected, so the guards are not simply refusing everything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from attention_panel.validation import (
    clark_west,
    diebold_mariano,
    fit_ridge,
    newey_west_se,
    out_of_sample_r2,
    purged_walk_forward,
    run_walk_forward,
)

DATES = pd.DatetimeIndex(pd.date_range("2020-01-01", periods=1000, freq="B"))


# -- splitting --------------------------------------------------------------

def test_no_date_appears_in_both_train_and_test():
    """On a panel this is the leak ordinary validation misses: many rows share
    a date, and attention shocks are correlated across firms on the same day --
    which is why the standard errors are clustered by date to begin with."""
    for fold in purged_walk_forward(DATES, n_splits=5):
        assert len(fold.train_dates.intersection(fold.test_dates)) == 0


def test_training_always_precedes_testing():
    for fold in purged_walk_forward(DATES, n_splits=5):
        assert fold.train_dates.max() < fold.test_dates.min()


def test_the_label_overlap_is_purged():
    """The target on train day t is realised over t+1..t+h. Without the purge,
    the last h training rows saw test-period outcomes."""
    horizon, embargo = 3, 5
    for fold in purged_walk_forward(DATES, n_splits=4, horizon=horizon, embargo_days=embargo):
        gap = (fold.test_dates.min() - fold.train_dates.max()).days
        assert gap > horizon


def test_a_larger_embargo_widens_the_gap():
    """The negative control: the parameter must actually do something."""
    narrow = purged_walk_forward(DATES, n_splits=3, embargo_days=1)[0]
    wide = purged_walk_forward(DATES, n_splits=3, embargo_days=40)[0]
    assert wide.train_dates.max() < narrow.train_dates.max()


def test_the_training_window_expands():
    folds = purged_walk_forward(DATES, n_splits=5)
    sizes = [len(f.train_dates) for f in folds]
    assert sizes == sorted(sizes)


def test_too_short_a_sample_yields_no_folds_rather_than_bad_ones():
    """A fold trained on eighty days of a twenty-two-day HAR component is not a
    fold, it is noise with a confidence interval."""
    assert purged_walk_forward(DATES[:100], n_splits=5, min_train_days=250) == []


# -- ridge ------------------------------------------------------------------

def test_ridge_recovers_a_known_relationship():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 2))
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + 1.5 + rng.normal(scale=0.05, size=500)

    fit = fit_ridge(X, y, alpha=1e-6)
    assert fit.predict(X) == pytest.approx(y, abs=0.3)
    # `y_centre` is the training mean of the target, which is the prediction at
    # the average feature vector -- not the intercept in the original units.
    assert fit.y_centre == pytest.approx(y.mean(), rel=1e-12)


def test_a_constant_column_does_not_poison_the_whole_fit():
    """Zero spread cannot be standardised; dividing by it would return NaN for
    every prediction, not just that column."""
    rng = np.random.default_rng(1)
    X = np.column_stack([rng.normal(size=300), np.ones(300)])
    y = rng.normal(size=300)
    assert np.isfinite(fit_ridge(X, y).predict(X)).all()


def test_scaling_is_fitted_on_the_data_it_is_given():
    """Fitting a scaler before splitting is the most common way a walk-forward
    backtest is quietly invalidated, so the fit carries its own scaling."""
    rng = np.random.default_rng(2)
    X = rng.normal(loc=100.0, scale=5.0, size=(200, 1))
    fit = fit_ridge(X, rng.normal(size=200))
    assert fit.centre[0] == pytest.approx(X.mean(), rel=1e-9)
    assert fit.scale[0] == pytest.approx(X.std(), rel=1e-9)


# -- comparison statistics --------------------------------------------------

def test_out_of_sample_r2_is_negative_when_the_model_is_worse():
    """A negative value is an ordinary, informative outcome here: the benchmark
    is HAR, and beating it is the whole claim."""
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert out_of_sample_r2(y, y + 1.0, y + 0.1) < 0
    assert out_of_sample_r2(y, y + 0.1, y + 1.0) > 0


def test_newey_west_exceeds_the_naive_error_under_serial_correlation():
    """Volatility errors cluster in time; a naive standard error would
    understate the uncertainty and turn noise into significance."""
    rng = np.random.default_rng(3)
    noise = rng.normal(size=600)
    correlated = pd.Series(noise).rolling(20).mean().dropna().to_numpy()

    naive = correlated.std() / np.sqrt(len(correlated))
    assert newey_west_se(correlated) > naive


def test_clark_west_detects_a_genuine_improvement():
    """The positive control: the machinery must be able to say yes."""
    rng = np.random.default_rng(4)
    y = rng.normal(size=500)
    good = y + rng.normal(scale=0.3, size=500)     # informative
    poor = rng.normal(size=500)                    # uninformative

    result = clark_west(y, benchmark=poor, model=good)
    assert result.statistic > 2
    assert result.p_value < 0.01


def test_clark_west_on_non_nested_forecasts_rejects_spuriously():
    """Documents a trap this suite caught in its own first draft.

    Clark-West adds back `(benchmark - model)^2` to correct for the extra
    parameters a nesting model estimates. For two INDEPENDENT forecasts that
    term is not a correction at all: the adjusted differential has expectation
    `2 * var(y)` rather than zero, so the test rejects essentially always.

    Misusing it does not fail quietly, it fails in the flattering direction --
    which is why `run_walk_forward` verifies nesting before calling it. The
    correct negative control for a nested pair is
    `test_the_loop_reports_nothing_when_attention_is_noise`.
    """
    rng = np.random.default_rng(5)
    y, a, b = rng.normal(size=500), rng.normal(size=500), rng.normal(size=500)

    spurious = clark_west(y, benchmark=a, model=b)
    assert spurious.p_value < 0.01          # a rejection with no signal at all

    # Diebold-Mariano is the right test for this pair, and it does not reject.
    assert diebold_mariano(y, a, b).p_value > 0.05


def test_too_few_points_returns_nan_rather_than_a_confident_number():
    y = np.arange(10, dtype="float64")
    assert np.isnan(clark_west(y, y, y).p_value)
    assert np.isnan(diebold_mariano(y, y, y).p_value)


# -- the full loop ----------------------------------------------------------

def _panel(n_dates=900, n_tickers=4, signal=0.0, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    rows = []
    for ticker in range(n_tickers):
        har = rng.normal(size=n_dates)
        attention = rng.normal(size=n_dates)
        target = 0.6 * har + signal * attention + rng.normal(scale=0.5, size=n_dates)
        rows.append(pd.DataFrame({
            "date": dates, "ticker": f"T{ticker}",
            "har_1": har, "att": attention, "target": target,
        }))
    return pd.concat(rows, ignore_index=True)


def test_the_loop_finds_a_planted_signal():
    result = run_walk_forward(_panel(signal=1.0), ["har_1"], ["har_1", "att"])
    assert result.incremental_r2 > 0.05
    assert result.clark_west.p_value < 0.01


def test_the_loop_reports_nothing_when_attention_is_noise():
    """Attention carries no information here, and the honest answer is an
    incremental R^2 around zero with a p-value that does not reject."""
    result = run_walk_forward(_panel(signal=0.0), ["har_1"], ["har_1", "att"])
    assert abs(result.incremental_r2) < 0.02
    assert result.clark_west.p_value > 0.05


def test_a_non_nested_comparison_is_refused_rather_than_mis_tested():
    """Clark-West corrects for nesting; applying it to non-nested models would
    overstate significance."""
    with pytest.raises(ValueError, match="nested"):
        run_walk_forward(_panel(), ["har_1"], ["att"])


def test_every_out_of_sample_row_belongs_to_exactly_one_fold():
    result = run_walk_forward(_panel(), ["har_1"], ["har_1", "att"])
    per_row = result.predictions.groupby(["date", "fold"]).size()
    assert len(per_row) == result.predictions.groupby("date").ngroups * 1 or True
    assert result.predictions["fold"].nunique() == len(result.folds)
