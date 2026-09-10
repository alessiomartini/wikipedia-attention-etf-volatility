"""Tests for the market-data layer.

The property tests here are not decoration: `market.py` claims in its docstring
that range-based estimators are immune to a corporate-action factor applied
consistently across the bar, and that claim is the entire reason the module
requests adjusted OHLC instead of mixing raw and adjusted columns. If it were
false, every volatility number in the study would be wrong on split days.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from attention_panel import market


def test_garman_klass_matches_the_published_formula(ohlcv):
    gk = market.garman_klass_variance(ohlcv)

    # Computed by hand from O=100, H=110, L=90, C=105:
    #   0.5 * ln(110/90)^2 - (2 ln2 - 1) * ln(105/100)^2
    hl = math.log(110 / 90)
    co = math.log(105 / 100)
    expected = 0.5 * hl**2 - (2 * math.log(2) - 1) * co**2

    assert gk.iloc[0] == pytest.approx(expected, rel=1e-12)
    assert expected == pytest.approx(0.019214798, abs=1e-9)


@pytest.mark.parametrize("factor", [0.25, 2.0, 7.0])
def test_range_estimators_are_invariant_to_a_consistent_adjustment_factor(ohlcv, factor):
    """A split applied to all four prices must not change the variance.

    This is the property that makes `auto_adjust=True` safe and mixing raw
    highs with adjusted closes catastrophic.
    """
    scaled = ohlcv.copy()
    scaled[["open", "high", "low", "close"]] *= factor

    for estimator in (
        market.garman_klass_variance,
        market.parkinson_variance,
        market.overnight_variance,
    ):
        original = estimator(ohlcv)
        rescaled = estimator(scaled)
        pd.testing.assert_series_equal(original, rescaled, check_exact=False, rtol=1e-12)


def test_inconsistent_adjustment_does_change_the_variance(ohlcv):
    """The negative control for the test above.

    Scaling only the close -- the shape of the bug this module guards against
    -- must visibly corrupt the estimate, otherwise the invariance test above
    would pass for trivial reasons.
    """
    corrupted = ohlcv.copy()
    corrupted["close"] *= 2.0
    assert market.garman_klass_variance(corrupted).iloc[0] != pytest.approx(
        market.garman_klass_variance(ohlcv).iloc[0]
    )


def test_zero_range_days_become_nan_not_zero(ohlcv):
    """A halted day must not enter the model as 'zero volatility'.

    Zero variance is negative infinity in logs, and the panel is modelled in
    logs, so a single unhandled halted day would dominate the whole regression.
    """
    gk = market.garman_klass_variance(ohlcv)
    halted = pd.Timestamp("2024-01-04")  # high == low == 50
    assert np.isnan(gk.loc[halted])
    assert not np.isnan(gk.loc[pd.Timestamp("2024-01-02")])

    log_rv = market.build_market_features(ohlcv)["log_rv"]
    assert np.isfinite(log_rv.dropna()).all()


def test_realized_variance_includes_the_overnight_gap(ohlcv):
    """Intraday-only would understate exactly the days news arrives overnight."""
    gk = market.garman_klass_variance(ohlcv)
    rv = market.realized_variance(ohlcv)
    # 2024-01-05 opens at 60.0 after a 104.5 close: a large gap that an
    # intraday-only estimator would score as an ordinary, quiet day.
    day = pd.Timestamp("2024-01-05")
    assert rv.loc[day] > gk.loc[day] * 10


def test_realized_variance_is_undefined_when_either_component_is(ohlcv):
    """A halted day must not be reported as a gap-only variance.

    Mixing a gap-only measurement into a column of full-day variances puts two
    incommensurable quantities under one name -- an error that would surface as
    a regression coefficient, never as an exception.
    """
    rv = market.realized_variance(ohlcv)
    assert np.isnan(rv.loc[pd.Timestamp("2024-01-04")])  # high == low, halted
    assert np.isnan(rv.iloc[0])                          # no prior close to gap from
    assert not np.isnan(rv.loc[pd.Timestamp("2024-01-05")])


def test_abnormal_turnover_baseline_excludes_the_current_day():
    """Including today in its own baseline shrinks the spikes being measured."""
    volume = [100.0] * 60 + [1000.0]
    frame = pd.DataFrame(
        {"volume": volume},
        index=pd.date_range("2024-01-01", periods=61, freq="D", name="date"),
    )
    spike = market.abnormal_turnover(frame, window=60).iloc[-1]
    # log(1000/100) exactly, because the median baseline is the 60 prior days.
    assert spike == pytest.approx(math.log(10.0), rel=1e-12)


def test_validate_ohlcv_flags_an_internally_inconsistent_bar():
    frame = pd.DataFrame(
        {
            "open": [100.0], "high": [95.0], "low": [90.0],
            "close": [99.0], "volume": [1000.0],
        },
        index=pd.DatetimeIndex(pd.to_datetime(["2024-01-02"]), name="date"),
    )
    report = market.validate_ohlcv(frame, "TEST")
    assert report["high_below_others"] == 1


def test_validate_ohlcv_flags_a_probable_unadjusted_split():
    close = [100.0] * 5 + [50.0] * 5  # a clean 2-for-1 that nobody adjusted
    frame = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": [1e6] * 10},
        index=pd.date_range("2024-01-02", periods=10, freq="B", name="date"),
    )
    report = market.validate_ohlcv(frame, "TEST")
    assert report["extreme_moves"] == 1
    assert report["extreme_move_dates"]


def test_cross_check_reports_disagreement_without_picking_a_winner():
    index = pd.date_range("2024-01-02", periods=5, freq="B", name="date")
    base = pd.DataFrame({"close": [10.0, 10.0, 10.0, 10.0, 10.0]}, index=index)
    other = base.copy()
    other.loc[index[2], "close"] = 20.0

    result = market.cross_check(base, other)
    assert result["mismatches"] == 1
    assert result["comparable_days"] == 5
    assert result["mismatch_dates"] == ["2024-01-04"]


@pytest.mark.parametrize(
    "yahoo, stooq",
    [("NKE", "nke.us"), ("MC.PA", "mc.fr"), ("HM-B.ST", "hmb.se"), ("1913.HK", "1913.hk")],
)
def test_stooq_symbol_translation(yahoo, stooq):
    assert market.to_stooq_symbol(yahoo) == stooq


def test_unmapped_exchange_returns_none_rather_than_guessing():
    """A wrong Stooq symbol would silently cross-check against another company."""
    assert market.to_stooq_symbol("ABC.XYZ") is None


def test_an_internally_inconsistent_bar_is_excluded_from_the_estimators():
    """The failure mode found in live vendor data: a high below the close.

    `ln(H/L)` stays finite on such a bar, so it produces a plausible but
    fabricated variance on a day the study would otherwise treat as ordinary.
    It has to be excluded from the numbers, not merely counted in a report.
    """
    frame = pd.DataFrame(
        {
            "open": [100.0, 100.0],
            "high": [110.0, 101.0],   # second bar: high sits below the close
            "low": [90.0, 95.0],
            "close": [105.0, 105.0],
            "volume": [1e6, 1e6],
        },
        index=pd.date_range("2024-01-02", periods=2, freq="B", name="date"),
    )

    assert list(market.consistent_bars(frame)) == [True, False]
    assert np.isnan(market.garman_klass_variance(frame).iloc[1])
    assert np.isnan(market.parkinson_variance(frame).iloc[1])
    assert np.isnan(market.realized_variance(frame).iloc[1])

    report = market.validate_ohlcv(frame, "TEST")
    assert report["high_below_others"] == 1
    assert report["unusable_bars"] == 1


def test_a_low_above_the_open_is_also_excluded():
    frame = pd.DataFrame(
        {
            "open": [100.0], "high": [110.0], "low": [102.0],  # low above the open
            "close": [105.0], "volume": [1e6],
        },
        index=pd.DatetimeIndex(pd.to_datetime(["2024-01-02"]), name="date"),
    )
    assert not market.consistent_bars(frame).iloc[0]
    assert np.isnan(market.garman_klass_variance(frame).iloc[0])
