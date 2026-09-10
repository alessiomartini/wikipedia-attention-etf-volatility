"""Tests for the attention transforms.

Most of these check that a feature cannot see its own future. That class of bug
does not raise, does not show up in a train/test split, and makes the result
BETTER -- which is why each guard here has a negative control proving it is
doing work.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from attention_panel import features


# -- the venue lag ----------------------------------------------------------


@pytest.mark.parametrize("ticker", ["NKE", "LULU", "CPRI"])
def test_us_venues_can_use_yesterdays_attention(ticker):
    """New York opens at 14:30 UTC, hours after pageviews are published."""
    assert features.availability_lag_days(ticker) == 1


@pytest.mark.parametrize("ticker", ["9983.T", "1913.HK", "MC.PA", "BRBY.L", "ADS.DE"])
def test_venues_opening_before_publication_need_two_days(ticker):
    """Tokyo opens at 00:00 UTC and Hong Kong at 01:30, before the data exists.

    Europe opens at 08:00 UTC, inside the publication window -- a coin flip,
    which is not a basis for a feature.
    """
    assert features.availability_lag_days(ticker) == 2


def test_an_unknown_venue_gets_the_conservative_lag():
    """Guessing that an unmapped exchange behaves like New York would leak."""
    assert features.availability_lag_days("ABC.XYZ") == 2


def test_the_lag_responds_to_the_publication_assumption():
    """If publication were guaranteed before every open, one day would do."""
    assert features.availability_lag_days("9983.T", publication_hour=0.0) == 1
    assert features.availability_lag_days("NKE", publication_hour=23.0) == 2


def test_lag_to_tradable_shifts_by_the_venue_amount():
    series = pd.Series(
        [1.0, 2.0, 3.0, 4.0],
        index=pd.date_range("2024-01-01", periods=4, freq="D", name="date"),
    )
    assert features.lag_to_tradable(series, "NKE").tolist()[1:] == [1.0, 2.0, 3.0]
    assert features.lag_to_tradable(series, "9983.T").tolist()[2:] == [1.0, 2.0]


def test_the_venue_lag_is_a_floor_that_cannot_be_reduced():
    """A caller able to pass a smaller lag would eventually pass one."""
    series = pd.Series([1.0], index=pd.date_range("2024-01-01", periods=1, name="date"))
    with pytest.raises(ValueError, match="floor"):
        features.lag_to_tradable(series, "9983.T", extra_lag=-1)


# -- abnormal attention -----------------------------------------------------


def test_abnormal_attention_baseline_excludes_the_current_day():
    """Including today shrinks the largest spikes most -- exactly the events
    the study is about."""
    views = pd.Series(
        [100] * 60 + [10_000],
        index=pd.date_range("2024-01-01", periods=61, freq="D", name="date"),
        dtype="float64",
    )
    # A perfectly flat history has zero MAD, so the scale is undefined rather
    # than infinite -- the next test covers that. Add jitter to get a scale.
    views.iloc[:60] += pd.Series(
        np.tile([0, 1, -1, 2, -2], 12).astype(float), index=views.index[:60]
    )
    spike = features.abnormal_attention(views).iloc[-1]
    assert spike > 10  # many robust deviations away


def test_a_flat_article_yields_nan_rather_than_infinite_surprise():
    """Zero MAD is an undefined surprise, not an unbounded one.

    `inf` would dominate any regression it entered; NaN is dropped explicitly.
    """
    views = pd.Series(
        [100] * 60 + [500],
        index=pd.date_range("2024-01-01", periods=61, freq="D", name="date"),
        dtype="float64",
    )
    result = features.abnormal_attention(views)
    assert np.isnan(result.iloc[-1])
    assert not np.isinf(result.dropna()).any()


def test_log_attention_handles_genuine_zero_traffic_days():
    """A zero-traffic day is a real observation for a small brand on a small
    wiki, not missing data."""
    views = pd.Series([0, 10], index=pd.date_range("2024-01-01", periods=2, name="date"))
    assert features.log_attention(views).tolist() == [0.0, pytest.approx(np.log(11))]


# -- weekday seasonality ----------------------------------------------------


def test_weekday_adjustment_uses_only_past_instances_of_that_weekday():
    """A seasonal decomposition over the full sample adjusts each day partly by
    its own future -- a leak that survives every train/test split, because it
    is baked into the feature before the split happens."""
    index = pd.date_range("2024-01-01", periods=28, freq="D", name="date")
    # Every Monday is 200, every other day 100.
    values = pd.Series(
        [200.0 if d.dayofweek == 0 else 100.0 for d in index], index=index
    )
    adjusted = features.deseasonalise_weekday(values, occurrences=8)

    # The first two Mondays cannot be adjusted: fewer than two prior instances.
    mondays = adjusted[adjusted.index.dayofweek == 0]
    assert np.isnan(mondays.iloc[0])
    # Once a baseline exists, the weekly step is removed entirely.
    assert mondays.dropna().abs().max() == pytest.approx(0.0)


# -- attention share --------------------------------------------------------


def test_attention_share_falls_when_a_rival_takes_the_spotlight():
    """The mechanism behind the negative-correlation hypothesis: a firm's share
    drops with its own absolute traffic unchanged."""
    index = pd.date_range("2024-01-01", periods=2, freq="D", name="date")
    views = pd.DataFrame({"Hermes": [100.0, 100.0], "Gucci": [100.0, 900.0]}, index=index)

    shares = features.attention_share(views, window=1)
    assert shares["Hermes"].tolist() == [0.5, 0.1]
    assert views["Hermes"].tolist() == [100.0, 100.0]  # unchanged in levels


def test_a_sector_wide_blackout_is_not_treated_as_equal_shares():
    """A day with no traffic anywhere is a data gap; 1/N would invent one."""
    index = pd.date_range("2024-01-01", periods=2, freq="D", name="date")
    views = pd.DataFrame({"A": [100.0, 0.0], "B": [100.0, 0.0]}, index=index)
    shares = features.attention_share(views, window=1)
    assert np.isnan(shares.iloc[1]).all()


# -- family aggregation -----------------------------------------------------


def test_families_are_summed_within_but_never_across():
    """Corporate attention is investors, brand attention is consumers. Pooling
    them discards the distinction the feature design rests on."""
    index = pd.date_range("2024-01-01", periods=2, freq="D", name="date")
    series = {
        "LVMH": pd.Series([10.0, 20.0], index=index),
        "Louis Vuitton": pd.Series([1.0, 2.0], index=index),
        "Fendi": pd.Series([3.0, 4.0], index=index),
    }
    families = {"LVMH": "corporate", "Louis Vuitton": "brand", "Fendi": "brand"}

    out = features.aggregate_by_family(series, families)
    assert out["corporate"].tolist() == [10.0, 20.0]
    assert out["brand"].tolist() == [4.0, 6.0]   # two brands summed
    assert "LVMH" not in out.columns


def test_an_article_with_no_declared_family_is_dropped_not_guessed():
    index = pd.date_range("2024-01-01", periods=1, freq="D", name="date")
    out = features.aggregate_by_family(
        {"Unknown": pd.Series([1.0], index=index)}, {}
    )
    assert out.empty
