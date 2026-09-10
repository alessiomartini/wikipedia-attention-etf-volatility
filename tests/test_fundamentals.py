"""Tests for the yfinance fundamentals layer.

The three-state earnings mask is the one that matters. Yahoo's earnings history
is shallow, so a plain boolean would report False for every day before coverage
begins -- telling the model that no company announced results for years. That is
a false statement, not a missing value, and it would bias the study's main
confounder control toward looking useless.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from attention_panel import fundamentals


CALENDAR = pd.date_range("2015-01-01", "2024-12-31", freq="B", name="date")


def test_days_before_coverage_are_unknown_not_false():
    """A shallow history must not be reported as "no earnings ever happened"."""
    earnings = pd.DatetimeIndex(["2020-02-05", "2020-05-06"])
    mask = fundamentals.earnings_window(
        CALENDAR, earnings, coverage=(dt.date(2020, 1, 1), dt.date(2020, 12, 31))
    )

    assert mask.loc[pd.Timestamp("2016-06-15")] is pd.NA      # outside coverage
    assert mask.loc[pd.Timestamp("2020-02-05")] == True       # an announcement
    assert mask.loc[pd.Timestamp("2020-03-16")] == False      # covered, quiet
    assert mask.loc[pd.Timestamp("2023-06-15")] is pd.NA      # after coverage


def test_a_plain_boolean_would_have_said_false_there():
    """The negative control: NA and False are genuinely different answers."""
    mask = fundamentals.earnings_window(
        CALENDAR, pd.DatetimeIndex(["2020-02-05"]),
        coverage=(dt.date(2020, 1, 1), dt.date(2020, 12, 31)),
    )
    covered = mask.loc["2020-01-01":"2020-12-31"]
    assert covered.notna().all()          # inside coverage everything is decided
    assert mask.loc["2015":"2019"].isna().all()   # outside, nothing is


def test_no_coverage_at_all_means_every_day_is_unknown():
    mask = fundamentals.earnings_window(CALENDAR, pd.DatetimeIndex([]), coverage=None)
    assert mask.isna().all()


def test_the_window_widens_around_each_announcement():
    """Attention and volatility both build ahead of results and decay after."""
    calendar = pd.date_range("2020-02-01", "2020-02-15", freq="D", name="date")
    mask = fundamentals.earnings_window(
        calendar, pd.DatetimeIndex(["2020-02-10"]),
        coverage=(dt.date(2020, 2, 1), dt.date(2020, 2, 15)), before=2, after=1,
    )
    assert mask.loc[pd.Timestamp("2020-02-08")] == True   # two days before
    assert mask.loc[pd.Timestamp("2020-02-11")] == True   # one day after
    assert mask.loc[pd.Timestamp("2020-02-07")] == False
    assert mask.loc[pd.Timestamp("2020-02-12")] == False


# -- corporate actions ------------------------------------------------------


def test_an_extreme_move_on_a_split_date_is_explained():
    """Closes the loop on a flag that would otherwise cost a manual check."""
    splits = pd.Series([2.0], index=pd.DatetimeIndex(["2024-06-10"]))
    verdicts = fundamentals.explain_extreme_moves(["2024-06-10"], splits)
    assert "explained" in verdicts["2024-06-10"]
    assert "2-for-1" in verdicts["2024-06-10"]


def test_an_extreme_move_with_no_corporate_action_stays_unexplained():
    """Checks that always come back clean stop being made -- so this one must
    be able to come back dirty."""
    splits = pd.Series([2.0], index=pd.DatetimeIndex(["2020-01-02"]))
    verdicts = fundamentals.explain_extreme_moves(["2024-06-10"], splits)
    assert "unexplained" in verdicts["2024-06-10"]


def test_a_ticker_with_no_splits_leaves_every_flag_unexplained():
    verdicts = fundamentals.explain_extreme_moves(
        ["2024-06-10"], pd.Series(dtype="float64")
    )
    assert "unexplained" in verdicts["2024-06-10"]


# -- share counts -----------------------------------------------------------


def test_share_counts_are_carried_forward_but_never_backwards():
    """Filling backwards would invent a share count from a later buyback."""
    close = pd.Series(
        [10.0, 11.0, 12.0],
        index=pd.date_range("2024-01-01", periods=3, freq="D", name="date"),
    )
    shares = pd.Series([100.0], index=pd.DatetimeIndex(["2024-01-02"]))

    caps = fundamentals.market_cap_history(close, shares)
    assert np.isnan(caps.iloc[0])            # before the first known count
    assert caps.iloc[1] == 1100.0
    assert caps.iloc[2] == 1200.0            # carried forward


def test_turnover_is_comparable_across_price_levels():
    """Raw volume is not: a EUR 3 share and a EUR 500 one differ by orders of
    magnitude for the same economic activity."""
    volume = pd.Series(
        [1_000.0, 2_000.0],
        index=pd.date_range("2024-01-01", periods=2, freq="D", name="date"),
    )
    shares = pd.Series([100_000.0], index=pd.DatetimeIndex(["2023-12-01"]))
    assert fundamentals.turnover(volume, shares).tolist() == [0.01, 0.02]


def test_a_zero_share_count_yields_nan_rather_than_infinity():
    volume = pd.Series([1_000.0], index=pd.DatetimeIndex(["2024-01-01"]))
    shares = pd.Series([0.0], index=pd.DatetimeIndex(["2023-12-01"]))
    assert np.isnan(fundamentals.turnover(volume, shares).iloc[0])


def test_empty_inputs_return_empty_rather_than_raising():
    """Every field here is optional; yfinance breaks these before it breaks
    prices, and a failure must never abort a whole run."""
    empty = pd.Series(dtype="float64")
    assert fundamentals.market_cap_history(empty, empty).empty
    assert fundamentals.turnover(empty, empty).empty
