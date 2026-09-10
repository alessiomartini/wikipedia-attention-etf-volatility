"""Tests for the FRED macro layer.

The publication-lag test is the one that matters. FRED stamps a monthly
observation with the first day of the month it DESCRIBES, not the day it was
released, so a naive forward-fill hands the model a number weeks before anyone
could have known it. That is a look-ahead bug no join would flag, and it would
inflate the apparent value of precisely the confounder controls the study leans
on to defend against a common driver.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from attention_panel import macro

START = dt.date(2024, 1, 1)
END = dt.date(2024, 3, 31)


def test_a_monthly_series_is_not_visible_before_it_was_published():
    """March sentiment is stamped 1 March but released in late March."""
    stamped = pd.Series(
        [70.0],
        index=pd.DatetimeIndex([pd.Timestamp("2024-03-01")], name="date"),
        name="UMCSENT",
    )
    calendar = pd.date_range("2024-03-01", "2024-04-10", freq="B")

    aligned = macro.to_daily(stamped, calendar, publication_lag_days=30)

    # Nothing on 5 March: the number did not exist yet.
    assert np.isnan(aligned.loc[pd.Timestamp("2024-03-05")])
    # Available once the release date has passed.
    assert aligned.loc[pd.Timestamp("2024-04-02")] == 70.0


def test_without_the_lag_the_same_series_leaks(monkeypatch):
    """The negative control: the guard must be doing real work."""
    stamped = pd.Series(
        [70.0],
        index=pd.DatetimeIndex([pd.Timestamp("2024-03-01")], name="date"),
        name="UMCSENT",
    )
    calendar = pd.date_range("2024-03-01", "2024-04-10", freq="B")
    leaked = macro.to_daily(stamped, calendar, publication_lag_days=0)
    assert leaked.loc[pd.Timestamp("2024-03-05")] == 70.0


def test_daily_market_series_carry_no_publication_lag():
    """A VIX close is stamped with the day it was observed, so it is known."""
    for spec in macro.DAILY_SERIES:
        assert spec.publication_lag_days == 0
    for spec in macro.MONTHLY_SERIES:
        assert spec.publication_lag_days > 0


def test_a_long_gap_is_left_visible_rather_than_filled():
    """Carrying a value across a fortnight would invent a fortnight of data."""
    series = pd.Series(
        [10.0],
        index=pd.DatetimeIndex([pd.Timestamp("2024-01-02")], name="date"),
        name="VIXCLS",
    )
    calendar = pd.date_range("2024-01-02", "2024-02-01", freq="B")
    aligned = macro.to_daily(series, calendar, max_staleness_days=7)

    assert aligned.iloc[0] == 10.0
    assert np.isnan(aligned.loc[pd.Timestamp("2024-01-31")])


def test_missing_observations_become_nan_not_zero(fake_client):
    """FRED writes '.' for a missing value; 0.0 would be a fabricated control."""
    body = "observation_date,VIXCLS\n2024-01-02,13.5\n2024-01-03,.\n2024-01-04,14.0\n"
    series = macro.fetch_series(fake_client({"fredgraph": body}), "VIXCLS", START, END)

    assert series.iloc[0] == 13.5
    assert np.isnan(series.iloc[1])
    assert series.iloc[2] == 14.0


def test_the_older_DATE_column_header_is_still_read(fake_client):
    """FRED has used both DATE and observation_date; take the first column."""
    body = "DATE,DGS10\n2024-01-02,3.95\n"
    series = macro.fetch_series(fake_client({"fredgraph": body}), "DGS10", START, END)
    assert series.iloc[0] == 3.95
    assert series.index[0] == pd.Timestamp("2024-01-02")


def test_a_series_that_returns_nothing_yields_an_empty_series(fake_client):
    assert macro.fetch_series(fake_client({}), "NOPE", START, END).empty


def test_build_macro_frame_aligns_everything_to_one_calendar(fake_client):
    body = "observation_date,X\n2024-01-02,1.0\n2024-01-03,2.0\n"
    calendar = pd.date_range("2024-01-02", "2024-01-04", freq="B")
    frame = macro.build_macro_frame(
        fake_client({"fredgraph": body}),
        calendar,
        START,
        END,
        series=(macro.Series("VIXCLS", "vix", "daily"),),
    )
    assert list(frame.index) == list(calendar)
    assert frame["VIXCLS"].tolist() == [1.0, 2.0, 2.0]  # last value carried one day
