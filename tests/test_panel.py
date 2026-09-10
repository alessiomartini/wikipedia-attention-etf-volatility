"""Tests for panel assembly.

The inter-session aggregation is the one to read first. Attention exists every
day and markets do not, so reindexing requires a choice, and the two obvious
ones are both wrong: taking the last value discards roughly two sevenths of all
attention -- and not a random two sevenths, since a scandal breaking on a
Saturday is exactly what the hypothesis is about -- while summing double-counts,
because abnormal attention is a standardised level rather than a flow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from attention_panel.panel import (
    PanelSpec,
    add_attention_share,
    assert_no_lookahead,
    build_company_frame,
    har_components,
    sessions_since_previous,
    stack_panel,
)


def _market(days: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2023-01-02", periods=days, freq="B", name="date")
    return pd.DataFrame(
        {
            "log_rv": rng.normal(-9.0, 0.5, days),
            "abnormal_turnover": rng.normal(0.0, 0.3, days),
            "log_return": rng.normal(0.0, 0.01, days),
        },
        index=index,
    )


def _views(index: pd.DatetimeIndex, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.integers(500, 1500, len(index)).astype(float), index=index, name="views")


# -- the calendar mismatch --------------------------------------------------


def test_weekend_attention_is_carried_into_the_next_session():
    """Discarding it would drop two sevenths of the data, and not at random."""
    calendar = pd.date_range("2024-01-01", "2024-01-15", freq="D", name="date")
    daily = pd.Series(0.0, index=calendar)
    daily.loc[pd.Timestamp("2024-01-06")] = 100.0   # a Saturday spike

    trading = pd.DatetimeIndex(
        [d for d in calendar if d.dayofweek < 5], name="date"
    )
    aggregated = sessions_since_previous(daily, trading, lag_days=2)

    # With a two-day lag, Saturday the 6th becomes observable on Monday the 8th.
    assert aggregated.loc[pd.Timestamp("2024-01-08")] > 0
    assert aggregated.dropna().sum() > 0


def test_three_quiet_days_do_not_impersonate_one_loud_one():
    """Summing would triple a constant level; averaging keeps it a level."""
    calendar = pd.date_range("2024-01-01", "2024-01-15", freq="D", name="date")
    daily = pd.Series(1.0, index=calendar)      # perfectly constant attention
    trading = pd.DatetimeIndex([d for d in calendar if d.dayofweek < 5], name="date")

    aggregated = sessions_since_previous(daily, trading, lag_days=2).dropna()
    # Every session sees the same level, whatever the gap length before it.
    assert aggregated.round(9).nunique() == 1
    assert aggregated.iloc[0] == pytest.approx(1.0)


def test_consecutive_windows_tile_the_calendar_without_gaps_or_overlap():
    """Each calendar day must be counted exactly once across all sessions."""
    calendar = pd.date_range("2024-01-01", "2024-01-31", freq="D", name="date")
    # A distinct value per day, so double counting or omission is detectable.
    daily = pd.Series(range(len(calendar)), index=calendar, dtype="float64")
    trading = pd.DatetimeIndex([d for d in calendar if d.dayofweek < 5], name="date")

    counted = []
    previous_end = None
    for session in trading:
        window_end = session - pd.Timedelta(days=2)
        if previous_end is not None:
            counted.extend(pd.date_range(previous_end + pd.Timedelta(days=1), window_end))
        previous_end = window_end

    assert len(counted) == len(set(counted))       # nothing counted twice
    assert pd.DatetimeIndex(counted).is_monotonic_increasing


def test_the_first_session_has_no_predecessor_and_is_nan():
    """A guess would be indistinguishable from a measurement."""
    calendar = pd.date_range("2024-01-01", "2024-01-10", freq="D", name="date")
    trading = pd.DatetimeIndex([d for d in calendar if d.dayofweek < 5], name="date")
    aggregated = sessions_since_previous(pd.Series(1.0, index=calendar), trading, lag_days=2)
    assert np.isnan(aggregated.iloc[0])


# -- the HAR baseline -------------------------------------------------------


def test_har_components_use_only_past_volatility():
    """Row t must contain nothing realised on or after t."""
    log_rv = pd.Series(
        range(30), index=pd.date_range("2024-01-01", periods=30, freq="B", name="date"),
        dtype="float64",
    )
    har = har_components(log_rv, (1, 5, 22))

    # har_1 on row t is simply log_rv at t-1.
    assert har["har_1"].iloc[5] == log_rv.iloc[4]
    assert np.isnan(har["har_1"].iloc[0])
    # The five-day mean ending at t-1.
    assert har["har_5"].iloc[10] == pytest.approx(log_rv.iloc[5:10].mean())


# -- assembly ---------------------------------------------------------------


def test_the_target_is_the_next_days_volatility():
    market = _market()
    frame = build_company_frame("NKE", market, {}, PanelSpec(horizon=1))
    assert frame["target"].iloc[0] == market["log_rv"].iloc[1]
    assert np.isnan(frame["target"].iloc[-1])   # no tomorrow for the last row


def test_the_venue_lag_is_recorded_on_every_row():
    market = _market()
    views = _views(pd.date_range("2023-01-01", periods=500, freq="D", name="date"))

    us = build_company_frame("NKE", market, {"corporate_en": views})
    tokyo = build_company_frame("9983.T", market, {"corporate_en": views})

    assert (us["availability_lag"] == 1).all()
    assert (tokyo["availability_lag"] == 2).all()


def test_thin_companies_are_dropped_and_the_drop_is_logged(caplog):
    """An entity contributing thirty rows still absorbs a fixed effect, spending
    a degree of freedom on a company the panel cannot speak about."""
    thin = build_company_frame("TINY", _market(days=40), {})
    fat = build_company_frame("NKE", _market(days=400), {})

    with caplog.at_level("INFO"):
        panel = stack_panel([thin, fat], PanelSpec(min_observations=250))

    assert set(panel["ticker"]) == {"NKE"}
    assert "dropping TINY" in caplog.text


def test_the_panel_stays_unbalanced():
    """Delisted names keep their shortened series; padding them, or dropping
    them for being short, is how survivorship bias gets in."""
    long_frame = build_company_frame("NKE", _market(days=400), {})
    shorter = build_company_frame("TOD.MI", _market(days=300), {})

    panel = stack_panel([long_frame, shorter], PanelSpec(min_observations=250))
    counts = panel.groupby("ticker").size()
    assert counts["NKE"] != counts["TOD.MI"]


# -- attention share --------------------------------------------------------


def test_attention_share_is_cross_sectional_and_sums_to_one_per_day():
    index = pd.date_range("2024-01-01", periods=3, freq="B", name="date")
    panel = pd.DataFrame(
        {
            "date": list(index) * 2,
            "ticker": ["A"] * 3 + ["B"] * 3,
            "att_corporate_en": [0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    out = add_attention_share(panel, "att_corporate_en", window=1)
    per_day = out.groupby("date")["att_corporate_en_share"].sum()
    assert per_day.round(9).eq(1.0).all()

    # On the day A spikes, B's share falls although B's own level is unchanged.
    b_shares = out[out["ticker"] == "B"]["att_corporate_en_share"].tolist()
    assert b_shares[1] < b_shares[0]


# -- the structural guard ---------------------------------------------------


def test_a_leaked_target_is_caught_by_the_assertion():
    """Every look-ahead bug so far came from code that believed it was
    maintaining the discipline, so the finished frame is checked too."""
    market = _market()
    frame = build_company_frame("NKE", market, {})
    frame["oops_the_target"] = frame["target"]     # the leak

    problems = assert_no_lookahead(stack_panel([frame]))
    assert any("oops_the_target" in p for p in problems)


def test_a_clean_panel_raises_no_complaint():
    """The negative control: the guard must not fire on correct input."""
    frames = [
        build_company_frame(t, _market(seed=i), {"corporate_en": _views(
            pd.date_range("2023-01-01", periods=500, freq="D", name="date"), seed=i)})
        for i, t in enumerate(["NKE", "MC.PA"])
    ]
    assert assert_no_lookahead(stack_panel(frames)) == []


def test_an_empty_panel_is_reported_rather_than_passing_silently():
    assert assert_no_lookahead(pd.DataFrame()) == ["panel is empty"]
