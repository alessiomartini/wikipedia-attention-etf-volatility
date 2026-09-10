"""Company-level data from yfinance beyond the price bar.

WHY THIS EXISTS

The design already committed to things it had no data for. DESIGN.md section 6
says cross-sectional heterogeneity is a *second-stage* question asked with few
tests -- "does the coefficient vary with market cap, analyst coverage, retail
ownership?" -- and section 8 names the study's central identification threat:
attention and volatility plausibly share a common driver, namely news.

yfinance carries all four, and using it is strictly better than adding another
vendor: it is already the primary price source, needs no key, and covers all
eleven venues in this universe.

    earnings dates       the datable subset of "news" -- the main confounder
    corporate actions    explains extreme-move flags instead of guessing at them
    shares outstanding   turnover, and market cap through time
    holders / analysts   the size, coverage and retail-ownership splits

THE THREE-STATE RULE, WHICH IS THE POINT OF THIS MODULE

Yahoo's earnings history is shallow -- roughly a decade at best, often far
less, and shorter for non-US listings. That matters more than it sounds.

A panel spanning 2015-2026 with earnings dates known only from 2020 would, if
the mask were a plain boolean, record `False` for every day before 2020 -- which
the model reads as "no earnings were announced in those five years". That is not
a gap, it is a false statement, and it biases the control in the direction of
looking useless. So coverage is tracked separately from the value, and days
outside the known window are `pd.NA`, never `False`. A partial control treated
as complete is worse than no control at all.

EVERYTHING HERE IS OPTIONAL AND DEGRADES QUIETLY. yfinance breaks periodically
and these endpoints break before the price endpoint does; a failure returns
empty with a reason, never an exception that would abort a whole run.

POINT-IN-TIME CAVEAT. `fast_info` and `info` are snapshots of TODAY. Today's
market cap is not 2016's, so splitting a ten-year panel by it uses information
from the end of the sample at the start of it. It is defensible only for a
slow-moving classification (is this a large or a small company?) and must be
labelled as such wherever it is used; `market_cap_history` builds the honest
version where share counts are available.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class Fundamentals:
    """Everything known about one company beyond its price bars."""

    ticker: str
    earnings: pd.DatetimeIndex = field(default_factory=lambda: pd.DatetimeIndex([]))
    #: The span earnings dates are actually known over. Outside it the mask is
    #: pd.NA rather than False -- see the module docstring.
    earnings_coverage: tuple[dt.date, dt.date] | None = None
    splits: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))
    dividends: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))
    shares: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))
    snapshot: dict = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)


def _ticker(ticker: str):
    import yfinance

    return yfinance.Ticker(ticker)


def fetch_fundamentals(ticker: str, limit: int = 80) -> Fundamentals:
    """Collect every optional field, tolerating failure in each independently.

    `limit` is the number of past earnings announcements requested. The default
    reaches for roughly twenty years; Yahoo will return whatever it has, which
    is the whole reason coverage is recorded rather than assumed.
    """
    out = Fundamentals(ticker=ticker)
    try:
        handle = _ticker(ticker)
    except Exception as exc:
        out.problems.append(f"yfinance unavailable: {exc}")
        return out

    out.earnings, out.earnings_coverage, problem = _earnings(handle, limit)
    if problem:
        out.problems.append(problem)

    for name, attribute in (("splits", "splits"), ("dividends", "dividends")):
        try:
            series = getattr(handle, attribute)
            if series is not None and len(series):
                series.index = pd.to_datetime(series.index).tz_localize(None).normalize()
                setattr(out, name, series.astype("float64").sort_index())
        except Exception as exc:
            out.problems.append(f"{name}: {type(exc).__name__}")

    try:
        shares = handle.get_shares_full()
        if shares is not None and len(shares):
            shares.index = pd.to_datetime(shares.index).tz_localize(None).normalize()
            # Yahoo reports several observations per day for some tickers; keep
            # the last, since a duplicated index breaks every later reindex.
            out.shares = shares.astype("float64").sort_index().groupby(level=0).last()
    except Exception as exc:
        out.problems.append(f"shares: {type(exc).__name__}")

    out.snapshot = _snapshot(handle, out)
    return out


def _earnings(handle, limit: int) -> tuple[pd.DatetimeIndex, tuple[dt.date, dt.date] | None, str]:
    try:
        frame = handle.get_earnings_dates(limit=limit)
    except Exception as exc:
        return pd.DatetimeIndex([]), None, f"earnings: {type(exc).__name__}"

    if frame is None or frame.empty:
        return pd.DatetimeIndex([]), None, "earnings: none returned"

    index = pd.to_datetime(frame.index)
    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)
    index = pd.DatetimeIndex(index).normalize().sort_values().unique()

    # Future dates are kept deliberately. A scheduled announcement is public
    # in advance, so excluding a day because an announcement is due is not
    # look-ahead -- what would be look-ahead is using its CONTENT.
    coverage = (index.min().date(), index.max().date())
    return pd.DatetimeIndex(index), coverage, ""


def _snapshot(handle, out: Fundamentals) -> dict:
    """Point-in-time cross-sectional attributes. TODAY's values, not history."""
    snapshot: dict = {}
    try:
        fast = handle.fast_info
        for key in ("market_cap", "currency", "exchange", "shares"):
            try:
                snapshot[key] = fast[key]
            except Exception:
                pass
    except Exception as exc:
        out.problems.append(f"fast_info: {type(exc).__name__}")

    # Analyst coverage: the count of recommending analysts, one of the three
    # heterogeneity splits pre-registered in DESIGN.md section 6.
    try:
        recommendations = handle.recommendations
        if recommendations is not None and len(recommendations):
            numeric = recommendations.select_dtypes("number")
            snapshot["analyst_count"] = int(numeric.iloc[0].sum()) if len(numeric) else None
    except Exception as exc:
        out.problems.append(f"recommendations: {type(exc).__name__}")

    # Institutional ownership share; its complement proxies retail ownership,
    # the third pre-registered split.
    try:
        major = handle.major_holders
        if major is not None and len(major):
            for label in ("institutionsPercentHeld", "institutionsFloatPercentHeld"):
                if label in major.index:
                    snapshot["institutional_share"] = float(major.loc[label].iloc[0])
                    break
    except Exception as exc:
        out.problems.append(f"major_holders: {type(exc).__name__}")

    return snapshot


# ---------------------------------------------------------------------------
# Derived features
# ---------------------------------------------------------------------------


def earnings_window(
    calendar: pd.DatetimeIndex,
    earnings: pd.DatetimeIndex,
    coverage: tuple[dt.date, dt.date] | None,
    before: int = 1,
    after: int = 1,
) -> pd.Series:
    """Three-state mask: in an earnings window, outside one, or unknown.

    Returns a nullable boolean Series -- True, False, or `pd.NA` for days
    outside the span earnings dates are known over.

    THE NA IS THE WHOLE POINT. Yahoo's earnings history is shallow, so a plain
    boolean mask would report `False` for every day before coverage begins,
    telling the model that no company announced results for years. That is a
    false statement rather than a missing value, and it would quietly bias the
    control toward looking useless.

    `before` and `after` widen the window around each announcement, because
    attention and volatility both build ahead of results and decay after them.
    """
    mask = pd.Series(pd.NA, index=calendar, dtype="boolean")
    if coverage is None:
        return mask

    start, end = pd.Timestamp(coverage[0]), pd.Timestamp(coverage[1])
    known = (calendar >= start) & (calendar <= end)
    mask[known] = False

    for date in earnings:
        window = (calendar >= date - pd.Timedelta(days=before)) & (
            calendar <= date + pd.Timedelta(days=after)
        )
        mask[window & known] = True
    return mask


def explain_extreme_moves(
    flagged: list[str], splits: pd.Series, tolerance_days: int = 3
) -> dict[str, str]:
    """Match extreme-move flags against known corporate actions.

    `validate_ohlcv` flags any single-day move beyond 50% as a probable
    unadjusted split -- deliberately a flag, not an auto-fix, because silently
    correcting prices makes a dataset untraceable. This closes the loop: a
    flagged day that coincides with a recorded split is EXPLAINED and needs no
    human, while one that does not is a genuine anomaly worth looking at.

    Without this, every flag costs a manual check, and checks that always come
    back clean stop being made.
    """
    verdicts: dict[str, str] = {}
    split_dates = pd.DatetimeIndex(splits.index) if len(splits) else pd.DatetimeIndex([])

    for day in flagged:
        stamp = pd.Timestamp(day)
        if len(split_dates):
            distance = (split_dates - stamp).days if hasattr(split_dates - stamp, "days") else None
            near = split_dates[abs((split_dates - stamp).days) <= tolerance_days]
            if len(near):
                ratio = splits.loc[near].iloc[0]
                verdicts[day] = f"explained: {ratio:g}-for-1 split on {near[0].date()}"
                continue
        verdicts[day] = "unexplained: no corporate action within tolerance"
    return verdicts


def market_cap_history(close: pd.Series, shares: pd.Series) -> pd.Series:
    """Market cap through time, rather than today's value applied to the past.

    Splitting a ten-year panel by TODAY's market cap uses information from the
    end of the sample at its start. This is the honest version, for the tickers
    where Yahoo has a share-count history; where it does not, the snapshot is
    the fallback and must be labelled as a snapshot.

    Share counts are step functions reported at irregular dates, so they are
    carried forward -- but not backwards before the first observation, which
    would invent a share count from a later buyback or issuance.
    """
    if close.empty or shares.empty:
        return pd.Series(dtype="float64", index=close.index, name="market_cap")
    aligned = shares.reindex(shares.index.union(close.index)).sort_index().ffill()
    return (close * aligned.reindex(close.index)).rename("market_cap")


def turnover(volume: pd.Series, shares: pd.Series) -> pd.Series:
    """Share turnover: volume as a fraction of shares outstanding.

    Directly comparable across companies, unlike raw volume, which differs by
    orders of magnitude between a EUR 3 share and a EUR 500 one. The panel's
    validation target already normalises within a stock via a trailing median,
    which entity fixed effects make sufficient; this is the cross-sectionally
    interpretable version, for the heterogeneity stage and for reporting.
    """
    if volume.empty or shares.empty:
        return pd.Series(dtype="float64", index=volume.index, name="turnover")
    aligned = shares.reindex(shares.index.union(volume.index)).sort_index().ffill()
    aligned = aligned.reindex(volume.index)
    return (volume / aligned.where(aligned > 0)).rename("turnover")
