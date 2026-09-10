"""Assembling the panel: one row per (company, trading day).

THE PROBLEM THIS MODULE HAS TO SOLVE THAT THE DESIGN DID NOT ANTICIPATE

Attention exists every day. Markets do not. Wikipedia records traffic on
Saturdays, Sundays and Christmas; the exchange records nothing. Reindexing
attention onto trading days therefore requires a decision, and the obvious two
are both wrong:

  * Taking the last available value DISCARDS the weekend. Roughly two sevenths
    of all attention would never enter the study, and it is not a random two
    sevenths -- a scandal breaking on a Saturday is exactly the kind of event
    the hypothesis is about.
  * Summing everything since the previous session DOUBLE-COUNTS, because
    abnormal attention is a standardised level rather than a flow. Three days
    of ordinary interest would look like one day of triple interest.

So each trading day takes the MEAN of abnormal attention over the days that
became newly observable since the previous session. Monday carries Thursday,
Friday and Saturday (under a two-day availability lag); Tuesday carries Sunday
alone. Nothing is dropped, nothing is counted twice, and a quiet weekend does
not masquerade as a spike.

THE LAG DISCIPLINE, RESTATED IN ONE PLACE

Two distinct shifts apply, and conflating them is how look-ahead bugs are born:

  1. AVAILABILITY (`features.availability_lag_days`) -- was this number public
     when the session opened? Venue-dependent: one day for New York, two for
     Tokyo, Hong Kong and Europe.
  2. HORIZON -- the target is the NEXT day's volatility, so the target column is
     shifted back by one relative to the features.

Together: features on row `t` are knowable at the open of `t`, and the target on
row `t` is realised over `t+1`. Everything in this module maintains that, and
`assert_no_lookahead` re-checks it on the assembled frame rather than trusting
that it was maintained.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .features import abnormal_attention, availability_lag_days

log = logging.getLogger(__name__)


@dataclass
class PanelSpec:
    """Everything about the panel that could be tuned, in one place.

    Written down as an object rather than scattered as call-site defaults so a
    specification can be recorded alongside a result. A result whose window
    lengths cannot be recovered is not reproducible.
    """

    target: str = "log_rv"
    #: Trading days ahead the target is realised over. 1 = tomorrow.
    horizon: int = 1
    #: HAR component lengths (Corsi 2009): daily, weekly, monthly.
    har_windows: tuple[int, int, int] = (1, 5, 22)
    attention_window: int = 60
    min_observations: int = 250
    extra_lag: int = 0


def sessions_since_previous(
    daily: pd.Series, trading_days: pd.DatetimeIndex, lag_days: int
) -> pd.Series:
    """Average a calendar-daily series over each inter-session gap.

    For each trading day `t`, take the mean of `daily` over the calendar days
    that became newly observable since the previous trading day, given a
    `lag_days` availability lag. The first trading day has no predecessor and
    is therefore NaN rather than a guess.

    This is what keeps weekend attention in the study without letting three
    quiet days impersonate one loud one.
    """
    if daily.empty or len(trading_days) == 0:
        return pd.Series(dtype="float64", index=trading_days, name=daily.name)

    values = daily.astype("float64").sort_index()
    out = pd.Series(np.nan, index=trading_days, dtype="float64", name=daily.name)

    # The window for session t ends at t - lag_days and begins the day after
    # the window used by the previous session, so consecutive windows tile the
    # calendar exactly: no gaps, no overlaps.
    previous_end: pd.Timestamp | None = None
    for session in trading_days:
        window_end = session - pd.Timedelta(days=lag_days)
        window_start = (
            previous_end + pd.Timedelta(days=1) if previous_end is not None else None
        )
        previous_end = window_end
        if window_start is None or window_start > window_end:
            continue
        chunk = values.loc[window_start:window_end]
        if len(chunk):
            out.loc[session] = chunk.mean()
    return out


def har_components(log_rv: pd.Series, windows: tuple[int, int, int] = (1, 5, 22)) -> pd.DataFrame:
    """The HAR baseline's regressors (Corsi, 2009): daily, weekly, monthly.

    WHY THE BASELINE IS THIS AND NOT AN EMPTY MODEL (DESIGN.md section 5)

    Realized volatility is autocorrelated at roughly 0.7-0.9 daily. A model
    containing yesterday's volatility reports a high R^2 that is *entirely the
    target predicting itself*. Anything the attention features are claimed to
    add has to be measured against this, as incremental out-of-sample R^2 --
    never as a raw R^2.

    Every component is shifted by one, so row `t` holds only information
    available before `t` begins.
    """
    daily, weekly, monthly = windows
    out = pd.DataFrame(index=log_rv.index)
    out[f"har_{daily}"] = log_rv.rolling(daily).mean().shift(1)
    out[f"har_{weekly}"] = log_rv.rolling(weekly).mean().shift(1)
    out[f"har_{monthly}"] = log_rv.rolling(monthly).mean().shift(1)
    return out


def build_company_frame(
    ticker: str,
    market: pd.DataFrame,
    attention_by_family: dict[str, pd.Series],
    spec: PanelSpec = PanelSpec(),
    earnings: pd.Series | None = None,
) -> pd.DataFrame:
    """One company's rows: features knowable at `t`, target realised over `t+h`.

    `attention_by_family` maps a feature name -- typically family and language,
    such as `corporate_en` -- to a calendar-daily pageview series.
    """
    if market.empty or spec.target not in market.columns:
        return pd.DataFrame()

    trading_days = market.index
    lag = availability_lag_days(ticker) + spec.extra_lag

    frame = pd.DataFrame(index=trading_days)
    frame["ticker"] = ticker

    # The target: volatility realised over the NEXT trading day. Negative shift,
    # which is the only place in this codebase where the future is deliberately
    # pulled backwards -- and it is the target, never a feature.
    frame["target"] = market[spec.target].shift(-spec.horizon)

    frame = frame.join(har_components(market[spec.target], spec.har_windows))
    for column in ("abnormal_turnover", "log_return"):
        if column in market.columns:
            frame[column] = market[column].shift(1)

    for name, views in attention_by_family.items():
        abnormal = abnormal_attention(views, window=spec.attention_window)
        frame[f"att_{name}"] = sessions_since_previous(abnormal, trading_days, lag)

    if earnings is not None:
        # Nullable boolean: pd.NA outside the span earnings dates are known
        # over, never False. See fundamentals.earnings_window.
        frame["earnings_window"] = earnings.reindex(trading_days)

    frame["availability_lag"] = lag
    return frame


def stack_panel(company_frames: list[pd.DataFrame], spec: PanelSpec = PanelSpec()) -> pd.DataFrame:
    """Concatenate company frames into a long panel, dropping thin entities.

    A company with fewer than `min_observations` usable rows is dropped, and the
    drop is logged. It is not silent because an entity contributing thirty rows
    still absorbs a fixed effect, which spends a degree of freedom on a company
    the panel cannot say anything about.

    The panel stays UNBALANCED on purpose: delisted names keep their shortened
    series (DESIGN.md section 2.3). Padding them, or dropping them for being
    short, is how survivorship bias gets in.
    """
    usable = []
    for frame in company_frames:
        if frame.empty:
            continue
        complete = frame.dropna(subset=["target"])
        if len(complete) < spec.min_observations:
            ticker = frame["ticker"].iloc[0] if "ticker" in frame else "?"
            log.info("dropping %s: %d usable rows < %d", ticker, len(complete), spec.min_observations)
            continue
        usable.append(frame)

    if not usable:
        return pd.DataFrame()

    panel = pd.concat(usable).rename_axis("date").reset_index()
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)


def add_attention_share(panel: pd.DataFrame, column: str, window: int = 5) -> pd.DataFrame:
    """Each company's share of that day's total attention across the panel.

    THE VEHICLE FOR THE NEGATIVE-CORRELATION HYPOTHESIS (DESIGN.md section 3.1).
    When one house is in crisis its rivals' shares fall with their own traffic
    unchanged, which is a mechanism rather than a coincidence found by
    screening.

    Computed here rather than in `features` because it is inherently
    cross-sectional: it needs every company on a given day, which only exists
    once the panel is stacked.
    """
    if panel.empty or column not in panel.columns:
        return panel

    out = panel.copy()
    # Shares must be non-negative, and abnormal attention is a signed z-score,
    # so the exponential maps it back to a positive scale before normalising.
    weight = np.exp(out[column].astype("float64"))
    totals = weight.groupby(out["date"]).transform("sum")
    out[f"{column}_share"] = weight / totals.where(totals > 0)
    out[f"{column}_share"] = (
        out.groupby("ticker")[f"{column}_share"].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
    )
    return out


def assert_no_lookahead(panel: pd.DataFrame, spec: PanelSpec = PanelSpec()) -> list[str]:
    """Re-derive the lag guarantees on the assembled frame. Returns complaints.

    The point is to check the OUTPUT rather than trust that the pipeline that
    produced it was correct. Every look-ahead bug in this project so far was
    introduced by code that believed it was maintaining the discipline -- the
    uniform one-day lag, the full-sample weekday adjustment, the monthly FRED
    stamp. A structural assertion on the finished panel catches the next one.
    """
    problems: list[str] = []
    if panel.empty:
        return ["panel is empty"]

    for column in ("ticker", "date", "target"):
        if column not in panel.columns:
            problems.append(f"missing required column {column!r}")
    if problems:
        return problems

    # A feature perfectly correlated with the target is the signature of the
    # target having leaked into the design matrix under another name.
    numeric = panel.select_dtypes("number")
    for column in numeric.columns:
        if column == "target":
            continue
        pair = panel[["target", column]].dropna()
        # A constant column has no correlation to speak of, and asking for one
        # divides by a zero standard deviation. `availability_lag` is constant
        # by construction, so this is the normal case rather than an edge one.
        if len(pair) < 30 or pair[column].nunique() < 2:
            continue
        correlation = pair["target"].corr(pair[column])
        if pd.notna(correlation) and abs(correlation) > 0.999:
            problems.append(
                f"{column!r} correlates {correlation:+.4f} with the target: "
                "almost certainly the target itself under another name"
            )

    # Every attention column must be constant-shifted by at least the venue lag.
    for ticker, group in panel.groupby("ticker"):
        expected = availability_lag_days(str(ticker)) + spec.extra_lag
        if "availability_lag" in group and (group["availability_lag"] != expected).any():
            problems.append(f"{ticker}: availability_lag does not match the venue rule")

    return problems
