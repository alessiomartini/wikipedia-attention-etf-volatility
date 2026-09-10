"""FRED: macroeconomic series, as controls and as sources of new hypotheses.

WHY THIS EXISTS WHEN THE PANEL ALREADY HAS TIME FIXED EFFECTS

The primary specification (DESIGN.md section 2.2) includes time fixed effects,
which absorb everything common to all firms on a given day -- including every
macro variable in this module. Adding VIX or EUR/USD as a regressor there is
not merely unnecessary, it is exactly collinear with the day dummies and would
be dropped. So this module is deliberately NOT about the primary test.

It earns its place in four other ways, and two of them are hypotheses rather
than housekeeping:

1. THE AGGREGATE TEST HAS NO TIME FIXED EFFECTS. The secondary specification
   regresses sector-ETF volatility on aggregate attention. There is no
   cross-section, so nothing absorbs the market factor, and an uncontrolled
   result there is close to meaningless: attention rises on days the whole
   market is agitated.

2. REGIME INTERACTION -- a real hypothesis. `attention x VIX` is NOT absorbed
   by time fixed effects even though `VIX` alone is, because the interaction
   varies across firms within a day. It asks something the base specification
   cannot: does attention matter more when volatility is already elevated?

3. EXPOSURE HETEROGENEITY -- likewise. `attention_i x FX_exposure_i` survives
   time fixed effects and asks whether attention transmits more strongly for
   firms with more foreign revenue. For a sector of exporters selling into
   China, Japan and the US, that is a plausible mechanism rather than a fishing
   expedition.

4. ROBUSTNESS WITHOUT TIME FIXED EFFECTS. A reader who wants the specification
   without day dummies needs controls, and dropping the dummies without them
   would be indefensible.

ACCESS: the `fredgraph.csv` endpoint needs NO API KEY, which is why it is used
here in preference to the keyed REST API. Series are revised, so nothing is
cached permanently.
"""

from __future__ import annotations

import datetime as dt
import io
import logging

import numpy as np
import pandas as pd

from .httpcache import CachePolicy, HttpClient

log = logging.getLogger(__name__)

FREDGRAPH_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"


class Series:
    """A FRED series id with the metadata that decides how it may be used."""

    def __init__(self, series_id: str, label: str, frequency: str, publication_lag_days: int = 0):
        self.series_id = series_id
        self.label = label
        self.frequency = frequency
        #: Days between the date FRED stamps an observation with and the date it
        #: actually became public. Zero for daily market data, which is stamped
        #: with the day it was observed. NON-ZERO FOR EVERYTHING SURVEY-BASED,
        #: and that difference is a look-ahead bug waiting to happen -- see
        #: `to_daily`.
        self.publication_lag_days = publication_lag_days


#: Daily market series. Stamped with the day they were observed, so no
#: publication lag: the VIX close for day D is known at the close of day D.
DAILY_SERIES = (
    Series("VIXCLS", "CBOE Volatility Index", "daily"),
    Series("DGS10", "10-Year Treasury constant maturity", "daily"),
    Series("DTWEXBGS", "Trade-weighted US dollar index, broad", "daily"),
    # FX matters here more than in most equity studies: this sector is made of
    # exporters whose revenue is earned in currencies other than the one they
    # report in, so a currency move is a direct earnings shock.
    Series("DEXUSEU", "US dollars per euro", "daily"),
    Series("DEXJPUS", "Japanese yen per US dollar", "daily"),
    Series("DEXCHUS", "Chinese yuan per US dollar", "daily"),
    Series("DEXSZUS", "Swiss francs per US dollar", "daily"),
    Series("DEXUSUK", "US dollars per pound sterling", "daily"),
)

#: Survey and national-accounts series. Monthly, and published WEEKS after the
#: date they are stamped with -- the lag is what makes them safe to use.
MONTHLY_SERIES = (
    # Consumer sentiment is a plausible common driver of both attention to
    # consumer brands and their volatility, which makes it the most interesting
    # confounder in the set and the most dangerous to align carelessly.
    Series("UMCSENT", "University of Michigan consumer sentiment", "monthly", publication_lag_days=30),
    Series("CPIAUCSL", "CPI, all urban consumers", "monthly", publication_lag_days=45),
)

ALL_SERIES = DAILY_SERIES + MONTHLY_SERIES


def fetch_series(
    client: HttpClient, series_id: str, start: dt.date, end: dt.date
) -> pd.Series:
    """One FRED series as a float Series indexed by observation date.

    Missing observations arrive as "." and become NaN rather than zero. FRED
    revises published series, so nothing here is cached permanently.
    """
    body = client.get_text(
        FREDGRAPH_CSV,
        params={"id": series_id, "cosd": start.isoformat(), "coed": end.isoformat()},
        policy=CachePolicy.VOLATILE,
        allow_404=True,
    )
    if not body or not body.strip():
        return _empty(series_id)

    frame = pd.read_csv(io.StringIO(body))
    if frame.empty or len(frame.columns) < 2:
        return _empty(series_id)

    # The date column has been named DATE and observation_date at different
    # times; take the first column positionally rather than by name.
    date_column, value_column = frame.columns[0], frame.columns[1]
    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
    # FRED writes "." for a missing observation. `errors="coerce"` turns it into
    # NaN; letting it become 0.0 would put a fabricated value into a control.
    values = pd.to_numeric(frame[value_column], errors="coerce")

    out = pd.Series(values.values, index=frame[date_column], dtype="float64")
    return out[out.index.notna()].sort_index().rename_axis("date").rename(series_id)


def to_daily(
    series: pd.Series,
    calendar: pd.DatetimeIndex,
    publication_lag_days: int = 0,
    max_staleness_days: int = 7,
) -> pd.Series:
    """Align a FRED series onto a trading calendar, without look-ahead.

    THE TRAP THIS EXISTS TO AVOID

    FRED stamps a monthly observation with the FIRST DAY OF THE MONTH IT
    DESCRIBES, not the day it was published. Consumer sentiment for March is
    dated 1 March and released in late March or April. Forward-filling from the
    stamped date therefore hands the model a number weeks before anyone could
    have known it -- a look-ahead bug that no join would flag, and one that
    inflates the apparent value of exactly the confounder controls the study
    leans on.

    So the index is shifted forward by `publication_lag_days` BEFORE filling.
    Daily market series carry a lag of zero because they are stamped with the
    day they were observed.

    `max_staleness_days` bounds the forward fill. A daily series with a
    fortnight-long hole is a data problem, and carrying its last value across
    that hole would silently invent a fortnight of observations; beyond the
    bound the result stays NaN so the gap is visible.
    """
    if series.empty:
        return pd.Series(np.nan, index=calendar, dtype="float64", name=series.name)

    shifted = series.copy()
    if publication_lag_days:
        shifted.index = shifted.index + pd.Timedelta(days=publication_lag_days)

    combined = shifted.reindex(shifted.index.union(calendar)).sort_index()
    filled = combined.ffill(limit=max_staleness_days)
    return filled.reindex(calendar).rename(series.name)


def build_macro_frame(
    client: HttpClient,
    calendar: pd.DatetimeIndex,
    start: dt.date,
    end: dt.date,
    series: tuple[Series, ...] = ALL_SERIES,
) -> pd.DataFrame:
    """Every configured series, aligned to one trading calendar.

    Levels are returned as fetched. Making them stationary is the panel
    builder's job, not this module's: a rate and a volatility index need
    different transforms, and burying that choice inside the fetch layer would
    hide it from the specification.
    """
    columns = {}
    for spec in series:
        raw = fetch_series(client, spec.series_id, start, end)
        if raw.empty:
            log.warning("FRED series %s returned nothing", spec.series_id)
        columns[spec.series_id] = to_daily(raw, calendar, spec.publication_lag_days)
    return pd.DataFrame(columns, index=calendar).rename_axis("date")


def _empty(name: str) -> pd.Series:
    return pd.Series(
        dtype="float64", index=pd.DatetimeIndex([], name="date"), name=name
    )
