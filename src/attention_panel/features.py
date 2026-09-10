"""Attention transforms, and the lag discipline that keeps them honest.

THE LOOK-AHEAD BUG THIS MODULE EXISTS TO PREVENT

The design (DESIGN.md section 2.1) states that the earliest tradable use of
day-D attention is "the open of day D+1". That is right for New York and wrong
for every other venue in this universe, which was not noticed until the
features were written.

Pageview counts for UTC day D are published the following morning, around
05:00-09:00 UTC. Compare that with when each market opens, in UTC:

    Tokyo        00:00      pageviews not published yet
    Hong Kong    01:30      pageviews not published yet
    London       08:00      publication window still open -- a coin flip
    Paris/Milan  08:00      same
    New York     14:30      safely published

So a uniform one-day lag would feed Tokyo and Hong Kong a number that did not
exist when their session opened, and would give Europe one that may or may not
have. The bug is invisible in a backtest: it produces a better result, not an
error.

The fix is `availability_lag_days`, which derives the lag per venue from the
opening hour rather than assuming one. The default is deliberately conservative
-- two days everywhere except the US -- because the cost of being wrong is
asymmetric: an extra day of lag loses a little power, while a day of leakage
invalidates the study.

WHY NOTHING ENTERS IN LEVELS

Raw pageview counts are non-stationary and carry a strong weekly cycle, and
their level differs by three orders of magnitude between `Nike, Inc.` and
`Brunello Cucinelli (brand)`. A panel pooling them in levels would be dominated
by article popularity rather than by changes in attention.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Opening time of each venue in UTC hours, keyed by Yahoo exchange suffix.
#: Standard time; daylight saving shifts these by an hour but never enough to
#: change which side of the publication window a venue falls on.
EXCHANGE_OPEN_UTC_HOUR = {
    "US": 14.5,   # New York 09:30 ET
    ".L": 8.0,    # London 08:00 GMT
    ".PA": 8.0,   # Euronext Paris 09:00 CET
    ".MI": 8.0,   # Borsa Italiana
    ".MC": 8.0,   # Bolsa de Madrid
    ".DE": 8.0,   # XETRA
    ".AS": 8.0,   # Euronext Amsterdam
    ".SW": 8.0,   # SIX Swiss
    ".ST": 8.0,   # Nasdaq Stockholm
    ".CO": 8.0,   # Nasdaq Copenhagen
    ".T": 0.0,    # Tokyo 09:00 JST
    ".HK": 1.5,   # Hong Kong 09:30 HKT
}

#: The hour by which day-D pageviews can be assumed public, on D+1, in UTC.
#: Wikimedia publishes in the early morning but does not guarantee a time, so
#: this is the late end of the observed window rather than the early one. It is
#: an ASSUMPTION, and the one worth measuring first if the lag ever needs to be
#: tightened: poll the API for yesterday's counts and record when they appear.
PAGEVIEW_PUBLICATION_UTC_HOUR = 9.0


def availability_lag_days(
    ticker: str, publication_hour: float = PAGEVIEW_PUBLICATION_UTC_HOUR
) -> int:
    """Days of lag needed for attention to be knowable at a venue's open.

    Attention for pageview-day D becomes public on D+1 at `publication_hour`.
    A session opening at hour O on day t may use it when either the publication
    day falls strictly before t, or it falls on t and the data is out before
    the bell. That gives a lag of one day for venues opening after publication
    and two for those opening before it.

    An unknown suffix returns the conservative answer rather than guessing that
    it behaves like New York.
    """
    _, _, suffix = ticker.partition(".")
    key = f".{suffix}" if suffix else "US"
    open_hour = EXCHANGE_OPEN_UTC_HOUR.get(key)
    if open_hour is None:
        return 2
    return 1 if open_hour >= publication_hour else 2


def lag_to_tradable(series: pd.Series, ticker: str, extra_lag: int = 0) -> pd.Series:
    """Shift an attention series so each row is knowable at that day's open.

    `extra_lag` adds further days for the t+1 target horizon or for robustness
    checks. It never subtracts: the venue floor cannot be overridden here,
    because a caller that could pass a smaller lag would eventually pass one.
    """
    if extra_lag < 0:
        raise ValueError("extra_lag must be >= 0; the venue lag is a floor, not a default")
    return series.shift(availability_lag_days(ticker) + extra_lag)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


def log_attention(views: pd.Series) -> pd.Series:
    """log(views + 1).

    The +1 handles genuine zero-traffic days, which are real observations for a
    small brand on a small wiki, not missing data. Logs also make the series
    comparable across articles whose levels differ by orders of magnitude.
    """
    return np.log(views.astype("float64") + 1.0).rename("log_attention")


def deseasonalise_weekday(series: pd.Series, occurrences: int = 8) -> pd.Series:
    """Remove the weekly cycle using only past observations of the same weekday.

    Wikipedia traffic has a pronounced weekly shape -- weekends differ
    systematically from weekdays -- and leaving it in would put a deterministic
    seven-day oscillation into every attention feature.

    The median is taken over the previous `occurrences` instances of the SAME
    weekday, excluding the current one. Using the full sample, as a seasonal
    decomposition would, means each day is adjusted by a quantity computed
    partly from its own future: a subtle look-ahead that survives every
    train/test split, because it is baked into the feature before splitting.
    """
    values = series.astype("float64")
    adjusted = pd.Series(np.nan, index=values.index, dtype="float64")

    for weekday, group in values.groupby(values.index.dayofweek):
        baseline = group.rolling(occurrences, min_periods=2).median().shift(1)
        adjusted.loc[group.index] = group - baseline

    return adjusted.rename("attention_dow_adjusted")


def abnormal_attention(views: pd.Series, window: int = 60) -> pd.Series:
    """How unusual today's attention is, in robust standard deviations.

    `(log_attention - trailing median) / (1.4826 * trailing MAD)`

    WHY MEDIAN AND MAD RATHER THAN MEAN AND STANDARD DEVIATION: attention is
    spike-dominated, so a mean baseline is itself dragged by the event being
    measured, and a standard-deviation scale is inflated by past spikes exactly
    when the series is most interesting. The 1.4826 factor makes the MAD
    comparable to a standard deviation under normality, so the units stay
    interpretable across articles.

    WHY THE WINDOW EXCLUDES TODAY (`.shift(1)`): including the current day in
    its own reference shrinks the largest spikes the most -- it attenuates
    precisely the observations the study is about.
    """
    logged = log_attention(views)
    baseline = logged.rolling(window, min_periods=window // 2).median().shift(1)
    deviation = (logged - baseline).abs()
    mad = deviation.rolling(window, min_periods=window // 2).median().shift(1)

    scale = 1.4826 * mad
    # A flat article -- identical traffic every day in the window -- has a zero
    # MAD and no scale to divide by. That is not an infinite surprise; it is an
    # undefined one, and NaN says so where inf would dominate any regression.
    scale = scale.where(scale > 0)
    return ((logged - baseline) / scale).rename("abnormal_attention")


def attention_share(views: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Each column's share of total attention across the panel, smoothed.

    THIS IS THE VEHICLE FOR THE NEGATIVE-CORRELATION HYPOTHESIS (DESIGN.md
    section 3.1). When Gucci is in crisis, Hermes' share of sector attention
    falls with its own absolute traffic unchanged. A negative coefficient found
    this way has a mechanism behind it; one found by screening does not.

    The share is relative by construction, so it is naturally mean-reverting
    and needs no separate detrending. It is smoothed over `window` days because
    a single-day share is dominated by whichever article had a news cycle.
    """
    counts = views.astype("float64").fillna(0.0)
    total = counts.sum(axis=1)
    # A day on which the whole sector recorded no traffic is a data gap, not a
    # day of equal shares; NaN keeps it out rather than inventing 1/N.
    shares = counts.div(total.where(total > 0), axis=0)
    return shares.rolling(window, min_periods=1).mean()


def aggregate_by_family(
    series_by_article: dict[str, pd.Series], families: dict[str, str]
) -> pd.DataFrame:
    """Sum a company's article series within each attention family.

    Families stay separate rather than being pooled because they measure
    different things (DESIGN.md section 3): corporate attention is largely
    investors and financial news, brand attention is consumers, people
    attention is succession and idiosyncratic risk. Summing them would discard
    the distinction the feature design is built on.

    Summing WITHIN a family is safe in a way summing across is not: two brands
    of the same company are the same kind of signal.
    """
    columns: dict[str, pd.Series] = {}
    for title, series in series_by_article.items():
        family = families.get(title)
        if family is None:
            continue
        if family in columns:
            columns[family] = columns[family].add(series, fill_value=0.0)
        else:
            columns[family] = series.copy()
    if not columns:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
    return pd.DataFrame(columns).rename_axis("date").sort_index()
