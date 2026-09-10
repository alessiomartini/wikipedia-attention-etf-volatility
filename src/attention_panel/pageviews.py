"""Wikimedia Pageviews API: the attention series.

RESOLUTION AND ITS CONSEQUENCES (DESIGN.md section 2.1)

Wikimedia exposes per-article traffic at *daily* granularity only. Hourly
per-article data exists solely in the raw dumps, at roughly 440 GB/year. Daily
is free, complete, and reaches back to 2015-07-01, which is what this study
uses -- and it fixes what the study can and cannot ask:

  * Answerable: does yesterday's attention carry information about today's
    trading activity and realized volatility?
  * NOT answerable: which geography reacted first within a day. Any lead-lag
    question below one day is out of scope by construction.

THE TIMEZONE RULE THAT PREVENTS LOOK-AHEAD BIAS

Pageview days are UTC, and the count for day D is published the following
morning, around 05:00-09:00 UTC.

An earlier version of this note said the earliest tradable use of day-D
attention is "the open of day D+1". That is true only for New York. Tokyo opens
at 00:00 UTC and Hong Kong at 01:30, both BEFORE the data exists; London and
the continental venues open at 08:00, inside the publication window. A uniform
one-day lag would therefore hand three quarters of this universe a number that
was not available when their session opened -- a bug that produces a better
backtest rather than an error.

The lag is consequently derived per venue from its opening hour, in
`features.availability_lag_days`. This module deliberately does no lagging at
all: `daily_views` returns each series stamped with its own observation date,
so that the shift happens exactly once, in one place, where it can be tested.
"""

from __future__ import annotations

import datetime as dt
import logging
from urllib.parse import quote

import pandas as pd

from .httpcache import CachePolicy, HttpClient
from .mediawiki import ResolvedArticle

log = logging.getLogger(__name__)

AQS_ROOT = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"

#: The pageviews API has no data before this date.
EARLIEST_DATE = dt.date(2015, 7, 1)

#: Days younger than this may still be revised as late-arriving request logs
#: are processed, so they are cached briefly rather than permanently. Everything
#: older is settled and never changes, which is what makes it safe -- and
#: courteous -- to re-run a full fetch as often as development requires.
SETTLED_AFTER_DAYS = 3


def daily_views(
    client: HttpClient,
    lang: str,
    article: str,
    start: dt.date,
    end: dt.date,
    *,
    agent: str = "user",
    access: str = "all-access",
) -> pd.Series:
    """Daily pageviews for one article title on one wiki.

    Args:
        agent: ``"user"`` filters out traffic classified as spiders and
            automated clients; ``"all-agents"`` is unfiltered. Note that the
            classification is a HEURISTIC, not ground truth. `article_frame`
            fetches both so their divergence can be used as a quality flag
            (DESIGN.md section 3.2).

    Returns:
        A Series of ints indexed by ``datetime64[ns]`` date, named ``views``.
        Days the API omits are genuinely zero-traffic days and are filled with
        0 rather than left absent: a missing index entry and a zero mean
        different things downstream, and only one of them is true here.
    """
    start = max(start, EARLIEST_DATE)
    if start > end:
        return _empty_series()

    frames = []
    for window_start, window_end, policy in _split_by_settlement(start, end):
        payload = client.get_json(
            _endpoint(lang, article, window_start, window_end, agent=agent, access=access),
            policy=policy,
            # A 404 here means "this title has no recorded traffic in this
            # window", which is a real answer, not a failure. Most brands have
            # no article on most wikis; raising would abort every full run.
            allow_404=True,
        )
        if payload is None:
            continue
        frames.extend(payload.get("items", ()))

    if not frames:
        return _empty_series()

    series = pd.Series(
        {pd.Timestamp(item["timestamp"][:8]): int(item["views"]) for item in frames},
        dtype="int64",
    ).sort_index()
    series.name = "views"

    # Reindex onto a complete calendar. The API omits zero days entirely, and
    # leaving them absent would make every rolling window silently span a
    # different number of calendar days per article.
    calendar = pd.date_range(series.index.min(), series.index.max(), freq="D")
    return series.reindex(calendar, fill_value=0).rename_axis("date")


def article_frame(
    client: HttpClient,
    lang: str,
    resolved: ResolvedArticle,
    start: dt.date,
    end: dt.date,
) -> pd.DataFrame:
    """Total attention for one resolved article, redirects included.

    Columns:
        views        canonical article plus every incoming redirect, agent=user.
                     This is the attention measure the study uses.
        views_all    canonical article only, agent=all-agents. A DIAGNOSTIC,
                     never a feature: a large and persistent gap between the
                     two flags an article whose traffic is bot-dominated.

    WHY REDIRECTS ARE SUMMED
    Each redirect title carries its own independent counter. Readers arrive via
    whichever alias a headline or search result used, so the redirect share of
    traffic is highest precisely on spike days. Ignoring redirects therefore
    does not add symmetric noise -- it attenuates the very events the study is
    about.

    WHY `views_all` COVERS ONLY THE CANONICAL TITLE
    Fetching both agent classes for every redirect of every title in every
    language multiplies the request count by roughly an order of magnitude for
    a diagnostic that only needs to be indicative. The canonical article
    carries the large majority of traffic, so it is a sufficient bot probe.
    """
    if resolved.canonical is None:
        return pd.DataFrame(columns=["views", "views_all"]).rename_axis("date")

    total: pd.Series | None = None
    for title in resolved.all_titles:
        series = daily_views(client, lang, title, start, end, agent="user")
        if series.empty:
            continue
        total = series if total is None else total.add(series, fill_value=0)

    if total is None:
        return pd.DataFrame(columns=["views", "views_all"]).rename_axis("date")

    all_agents = daily_views(client, lang, resolved.canonical, start, end, agent="all-agents")

    frame = pd.DataFrame({"views": total.astype("int64")})
    frame["views_all"] = all_agents.reindex(frame.index).fillna(0).astype("int64")
    return frame.rename_axis("date")


def bot_divergence(frame: pd.DataFrame) -> float:
    """Share of canonical traffic the `user` filter removed, over the sample.

    Returned as a number in [0, 1]; values near 1 mean the article's traffic is
    almost entirely classified as automated. This is a quality flag for the
    data-quality table, not a feature: the classifier is a heuristic and
    treating its output as truth would be a stronger assumption than the study
    needs to make.
    """
    if frame.empty or frame["views_all"].sum() == 0:
        return 0.0
    filtered_out = frame["views_all"].sum() - frame["views"].sum()
    return float(max(0.0, filtered_out) / frame["views_all"].sum())


def _endpoint(
    lang: str, article: str, start: dt.date, end: dt.date, *, agent: str, access: str
) -> str:
    # Article titles go in the PATH, so every reserved character must be
    # percent-encoded -- `safe=""` in particular encodes the slash. Titles like
    # "AC/DC" produce a 404 or, worse, a wrong article without this.
    encoded = quote(article.replace(" ", "_"), safe="")
    return (
        f"{AQS_ROOT}/{lang}.wikipedia/{access}/{agent}/{encoded}/daily/"
        f"{start:%Y%m%d}/{end:%Y%m%d}"
    )


def _split_by_settlement(
    start: dt.date, end: dt.date, today: dt.date | None = None
) -> list[tuple[dt.date, dt.date, CachePolicy]]:
    """Split a date range into a permanently cacheable part and a volatile tail.

    This is what makes repeated full runs cheap: ten years of settled history
    is fetched once and never again, while only the last few days are re-asked.
    """
    today = today or dt.date.today()
    cutoff = today - dt.timedelta(days=SETTLED_AFTER_DAYS)

    if end <= cutoff:
        return [(start, end, CachePolicy.IMMUTABLE)]
    if start > cutoff:
        return [(start, end, CachePolicy.VOLATILE)]
    return [
        (start, cutoff, CachePolicy.IMMUTABLE),
        (cutoff + dt.timedelta(days=1), end, CachePolicy.VOLATILE),
    ]


def _empty_series() -> pd.Series:
    return pd.Series(dtype="int64", index=pd.DatetimeIndex([], name="date"), name="views")
