"""Tests for the pageviews client."""

from __future__ import annotations

import datetime as dt

import pandas as pd

from attention_panel import pageviews
from attention_panel.mediawiki import ResolvedArticle


def _items(*pairs):
    return {"items": [{"timestamp": f"{day}00", "views": views} for day, views in pairs]}


def test_titles_are_percent_encoded_including_slashes():
    """A title like 'AC/DC' would otherwise split the URL path and 404."""
    url = pageviews._endpoint(
        "en", "AC/DC", dt.date(2020, 1, 1), dt.date(2020, 1, 2), agent="user", access="all-access"
    )
    assert "AC%2FDC" in url
    assert "/AC/DC/" not in url


def test_spaces_become_underscores():
    url = pageviews._endpoint(
        "en", "Met Gala", dt.date(2020, 1, 1), dt.date(2020, 1, 2), agent="user", access="all-access"
    )
    assert "Met_Gala" in url


def test_settled_history_is_cached_forever_and_the_tail_is_not():
    """This split is what makes repeated full runs cheap and polite."""
    windows = pageviews._split_by_settlement(
        dt.date(2015, 7, 1), dt.date(2026, 9, 8), today=dt.date(2026, 9, 8)
    )
    assert len(windows) == 2
    settled, tail = windows
    assert settled[2].ttl_seconds is None          # immutable
    assert tail[2].ttl_seconds is not None         # revisable
    assert settled[1] < tail[0]                    # no overlap, no double count


def test_a_fully_historical_range_is_a_single_immutable_window():
    windows = pageviews._split_by_settlement(
        dt.date(2015, 7, 1), dt.date(2020, 1, 1), today=dt.date(2026, 9, 8)
    )
    assert len(windows) == 1 and windows[0][2].ttl_seconds is None


def test_missing_days_are_filled_with_zero_not_left_absent(fake_client):
    """The API omits zero-traffic days; a gap and a zero mean different things.

    Left absent, every rolling window would silently span a different number of
    calendar days per article.
    """
    client = fake_client({"/Zara/": _items(("20240101", 10), ("20240104", 40))})
    series = pageviews.daily_views(
        client, "en", "Zara", dt.date(2024, 1, 1), dt.date(2024, 1, 4)
    )
    assert list(series.index) == list(pd.date_range("2024-01-01", "2024-01-04"))
    assert list(series.values) == [10, 0, 0, 40]


def test_an_article_with_no_traffic_yields_an_empty_series_not_an_error(fake_client):
    """Most brands have no article on most wikis; a 404 must not abort a run."""
    series = pageviews.daily_views(
        fake_client({}), "ko", "Nonexistent", dt.date(2024, 1, 1), dt.date(2024, 1, 4)
    )
    assert series.empty


def test_requests_before_the_apis_first_day_are_clamped(fake_client):
    """Wikimedia has no data before 2015-07-01; asking for 2010 is a 400."""
    client = fake_client({"/Zara/": _items(("20150701", 5))})
    pageviews.daily_views(client, "en", "Zara", dt.date(2010, 1, 1), dt.date(2015, 7, 1))
    requested_url = client.calls[0][0]
    assert "/20150701/" in requested_url


def test_redirect_traffic_is_summed_into_the_canonical_article(fake_client):
    """The core correctness property of the attention series.

    Readers arrive via whichever alias a headline used, so redirect traffic is
    highest on exactly the spike days that carry the signal. Ignoring it does
    not add symmetric noise -- it attenuates the events under study.
    """
    client = fake_client(
        {
            "/Christian_Dior_SE/": _items(("20240101", 100), ("20240102", 300)),
            "/Dior/": _items(("20240101", 50), ("20240102", 700)),
        }
    )
    resolved = ResolvedArticle(
        requested="Dior", canonical="Christian Dior SE", redirects=("Dior",)
    )
    frame = pageviews.article_frame(
        client, "en", resolved, dt.date(2024, 1, 1), dt.date(2024, 1, 2)
    )
    assert list(frame["views"]) == [150, 1000]


def test_bot_divergence_reports_the_filtered_share(fake_client):
    frame = pd.DataFrame(
        {"views": [50, 50], "views_all": [100, 100]},
        index=pd.date_range("2024-01-01", periods=2, name="date"),
    )
    assert pageviews.bot_divergence(frame) == 0.5
    assert pageviews.bot_divergence(pd.DataFrame(columns=["views", "views_all"])) == 0.0
