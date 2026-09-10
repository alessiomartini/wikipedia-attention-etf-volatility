"""Tests for the dataset store.

The re-adjustment test is the reason this module exists in the form it does.
Adjusted price series are not append-only: a split retroactively divides every
historical price, so a cache that appends fresh rows to stored ones would hold
two adjustment conventions either side of the join -- a fabricated jump at the
seam, and Garman-Klass estimates that are meaningless across it. That is the
exact failure `market.py` warns about, arriving through a cache instead of
through a mixed-column read.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from attention_panel.store import DataStore, merge_incremental


def _prices(dates, closes):
    index = pd.DatetimeIndex(pd.to_datetime(dates), name="date")
    return pd.DataFrame({"close": closes, "volume": [1e6] * len(closes)}, index=index)


# -- the split trap ---------------------------------------------------------


def test_a_retroactive_readjustment_replaces_rather_than_appends():
    """After a 2-for-1 split every historical price halves. Appending would
    leave pre-split prices before the seam and post-split ones after it."""
    stored = _prices(["2024-01-02", "2024-01-03"], [100.0, 102.0])
    fresh = _prices(["2024-01-03", "2024-01-04"], [51.0, 52.0])  # halved history

    outcome = merge_incremental(stored, fresh)

    assert outcome.action == "replaced"
    assert "re-adjusted" in outcome.detail
    # The stored pre-split price must NOT survive into the merged frame.
    assert 100.0 not in outcome.frame["close"].tolist()
    assert outcome.frame["close"].tolist() == [51.0, 52.0]


def test_an_ordinary_extension_appends():
    """The negative control: without a re-adjustment, appending is correct and
    the cheap path must actually be taken."""
    stored = _prices(["2024-01-02", "2024-01-03"], [100.0, 102.0])
    fresh = _prices(["2024-01-03", "2024-01-04"], [102.0, 103.0])

    outcome = merge_incremental(stored, fresh)

    assert outcome.action == "appended"
    assert outcome.frame["close"].tolist() == [100.0, 102.0, 103.0]
    assert list(outcome.frame.index) == list(
        pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    )


def test_tiny_rounding_differences_do_not_trigger_a_refetch():
    """A vendor storing two decimals must not look like a corporate action."""
    stored = _prices(["2024-01-03"], [102.0])
    fresh = _prices(["2024-01-03", "2024-01-04"], [102.000001, 103.0])
    assert merge_incremental(stored, fresh).action == "appended"


def test_the_check_can_be_disabled_for_genuinely_append_only_data():
    """Wikimedia never revises a settled pageview day, so there is nothing to
    detect and no price column to detect it with."""
    stored = _prices(["2024-01-03"], [100.0])
    fresh = _prices(["2024-01-03", "2024-01-04"], [51.0, 52.0])
    assert merge_incremental(stored, fresh, price_column=None).action == "appended"


def test_fresh_data_wins_on_overlapping_dates():
    """An overlapping day is a later vintage of a revisable observation."""
    stored = _prices(["2024-01-03"], [102.0])
    fresh = _prices(["2024-01-03"], [102.00001])
    merged = merge_incremental(stored, fresh).frame
    assert merged.loc[pd.Timestamp("2024-01-03"), "close"] == 102.00001


def test_an_empty_fetch_leaves_stored_data_alone():
    """A failed download must not be able to erase a good dataset."""
    stored = _prices(["2024-01-03"], [102.0])
    outcome = merge_incremental(stored, pd.DataFrame())
    assert outcome.action == "unchanged"
    assert outcome.frame.equals(stored)


# -- incremental windows ----------------------------------------------------


def test_nothing_stored_means_fetch_the_whole_window(tmp_path):
    store = DataStore(tmp_path)
    assert store.missing_window("prices", "NKE", dt.date(2020, 1, 1), dt.date(2024, 1, 1)) == (
        dt.date(2020, 1, 1),
        dt.date(2024, 1, 1),
    )


def test_only_the_tail_is_refetched(tmp_path):
    """A second run costs a few days, not ten years."""
    store = DataStore(tmp_path)
    store.save("prices", "NKE", _prices(pd.date_range("2020-01-01", "2024-01-01", freq="B"),
                                        [1.0] * len(pd.date_range("2020-01-01", "2024-01-01", freq="B"))))
    window = store.missing_window(
        "prices", "NKE", dt.date(2020, 1, 1), dt.date(2024, 1, 10), revisable_tail_days=5
    )
    assert window is not None
    assert window[0] > dt.date(2023, 12, 1)   # only the tail
    assert window[1] == dt.date(2024, 1, 10)


def test_the_last_few_days_are_always_refetched(tmp_path):
    """The tail of a series is provisional: pageviews are still being revised
    and a price bar can be corrected after the close."""
    store = DataStore(tmp_path)
    dates = pd.date_range("2024-01-01", "2024-01-31", freq="B")
    store.save("prices", "NKE", _prices(dates, [1.0] * len(dates)))

    window = store.missing_window(
        "prices", "NKE", dt.date(2024, 1, 1), dt.date(2024, 1, 31), revisable_tail_days=5
    )
    assert window is not None            # not None, despite covering the range
    assert window[0] < dt.date(2024, 1, 31)


def test_a_store_that_does_not_reach_far_enough_back_is_refetched_whole(tmp_path):
    """A gap in the middle of a series is worse than a redundant download."""
    store = DataStore(tmp_path)
    dates = pd.date_range("2022-01-03", "2024-01-01", freq="B")
    store.save("prices", "NKE", _prices(dates, [1.0] * len(dates)))

    window = store.missing_window("prices", "NKE", dt.date(2015, 7, 1), dt.date(2024, 1, 1))
    assert window == (dt.date(2015, 7, 1), dt.date(2024, 1, 1))


# -- persistence ------------------------------------------------------------


def test_a_saved_frame_round_trips(tmp_path):
    store = DataStore(tmp_path)
    frame = _prices(["2024-01-02", "2024-01-03"], [100.0, 102.0])
    store.save("prices", "NKE", frame)

    loaded = store.load("prices", "NKE")
    assert loaded is not None
    assert loaded["close"].tolist() == [100.0, 102.0]
    assert isinstance(loaded.index, pd.DatetimeIndex)


def test_keys_that_are_illegal_filenames_are_sanitised(tmp_path):
    """Tickers carry dots and dashes, and a slash would escape the directory."""
    store = DataStore(tmp_path)
    store.save("prices", "HM-B.ST", _prices(["2024-01-02"], [1.0]))
    assert store.load("prices", "HM-B.ST") is not None
    assert store.path_for("prices", "A/B").parent == tmp_path / "prices"


def test_a_corrupt_entry_is_treated_as_absent_rather_than_raising(tmp_path):
    """An interrupted write must never be able to fail a later run."""
    store = DataStore(tmp_path)
    path = store.path_for("prices", "NKE")
    path.write_bytes(b"not gzip at all")
    assert store.load("prices", "NKE") is None


def test_the_manifest_records_the_vintage(tmp_path):
    """A result that cannot say which vintage of a revised series produced it
    cannot be rerun."""
    store = DataStore(tmp_path)
    store.save("prices", "NKE", _prices(["2024-01-02"], [1.0]), source="yahoo")

    entry = store.manifest()["prices/NKE"]
    assert entry["rows"] == 1
    assert entry["first"] == "2024-01-02"
    assert entry["source"] == "yahoo"
    assert entry["fetched_at"]
    assert not store.summary().empty
