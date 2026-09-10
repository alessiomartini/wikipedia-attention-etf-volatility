"""Tests for the price-source adapters.

Every one of these vendors reports failures as HTTP 200 with an error document
-- a rate-limit notice, an invalid-symbol object, an HTML challenge page. An
adapter that maps all of those onto "empty frame" makes the difference between
an unlisted symbol and an exhausted quota invisible, and those need opposite
responses. That is the lesson Stooq taught the hard way, so it is tested here.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from attention_panel import sources

START = dt.date(2024, 1, 1)
END = dt.date(2024, 1, 5)


def _twelvedata(fake_client, payload, key="k"):
    source = sources.TwelveDataSource(api_key=key)
    source.client = fake_client({"twelvedata": payload})
    return source


def _alphavantage(fake_client, body, key="k"):
    source = sources.AlphaVantageSource(api_key=key)
    source.client = fake_client({"alphavantage": body})
    return source


# -- configuration ----------------------------------------------------------


def test_a_source_without_a_key_is_unavailable_rather_than_failing_later():
    """A user who configures nothing gets the primary source and no errors."""
    assert not sources.TwelveDataSource(api_key=None).available()
    assert not sources.AlphaVantageSource(api_key=None).available()
    assert sources.YahooSource().available()


def test_an_unconfigured_source_explains_itself_instead_of_returning_empty():
    result = sources.TwelveDataSource(api_key=None).fetch("NKE", START, END)
    assert not result.ok
    assert "TWELVEDATA_API_KEY" in result.reason


# -- symbol mapping ---------------------------------------------------------


@pytest.mark.parametrize(
    "ticker, mic",
    [("MC.PA", "XPAR"), ("MONC.MI", "XMIL"), ("BRBY.L", "XLON"),
     ("ADS.DE", "XETR"), ("9983.T", "XTKS"), ("1913.HK", "XHKG")],
)
def test_twelvedata_addresses_venues_by_mic_code(fake_client, ticker, mic):
    """MICs are an ISO standard, not a vendor spelling -- the safer mapping."""
    source = _twelvedata(fake_client, {"values": [
        {"datetime": "2024-01-02", "open": "1", "high": "2", "low": "0.5", "close": "1.5", "volume": "10"}
    ]})
    assert source.fetch(ticker, START, END).ok
    assert source.client.calls[0][1]["mic_code"] == mic
    assert source.client.calls[0][1]["symbol"] == ticker.split(".")[0]


def test_an_unmapped_exchange_is_refused_rather_than_guessed(fake_client):
    """Sending a bare symbol would cross-check against a different company."""
    source = _twelvedata(fake_client, {"values": []})
    result = source.fetch("ABC.XYZ", START, END)
    assert not result.ok and "no MIC mapped" in result.reason


def test_a_us_ticker_carries_no_exchange_qualifier(fake_client):
    source = _twelvedata(fake_client, {"values": [
        {"datetime": "2024-01-02", "open": "1", "high": "2", "low": "0.5", "close": "1.5", "volume": "10"}
    ]})
    source.fetch("NKE", START, END)
    assert "mic_code" not in source.client.calls[0][1]


# -- failures that arrive as HTTP 200 ---------------------------------------


def test_twelvedata_surfaces_an_exhausted_credit_budget(fake_client):
    """A spent quota and an unknown symbol are indistinguishable by status."""
    source = _twelvedata(fake_client, {
        "status": "error", "code": 429, "message": "You have run out of API credits",
    })
    result = source.fetch("NKE", START, END)
    assert not result.ok
    assert "429" in result.reason and "credits" in result.reason


def test_alphavantage_surfaces_a_rate_limit_note_returned_as_json(fake_client):
    """CSV was requested; a JSON body means the request did not succeed."""
    source = _alphavantage(fake_client, '{"Information": "rate limit is 25 requests per day"}')
    result = source.fetch("NKE", START, END)
    assert not result.ok
    assert "not CSV" in result.reason and "25 requests per day" in result.reason


def test_alphavantage_requests_full_history_not_the_hundred_row_default(fake_client):
    """`compact` returns 100 rows and looks exactly like a successful fetch."""
    source = _alphavantage(fake_client, "timestamp,open,high,low,close,volume\n2024-01-02,1,2,0.5,1.5,10\n")
    source.fetch("NKE", START, END)
    assert source.client.calls[0][1]["outputsize"] == "full"


# -- parsing ----------------------------------------------------------------


def test_twelvedata_rows_become_a_sorted_numeric_frame(fake_client):
    """The API returns newest-first strings; everything downstream assumes
    oldest-first floats."""
    source = _twelvedata(fake_client, {"values": [
        {"datetime": "2024-01-03", "open": "2", "high": "3", "low": "1", "close": "2.5", "volume": "20"},
        {"datetime": "2024-01-02", "open": "1", "high": "2", "low": "0.5", "close": "1.5", "volume": "10"},
    ]})
    frame = source.fetch("NKE", START, END).frame

    assert list(frame.index) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert frame["close"].tolist() == [1.5, 2.5]
    assert frame["open"].dtype.kind == "f"


def test_alphavantage_rows_outside_the_window_are_dropped(fake_client):
    """It only serves full history, so the window is applied on our side."""
    body = (
        "timestamp,open,high,low,close,volume\n"
        "2024-01-10,9,9,9,9,1\n"      # after END
        "2024-01-03,2,3,1,2.5,20\n"
        "2023-12-01,1,1,1,1,1\n"      # before START
    )
    frame = _alphavantage(fake_client, body).fetch("NKE", START, END).frame
    assert list(frame.index) == [pd.Timestamp("2024-01-03")]


# -- helpers ----------------------------------------------------------------


@pytest.mark.parametrize(
    "ticker, exchange", [("NKE", "US"), ("MC.PA", ".PA"), ("1913.HK", ".HK")]
)
def test_exchange_of(ticker, exchange):
    assert sources.exchange_of(ticker) == exchange
