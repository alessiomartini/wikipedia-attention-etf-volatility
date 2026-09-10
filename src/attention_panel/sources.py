"""Price sources behind one interface, so the cross-check can be restored.

WHY THIS MODULE EXISTS

The design wants two independent daily OHLCV sources, because the failure mode
that threatens the result is a silently wrong bar, not downtime, and two
sources disagreeing is the only way to see one. Stooq filled that role until it
put a JavaScript anti-bot wall in front of every request.

The free API tiers that could replace it (Alpha Vantage, Twelve Data, Tiingo,
Finnhub) all publish coverage claims that are hard to pin down for the tier
that costs nothing, and this universe is two-thirds non-US: Euronext Paris,
Borsa Italiana, LSE, XETRA, SIX, BME, Nasdaq Stockholm and Copenhagen, Tokyo
and Hong Kong. "Free" and "covers Borsa Italiana on the free plan" are very
different claims, and the second is the one that matters here.

So rather than pick one on a vendor's marketing page, every candidate is an
adapter behind the same interface, and `attention-panel check-sources` probes
them with one real ticker per exchange and prints what actually worked. The
same move diagnosed Stooq: make the code report instead of guessing.

WHAT THE ADAPTERS ASSUME, AND WHY THAT IS FLAGGED

Ticker conventions differ per vendor and are the likeliest thing to be wrong
here, since they were written without access to any of these APIs. Twelve Data
is addressed by ISO 10383 MIC code (XPAR, XMIL, XETR ...), which is a published
standard rather than a vendor convention and so is the safer bet; Alpha
Vantage's international suffixes are a house convention and are marked
unverified. `check-sources` is what turns either into a checked fact.

NEVER LOG A KEY. Keys come from the environment, are sent as query parameters,
and must not appear in any reason string, log line or cached file. The HTTP
cache stores the bare URL without its query string, so cached files are safe to
keep in a working tree.
"""

from __future__ import annotations

import datetime as dt
import io
import logging
import os
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from .httpcache import CachePolicy, HttpClient

log = logging.getLogger(__name__)

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

CLIENT_USER_AGENT = "attention-panel/0.1 (research; https://github.com/alessiomartini/wikipedia-attention-etf-volatility)"


@dataclass
class SourceResult:
    """A fetch attempt, carrying why it failed when it did.

    Every one of these vendors answers failures with HTTP 200 and an error
    document -- a rate-limit notice, an "invalid symbol" object, an HTML
    challenge page. A caller that sees only an empty frame cannot tell an
    unlisted symbol from an exhausted quota, and those need opposite responses.
    Stooq taught this lesson concretely: the diagnosis only became possible once
    the body prefix was carried back.
    """

    frame: pd.DataFrame
    reason: str = ""

    @property
    def ok(self) -> bool:
        return not self.frame.empty


def normalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Put every source's output into one identical shape.

    Float OHLCV, a tz-naive midnight DatetimeIndex named `date`, sorted, with
    duplicate dates dropped.

    WHY THIS IS CENTRALISED AND NOT LEFT TO EACH ADAPTER

    The whole point of a second source is comparing two frames. If one vendor
    hands back integer prices and another floats, or one a tz-aware index and
    another naive, the comparison silently changes meaning -- an index that
    fails to align produces zero overlapping days, which `cross_check` would
    then report as "nothing to compare" rather than as a bug. Normalising once
    means a disagreement between sources is always about the DATA.

    Duplicate dates are dropped keeping the last: a repeated date is a vendor
    artefact, and leaving it in would silently reindex one frame against the
    other.
    """
    out = frame.copy()
    for column in OHLCV_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("float64")

    index = pd.to_datetime(out.index)
    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)
    out.index = index.normalize()
    out = out.rename_axis("date").sort_index()
    return out[~out.index.duplicated(keep="last")][OHLCV_COLUMNS]


class PriceSource(Protocol):
    """What every price source must provide."""

    name: str

    def available(self) -> bool:
        """False when the source needs a key that is not configured."""

    def fetch(self, ticker: str, start: dt.date, end: dt.date) -> SourceResult:
        """Daily OHLCV for a Yahoo-style ticker, or an explained failure."""


# ---------------------------------------------------------------------------
# Yahoo (yfinance) -- the primary source
# ---------------------------------------------------------------------------


@dataclass
class YahooSource:
    """yfinance. Broadest coverage, no key, no quota, unstable interface."""

    name: str = "yahoo"

    def available(self) -> bool:
        return True

    def fetch(self, ticker: str, start: dt.date, end: dt.date) -> SourceResult:
        try:
            import yfinance  # lazily: the package must stay usable when this breaks
        except ImportError as exc:
            return SourceResult(_empty(), f"yfinance not installed ({exc})")

        try:
            raw = yfinance.download(
                ticker,
                start=start.isoformat(),
                end=(end + dt.timedelta(days=1)).isoformat(),  # yfinance end is exclusive
                # Not an inherited default but a deliberate choice: this returns
                # O, H, L and C all divided by the SAME split/dividend factor,
                # which is the only form in which the range estimators are
                # meaningful. Mixing raw highs with an adjusted close produces a
                # nonsense range on every split day.
                auto_adjust=True,
                progress=False,
                actions=False,
            )
        except Exception as exc:  # yfinance raises a different type every release
            return SourceResult(_empty(), f"yfinance error: {type(exc).__name__}: {exc}")

        if raw is None or raw.empty:
            return SourceResult(_empty(), "no rows returned")

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        frame = raw.rename(columns=str.lower)
        missing = [c for c in OHLCV_COLUMNS if c not in frame.columns]
        if missing:
            return SourceResult(_empty(), f"missing columns {missing}")

        return SourceResult(normalise(frame))


# ---------------------------------------------------------------------------
# Twelve Data
# ---------------------------------------------------------------------------

#: Yahoo exchange suffix -> ISO 10383 MIC code.
#: MICs are a published standard rather than a vendor convention, which makes
#: this the least guess-laden of the symbol mappings here. Still unverified
#: against the API: `check-sources` is what confirms it.
YAHOO_SUFFIX_TO_MIC = {
    ".PA": "XPAR",   # Euronext Paris
    ".MI": "XMIL",   # Borsa Italiana
    ".L": "XLON",    # London Stock Exchange
    ".DE": "XETR",   # XETRA
    ".SW": "XSWX",   # SIX Swiss Exchange
    ".MC": "XMAD",   # Bolsa de Madrid
    ".ST": "XSTO",   # Nasdaq Stockholm
    ".CO": "XCSE",   # Nasdaq Copenhagen
    ".AS": "XAMS",   # Euronext Amsterdam
    ".T": "XTKS",    # Tokyo Stock Exchange
    ".HK": "XHKG",   # Hong Kong Stock Exchange
}


@dataclass
class TwelveDataSource:
    """Twelve Data `time_series`. Free tier: an API key, a daily credit budget.

    Addressed by MIC code, so a symbol like `MC.PA` becomes `MC` on `XPAR`
    rather than relying on a vendor's own suffix spelling.
    """

    api_key: str | None = field(default_factory=lambda: os.environ.get("TWELVEDATA_API_KEY"))
    name: str = "twelvedata"
    # The documented free-tier ceiling is 8 requests/minute. Pacing at 8s keeps
    # a full universe pass inside it without ever tripping a 429, which on this
    # vendor arrives as an HTTP 200 error document rather than a status code.
    min_interval: float = 8.0
    client: HttpClient | None = None

    def available(self) -> bool:
        return bool(self.api_key)

    def _http(self) -> HttpClient:
        if self.client is None:
            self.client = HttpClient(
                CLIENT_USER_AGENT, cache_dir=".httpcache", min_interval=self.min_interval
            )
        return self.client

    def fetch(self, ticker: str, start: dt.date, end: dt.date) -> SourceResult:
        if not self.api_key:
            return SourceResult(_empty(), "TWELVEDATA_API_KEY not set")

        base, _, suffix = ticker.partition(".")
        params = {
            "symbol": base,
            "interval": "1day",
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            # 5000 daily bars is ~20 years, comfortably more than the study's
            # 2015-07 start, and avoids a silent truncation at the default page.
            "outputsize": "5000",
            "format": "JSON",
            "apikey": self.api_key,
        }
        if suffix:
            mic = YAHOO_SUFFIX_TO_MIC.get(f".{suffix}")
            if mic is None:
                return SourceResult(_empty(), f"no MIC mapped for suffix .{suffix}")
            params["mic_code"] = mic

        payload = self._http().get_json(
            "https://api.twelvedata.com/time_series",
            params=params,
            policy=CachePolicy.VOLATILE,
            allow_404=True,
        )
        if payload is None:
            return SourceResult(_empty(), "HTTP 404")

        # Errors arrive as HTTP 200 with a status field: an exhausted credit
        # budget and an unknown symbol look identical at the transport layer.
        if isinstance(payload, dict) and payload.get("status") == "error":
            return SourceResult(_empty(), f"api error {payload.get('code')}: {payload.get('message')}")

        values = payload.get("values") if isinstance(payload, dict) else None
        if not values:
            return SourceResult(_empty(), "no values in response")

        frame = pd.DataFrame(values)
        if "datetime" not in frame.columns:
            return SourceResult(_empty(), f"unexpected keys {sorted(frame.columns)}")
        frame = frame.set_index(pd.to_datetime(frame["datetime"]))
        return SourceResult(normalise(frame))


# ---------------------------------------------------------------------------
# Alpha Vantage
# ---------------------------------------------------------------------------

#: Yahoo exchange suffix -> Alpha Vantage suffix.
#: UNVERIFIED, and more of a guess than the MIC table above: these are a house
#: convention, not a standard, and Alpha Vantage's non-US coverage has always
#: been patchier than its documentation suggests. `check-sources` decides.
YAHOO_SUFFIX_TO_ALPHAVANTAGE = {
    ".L": ".LON",
    ".DE": ".DEX",
    ".PA": ".PAR",
    ".MI": ".MIL",
    ".SW": ".SWI",
    ".MC": ".MAD",
    ".ST": ".STO",
    ".CO": ".COP",
    ".AS": ".AMS",
    ".T": ".TYO",
    ".HK": ".HKG",
}


@dataclass
class AlphaVantageSource:
    """Alpha Vantage `TIME_SERIES_DAILY`. Free tier: a key and ~25 calls a DAY.

    The daily cap is the binding constraint, and it rules this source out as a
    primary: 48 tickers would take two days. As a cross-check it is still
    useful, because a cross-check does not have to run every day -- and the
    on-disk cache means a completed pass is not repeated.
    """

    api_key: str | None = field(default_factory=lambda: os.environ.get("ALPHAVANTAGE_API_KEY"))
    name: str = "alphavantage"
    min_interval: float = 13.0  # documented free tier: ~5 requests/minute
    client: HttpClient | None = None

    def available(self) -> bool:
        return bool(self.api_key)

    def _http(self) -> HttpClient:
        if self.client is None:
            self.client = HttpClient(
                CLIENT_USER_AGENT, cache_dir=".httpcache", min_interval=self.min_interval
            )
        return self.client

    def fetch(self, ticker: str, start: dt.date, end: dt.date) -> SourceResult:
        if not self.api_key:
            return SourceResult(_empty(), "ALPHAVANTAGE_API_KEY not set")

        base, _, suffix = ticker.partition(".")
        symbol = base
        if suffix:
            mapped = YAHOO_SUFFIX_TO_ALPHAVANTAGE.get(f".{suffix}")
            if mapped is None:
                return SourceResult(_empty(), f"no Alpha Vantage suffix mapped for .{suffix}")
            symbol = base + mapped

        body = self._http().get_text(
            "https://www.alphavantage.co/query",
            params={
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol,
                # `full` returns 20+ years; the default `compact` silently
                # returns only the last 100 rows, which would look like a
                # successful fetch of a very short history.
                "outputsize": "full",
                "datatype": "csv",
                "apikey": self.api_key,
            },
            policy=CachePolicy.VOLATILE,
            allow_404=True,
        )
        if body is None:
            return SourceResult(_empty(), "HTTP 404")
        if not body.strip():
            return SourceResult(_empty(), "empty response body")

        # Rate-limit notices and invalid-symbol errors come back as HTTP 200
        # with a JSON object, even when CSV was requested.
        if not body.lstrip().lower().startswith("timestamp"):
            prefix = " ".join(body.strip().split())[:160]
            return SourceResult(_empty(), f"not CSV -> {prefix!r}")

        frame = pd.read_csv(io.StringIO(body))
        frame.columns = [c.strip().lower() for c in frame.columns]
        if "timestamp" not in frame.columns:
            return SourceResult(_empty(), f"unexpected columns {list(frame.columns)}")

        frame = normalise(frame.set_index(pd.to_datetime(frame["timestamp"])))
        # Alpha Vantage only serves full history, so the window is applied here.
        window = frame.loc[str(start) : str(end)]
        if window.empty:
            return SourceResult(_empty(), "no rows in the requested window")
        return SourceResult(window)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Every cross-check candidate. `available()` decides which are usable, so a
#: user who configures no key simply gets the primary source and the
#: source-independent structural checks, with nothing failing.
CROSS_CHECK_SOURCES: tuple[type, ...] = (TwelveDataSource, AlphaVantageSource)


def available_cross_checks() -> list[PriceSource]:
    return [source for source in (cls() for cls in CROSS_CHECK_SOURCES) if source.available()]


def exchange_of(ticker: str) -> str:
    """The exchange suffix, or 'US' for an unsuffixed ticker.

    Used to probe one representative ticker per venue: coverage gaps in free
    tiers are per-exchange, not per-company, so 48 probes would mostly repeat
    the same answer eleven times.
    """
    _, _, suffix = ticker.partition(".")
    return f".{suffix}" if suffix else "US"


def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        columns=OHLCV_COLUMNS, index=pd.DatetimeIndex([], name="date"), dtype="float64"
    )
