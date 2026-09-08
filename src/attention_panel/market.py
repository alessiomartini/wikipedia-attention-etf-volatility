"""Market data: two free sources, cross-validated, and range-based volatility.

WHY TWO FREE SOURCES INSTEAD OF ONE PAID ONE

`yfinance` scrapes an undocumented endpoint; it breaks periodically and its
behaviour changes between releases. The usual remedy is a paid vendor, but for
this study two independent *free* sources are strictly better than one paid
one, because the failure mode that actually threatens the result is not
downtime -- it is a silently wrong bar. An unadjusted split, a zero-volume
placeholder or a stale close does not raise anything; it just changes the
answer. Two sources that disagree make that visible, which a single source of
any price cannot.

  * yfinance  -- broad coverage, adjusted OHLC, unstable interface.
  * Stooq     -- free CSV, no API key, decades of history including European
                 venues; used as the cross-check, not as a fallback.

THE ADJUSTMENT TRAP

The Garman-Klass estimator uses only within-day ratios, H/L and C/O, so a
corporate-action factor applied to all four prices of a day cancels out. The
danger is a bar where the four prices are adjusted INCONSISTENTLY: with
`auto_adjust=False`, Yahoo returns raw O/H/L/C alongside a separately adjusted
close, and any code that mixes the raw high with the adjusted close produces a
meaningless range on every split day. This module therefore always requests
consistently adjusted OHLC and never mixes the two conventions.
"""

from __future__ import annotations

import datetime as dt
import io
import logging
import math

import numpy as np
import pandas as pd

from .httpcache import CachePolicy, HttpClient

log = logging.getLogger(__name__)

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

#: Yahoo exchange suffix -> Stooq suffix.
#: UNVERIFIED: written without network access. `cli validate-universe` reports
#: every symbol that fails to resolve on Stooq, and this table is the first
#: place to look when it does.
STOOQ_SUFFIX = {
    "": ".us",
    ".PA": ".fr",
    ".MI": ".it",
    ".L": ".uk",
    ".DE": ".de",
    ".SW": ".ch",
    ".MC": ".es",
    ".ST": ".se",
    ".CO": ".dk",
    ".AS": ".nl",
    ".T": ".jp",
    ".HK": ".hk",
}


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def fetch_yfinance(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Daily OHLCV from Yahoo, consistently adjusted.

    `auto_adjust=True` is not a default to be inherited but a deliberate
    choice: it returns O, H, L and C all divided by the same split/dividend
    factor, which is the only form in which the range estimators below are
    meaningful. See the module docstring.
    """
    import yfinance  # imported lazily: the rest of the package must stay usable
                     # when yfinance is broken, which it periodically is.

    raw = yfinance.download(
        ticker,
        start=start.isoformat(),
        end=(end + dt.timedelta(days=1)).isoformat(),  # yfinance `end` is exclusive
        auto_adjust=True,
        progress=False,
        actions=False,
    )
    if raw is None or raw.empty:
        return _empty_ohlcv()

    # Recent yfinance returns a MultiIndex column even for a single ticker.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    frame = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
    return frame.rename_axis("date").sort_index()


def to_stooq_symbol(ticker: str) -> str | None:
    """Translate a Yahoo ticker into Stooq's convention, or None if unmapped."""
    base, _, suffix = ticker.partition(".")
    key = f".{suffix}" if suffix else ""
    mapped = STOOQ_SUFFIX.get(key)
    if mapped is None:
        return None
    # Stooq lowercases symbols and does not use Yahoo's dash convention
    # (Yahoo "HM-B.ST" is Stooq "hmb.se").
    return base.replace("-", "").lower() + mapped


def fetch_stooq(client: HttpClient, ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Daily OHLCV from Stooq's free CSV endpoint. No API key required."""
    symbol = to_stooq_symbol(ticker)
    if symbol is None:
        log.info("no Stooq mapping for %s; cross-check skipped", ticker)
        return _empty_ohlcv()

    body = client.get_text(
        "https://stooq.com/q/d/l/",
        params={"s": symbol, "d1": f"{start:%Y%m%d}", "d2": f"{end:%Y%m%d}", "i": "d"},
        policy=CachePolicy.VOLATILE,
        allow_404=True,
    )
    # Stooq answers an unknown symbol with HTTP 200 and the body "No data",
    # so the status code alone is not enough to detect failure.
    if not body or not body.lstrip().lower().startswith("date"):
        return _empty_ohlcv()

    frame = pd.read_csv(io.StringIO(body))
    frame.columns = [c.strip().lower() for c in frame.columns]
    if "volume" not in frame:
        frame["volume"] = np.nan
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.set_index("date")[OHLCV_COLUMNS].sort_index()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_ohlcv(frame: pd.DataFrame, ticker: str) -> dict[str, object]:
    """Structural checks on a price frame. Returns a report; never raises.

    Every check here corresponds to a way a free price feed has been observed
    to be wrong while looking fine. None of these would surface as an exception
    downstream -- they would surface as a slightly different regression
    coefficient, which is why they are checked explicitly and reported in the
    run log rather than silently cleaned.
    """
    report: dict[str, object] = {"ticker": ticker, "rows": int(len(frame))}
    if frame.empty:
        report["empty"] = True
        return report

    o, h, l, c, v = (frame[k] for k in OHLCV_COLUMNS)

    report["duplicate_dates"] = int(frame.index.duplicated().sum())
    report["non_positive_prices"] = int((frame[["open", "high", "low", "close"]] <= 0).any(axis=1).sum())
    report["negative_volume"] = int((v < 0).sum())
    report["nan_rows"] = int(frame[OHLCV_COLUMNS].isna().any(axis=1).sum())

    # An OHLC bar is internally inconsistent if the high is not the day's
    # maximum or the low not its minimum. A tiny tolerance absorbs the rounding
    # a vendor applies when it stores prices at two decimals.
    tol = 1e-6
    report["high_below_others"] = int((h < np.maximum(o, c) - tol).sum())
    report["low_above_others"] = int((l > np.minimum(o, c) + tol).sum())
    report["high_below_low"] = int((h < l - tol).sum())

    # A zero range means high == low: a halted, suspended or untraded day.
    # These must be excluded from the volatility estimator rather than fed to
    # it, since log(0) variance is negative infinity.
    report["zero_range_days"] = int((h <= l + tol).sum())

    # A single-day move beyond ~50% in a large-cap name is far more likely an
    # unadjusted split than a real return. This is a FLAG, not an auto-fix:
    # silently correcting prices is how a dataset becomes untraceable.
    log_ret = np.log(c / c.shift(1))
    report["extreme_moves"] = int((log_ret.abs() > 0.5).sum())
    report["extreme_move_dates"] = [
        d.date().isoformat() for d in log_ret.index[log_ret.abs() > 0.5]
    ][:10]

    report["first_date"] = frame.index.min().date().isoformat()
    report["last_date"] = frame.index.max().date().isoformat()
    report["zero_volume_days"] = int((v.fillna(0) == 0).sum())
    return report


def cross_check(primary: pd.DataFrame, secondary: pd.DataFrame, tolerance: float = 0.02) -> dict:
    """Compare two sources' closes on their common dates.

    A disagreement does not say which source is right. It says the bar cannot
    be trusted, which is the information actually needed: the affected dates go
    into the data-quality table and, if numerous, the ticker is dropped before
    any model sees it -- not after.
    """
    if primary.empty or secondary.empty:
        return {"comparable_days": 0, "mismatches": 0, "coverage": 0.0}

    common = primary.index.intersection(secondary.index)
    if len(common) == 0:
        return {"comparable_days": 0, "mismatches": 0, "coverage": 0.0}

    a = primary.loc[common, "close"].astype(float)
    b = secondary.loc[common, "close"].astype(float)
    relative = (a - b).abs() / b.abs().replace(0, np.nan)
    bad = relative > tolerance

    return {
        "comparable_days": int(len(common)),
        "coverage": float(len(common) / max(len(primary), 1)),
        "mismatches": int(bad.sum()),
        "mismatch_rate": float(bad.mean()),
        "max_relative_diff": float(relative.max(skipna=True)) if len(relative) else 0.0,
        "mismatch_dates": [d.date().isoformat() for d in common[bad]][:10],
    }


# ---------------------------------------------------------------------------
# Volatility estimators
# ---------------------------------------------------------------------------


def garman_klass_variance(frame: pd.DataFrame) -> pd.Series:
    """Daily variance from the full OHLC bar (Garman & Klass, 1980).

    WHY NOT SQUARED CLOSE-TO-CLOSE RETURNS (the estimator the first prototype
    used): a single close-to-close return is one draw, and as a variance
    estimate it is extremely noisy. Parkinson's high/low range estimator is
    roughly 5x more efficient and Garman-Klass, which uses all four prices,
    roughly 7x. The OHLC bar is already being downloaded, so this is free
    statistical power -- and it is decisive here, because with a 7x noisier
    target a real effect is indistinguishable from no effect at this sample
    size (DESIGN.md section 4).

        sigma^2 = 0.5 * ln(H/L)^2 - (2 ln 2 - 1) * ln(C/O)^2

    WHAT IT DELIBERATELY EXCLUDES: the overnight gap. Garman-Klass measures
    intraday variance only. For an attention study that omission is material,
    since company news typically arrives while the market is shut, so the gap
    is signal rather than noise -- `overnight_variance` returns it separately
    and `realized_variance` combines the two.
    """
    o, h, l, c = (frame[k].astype(float) for k in ("open", "high", "low", "close"))

    # Zero-range or non-positive bars are halted/untraded days. They are made
    # NaN rather than zero: a zero variance becomes -inf in logs and would
    # dominate every regression it entered.
    valid = (h > l) & (o > 0) & (c > 0) & (l > 0)

    hl = np.log(h.where(valid) / l.where(valid))
    co = np.log(c.where(valid) / o.where(valid))
    variance = 0.5 * hl**2 - (2.0 * math.log(2.0) - 1.0) * co**2

    # The estimator can go slightly negative on bars where the open and close
    # sit at the extremes of a narrow range. That is an artefact of the
    # estimator, not a measurement, so it is dropped rather than clipped to a
    # floor that would masquerade as an observation.
    return variance.where(variance > 0).rename("gk_variance")


def parkinson_variance(frame: pd.DataFrame) -> pd.Series:
    """Daily variance from the high/low range alone (Parkinson, 1980).

    Kept as a robustness check: it uses less information than Garman-Klass but
    makes fewer assumptions, so a result that holds under one estimator and not
    the other is an estimator artefact rather than a finding.
    """
    h, l = frame["high"].astype(float), frame["low"].astype(float)
    valid = (h > l) & (l > 0)
    hl = np.log(h.where(valid) / l.where(valid))
    variance = hl**2 / (4.0 * math.log(2.0))
    return variance.where(variance > 0).rename("parkinson_variance")


def overnight_variance(frame: pd.DataFrame) -> pd.Series:
    """Squared overnight log return, ln(O_t / C_{t-1})^2.

    Reported separately from the intraday estimator because for THIS study the
    split is substantive: attention-driven news reaches the tape at the open,
    so an estimator that ignores the gap discards the part of the day where the
    hypothesised effect should be strongest.
    """
    o, c = frame["open"].astype(float), frame["close"].astype(float)
    valid = (o > 0) & (c > 0)
    gap = np.log(o.where(valid) / c.where(valid).shift(1))
    return (gap**2).rename("overnight_variance")


def realized_variance(frame: pd.DataFrame) -> pd.Series:
    """Full-day variance: Garman-Klass intraday plus the overnight gap.

    This is the study's primary target, in variance units. The panel builder
    takes its log (DESIGN.md section 4): realized variance is right-skewed and
    approximately log-normal, so modelling it in logs gives near-Gaussian
    residuals and a correctly specified linear model.

    BOTH COMPONENTS ARE REQUIRED. If the intraday part is undefined -- a halted
    or untraded day, where high == low -- the result is NaN rather than the gap
    alone. A gap-only number is a different quantity from a full-day variance,
    and letting it into the same column would put two incommensurable
    measurements under one name: precisely the kind of silent error that shows
    up as a coefficient rather than as an exception. The first row of a sample
    is NaN for the same reason, since it has no prior close to gap from.
    """
    intraday = garman_klass_variance(frame)
    overnight = overnight_variance(frame)
    total = intraday + overnight  # NaN propagates if either side is undefined
    return total.where(total > 0).rename("realized_variance")


def close_to_close_variance(frame: pd.DataFrame) -> pd.Series:
    """Squared daily log return. Present ONLY as the naive benchmark.

    Included so the efficiency claim above can be demonstrated on this dataset
    rather than merely cited: rerunning the study on this target should widen
    every confidence interval substantially.
    """
    c = frame["close"].astype(float)
    return (np.log(c / c.shift(1)) ** 2).rename("close_to_close_variance")


# ---------------------------------------------------------------------------
# Activity
# ---------------------------------------------------------------------------


def abnormal_turnover(frame: pd.DataFrame, window: int = 60) -> pd.Series:
    """Log volume relative to its own trailing median. The validation target.

    WHY THE BASELINE WINDOW EXCLUDES TODAY (`.shift(1)`): including the current
    day in its own reference window shrinks exactly the spikes the study is
    trying to measure, and does so most on the largest ones.

    WHY MEDIAN AND NOT MEAN: volume is spike-dominated, so a mean baseline is
    itself moved by the event being measured.

    WHY LOG: raw volume is not comparable across a panel that mixes a EUR 500
    share with a EUR 3 share. The log ratio is dimensionless, which is what
    makes pooling the companies legitimate at all (DESIGN.md section 2.2).
    """
    volume = frame["volume"].astype(float).replace(0.0, np.nan)
    baseline = volume.rolling(window, min_periods=window // 2).median().shift(1)
    return np.log(volume / baseline).rename("abnormal_turnover")


def daily_log_return(frame: pd.DataFrame) -> pd.Series:
    """Close-to-close log return: the exploratory third target."""
    c = frame["close"].astype(float)
    return np.log(c / c.shift(1)).rename("log_return")


def build_market_features(frame: pd.DataFrame, turnover_window: int = 60) -> pd.DataFrame:
    """Assemble every market-side column the panel needs, in one frame."""
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "close", "volume", "log_return", "realized_variance", "gk_variance",
                "parkinson_variance", "overnight_variance", "close_to_close_variance",
                "abnormal_turnover", "log_rv",
            ]
        ).rename_axis("date")

    out = pd.DataFrame(index=frame.index)
    out["close"] = frame["close"].astype(float)
    out["volume"] = frame["volume"].astype(float)
    out["log_return"] = daily_log_return(frame)
    out["gk_variance"] = garman_klass_variance(frame)
    out["parkinson_variance"] = parkinson_variance(frame)
    out["overnight_variance"] = overnight_variance(frame)
    out["close_to_close_variance"] = close_to_close_variance(frame)
    out["realized_variance"] = realized_variance(frame)
    out["abnormal_turnover"] = abnormal_turnover(frame, turnover_window)
    # Modelled in logs; see `realized_variance`. NaN where the variance is
    # undefined, which the panel builder drops explicitly rather than filling.
    out["log_rv"] = np.log(out["realized_variance"])
    return out.rename_axis("date")


def _empty_ohlcv() -> pd.DataFrame:
    return pd.DataFrame(
        columns=OHLCV_COLUMNS, index=pd.DatetimeIndex([], name="date"), dtype="float64"
    )
