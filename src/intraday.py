"""Intraday extension: realized volatility from intraday bars (Alpaca, free IEX feed).

Daily realized variance  RV_t = sum_i r_{t,i}^2  over the intraday log returns r_{t,i} of the regular
session (Andersen-Bollerslev 1998). Much less noisy than the std of 5 daily returns, and the target is
now ONE day (no overlapping windows).  Keys are read from the environment (ALPACA_KEY, ALPACA_SECRET).
"""
import os
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from statsmodels.stats.multitest import multipletests
from . import model_pipeline as mp

BAR = "15Min"  # IEX feed is thin for some ETFs (XLE, XLF, GLD): 15-min bars are more robust than 5-min
URL = "https://data.alpaca.markets/v2/stocks/{}/bars"
BASE_I = ["rv_d_l1", "rv_w_l1", "rv_m_l1"]  # HAR-RV: day / week / month averages of past daily RV
ATTN_I = ["views_l1", "views_l2"]


def fetch_bars(ticker, start, end, bar=BAR):
    f = mp.CACHE / f"alpaca_{ticker}_{bar}_{start}_{end}.csv"
    if not f.exists():
        key, sec = os.environ.get("ALPACA_KEY"), os.environ.get("ALPACA_SECRET")
        if not (key and sec):
            raise RuntimeError("Set ALPACA_KEY and ALPACA_SECRET (free Alpaca account) in the environment.")
        hdr = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec}
        params = dict(timeframe=bar, start=f"{start}T00:00:00Z", end=f"{end}T00:00:00Z",
                      feed="iex", adjustment="all", limit=10000)
        rows, token = [], None
        while True:
            r = requests.get(URL.format(ticker), headers=hdr, params={**params, **({"page_token": token} if token else {})}, timeout=60)
            r.raise_for_status()
            j = r.json()
            rows += [(b["t"], b["c"]) for b in (j.get("bars") or [])]
            token = j.get("next_page_token")
            if not token:
                break
        mp.CACHE.mkdir(exist_ok=True)
        pd.DataFrame(rows, columns=["t", "close"]).to_csv(f, index=False)
    return pd.read_csv(f, parse_dates=["t"]).set_index("t")["close"]


def daily_rv(close):
    """Daily realized variance from intraday closes (regular session 09:30-16:00 New York).
    Overnight returns are excluded: we measure intraday variation only."""
    c = close.copy()
    c.index = c.index.tz_convert("America/New_York") if c.index.tz is not None else c.index
    c = c.between_time("09:30", "15:59")
    day = c.index.date
    r = np.log(c).groupby(day).diff()  # first bar of each day has NaN: no overnight return
    rv = (r ** 2).groupby(day).sum(min_count=1)
    n = r.groupby(day).count()
    rv = rv[n >= 10]  # drop half-days / days with too few bars
    rv.index = pd.to_datetime(rv.index)
    return rv


def build_intraday(ticker, page, start, end, close=None, views=None, rich=False):
    """X, y with HAR-RV features at t-1 and target ln RV_t (one day, non-overlapping)."""
    rv = daily_rv(fetch_bars(ticker, start, end) if close is None else close)
    rv = rv[rv > 0]
    lrv = np.log(rv)
    d = pd.DataFrame({"rv_d": lrv,
                      "rv_w": np.log(rv.rolling(5).mean()),
                      "rv_m": np.log(rv.rolling(22).mean())})
    v = mp.fetch_views(page, start, end) if views is None else views
    d["views"] = np.log(v.reindex(d.index)).diff()
    X = pd.DataFrame({f"{c}_l1": d[c].shift(1) for c in ["rv_d", "rv_w", "rv_m", "views"]})
    X["views_l2"] = d["views"].shift(2)
    if rich:
        lv = np.log(v.reindex(d.index))
        X["views_ma5_l1"] = d["views"].rolling(5).mean().shift(1)
        X["views_z_l1"] = ((lv - lv.rolling(60).mean()) / lv.rolling(60).std()).shift(1)
    data = pd.concat([lrv.rename("y"), X], axis=1).dropna()
    cols = BASE_I + ATTN_I + (mp.RICH if rich else [])
    return data[cols], data["y"]


def evaluate_intraday(ticker, page, start, end, model="ols"):
    X, y = build_intraday(ticker, page, start, end, rich=(model != "ols"))
    attn = BASE_I + ATTN_I + (mp.RICH if model != "ols" else [])
    refit = mp.REFIT if model == "ols" else 63
    pb = mp.walk_forward(X, y, BASE_I, model, refit=refit)
    pa = mp.walk_forward(X, y, attn, model, refit=refit)
    yt = y.loc[pb.index]
    res, dl = mp.compare(y, pb, pa)
    eb = (yt - pb) ** 2
    return dict(ticker=ticker, page=page, model=model, n_oos=len(yt),
                r2_oos_baseline=1 - eb.sum() / ((yt - yt.expanding().mean().shift(1).bfill()) ** 2).sum(),
                **res), dl


def main(start="2016-01-01", end="2026-10-31"):
    rows = []
    for t, p in mp.PAIRS:
        for model in ("ols", "gbm"):
            row, _ = evaluate_intraday(t, p, start, end, model)
            rows.append(row)
    df = pd.DataFrame(rows)
    for m in ("ols", "gbm"):
        i = df.model == m
        df.loc[i, "p_dm_holm"] = multipletests(df.loc[i, "p_dm"], method="holm")[1]
        if m == "ols":  # Clark-West only valid for nested linear models
            df.loc[i, "p_cw_holm"] = multipletests(df.loc[i, "p_cw"], method="holm")[1]
    df.round(5).to_csv("results/intraday_results.csv", index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
