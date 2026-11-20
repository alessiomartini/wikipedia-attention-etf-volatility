"""Panel of pre-specified pairs + earnings-window analysis (free data: Alpaca SIP, yfinance, Wikimedia).

The pair list is FIXED here before looking at any result (to avoid choosing pairs after the fact).
Earnings dates (yfinance, unofficial) are known in advance, so an `earn` dummy (reaction day of the
company's own earnings) is a legitimate control added to BOTH models, like the weekday dummies.
Reaction day = announcement day if released before the close, otherwise the next trading day.
"""
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.stats.multitest import multipletests
from . import intraday as it
from . import model_pipeline as mp

# (ETF, Wikipedia article, company ticker or None for macro articles)
# Data-quality filter applied before computing any result: articles renamed during 2016-2026 have broken
# view histories and were excluded (Amazon: zero views until 2018; Meta_Platforms: exists only from 2021).
PANEL = [("QQQ", "Nvidia", "NVDA"), ("QQQ", "Apple_Inc.", "AAPL"), ("QQQ", "Microsoft", "MSFT"),
         ("QQQ", "Alphabet_Inc.", "GOOGL"),
         ("XLK", "Apple_Inc.", "AAPL"), ("XLK", "Microsoft", "MSFT"), ("XLK", "Nvidia", "NVDA"),
         ("XLF", "JPMorgan_Chase", "JPM"), ("XLF", "Bank_of_America", "BAC"), ("XLF", "Goldman_Sachs", "GS"),
         ("XLE", "ExxonMobil", "XOM"), ("XLE", "Chevron_Corporation", "CVX"),
         ("SPY", "Federal_Reserve", None), ("XLE", "Petroleum", None), ("GLD", "Gold", None)]


def earnings_days(ticker, index):
    """Boolean Series on `index` (trading days): True on the earnings reaction day."""
    f = mp.CACHE / f"earnings_{ticker}.csv"
    if not f.exists():
        mp.CACHE.mkdir(exist_ok=True)
        d = yf.Ticker(ticker).get_earnings_dates(limit=100)
        pd.Series(d.index.astype(str)).to_csv(f, index=False)
    ts = pd.to_datetime(pd.read_csv(f).iloc[:, 0].str[:19])  # local New York time as published
    day = ts.dt.normalize() + pd.to_timedelta((ts.dt.hour >= 16).astype(int), unit="D")
    pos = index.searchsorted(day.values)  # first trading day on/after the reaction date
    pos = pos[pos < len(index)]
    out = pd.Series(False, index=index)
    out.iloc[np.unique(pos)] = True
    return out


def evaluate_pair(etf, page, comp, start, end, bar="5Min", feed="sip"):
    X, y = it.build_intraday(etf, page, start, end, bar=bar, feed=feed)
    ctl = it.BASE_I + it.DOW
    if comp:
        X["earn"] = earnings_days(comp, X.index).astype(float)
        ctl = ctl + ["earn"]
    pb = mp.walk_forward(X, y, ctl)
    pa = mp.walk_forward(X, y, ctl + it.ATTN_I)
    res, dl = mp.compare(y, pb, pa)
    row = dict(etf=etf, page=page, company=comp, n_oos=len(pb),
               r2_oos_baseline=1 - ((y.loc[pb.index] - pb) ** 2).sum() / ((y.loc[pb.index] - y.loc[pb.index].expanding().mean().shift(1).bfill()) ** 2).sum(),
               **res)
    # attention gain restricted to the company's earnings reaction days
    if comp:
        e = X["earn"].loc[pb.index] == 1
        row["n_earn_days"] = int(e.sum())
        if e.sum() >= 10:
            r_e, _ = mp.compare(y, pb[e], pa[e])
            row.update(earn_r2_gain=r_e["r2_oos_gain"], earn_r2_se=r_e["r2_gain_se"], earn_cw_stat=r_e["cw_stat"], earn_p_cw=r_e["p_cw"])
    # keep the per-date loss differential (and the CW-adjusted version) for the pooled test
    yt = y.loc[pb.index]
    cw = (yt - pb) ** 2 - ((yt - pa) ** 2 - (pb - pa) ** 2)
    return row, cw


def main(start="2016-01-01", end="2026-10-31", out="results/panel_results.csv"):
    rows, cws = [], {}
    for etf, page, comp in PANEL:
        row, cw = evaluate_pair(etf, page, comp, start, end)
        rows.append(row)
        cws[f"{etf}/{page}"] = cw
        print(etf, page, round(row["r2_oos_gain"], 4), round(row["p_cw"], 4), flush=True)
    df = pd.DataFrame(rows)
    df["p_cw_holm"] = multipletests(df["p_cw"], method="holm")[1]
    ok = df["earn_p_cw"].notna()
    df.loc[ok, "earn_p_cw_holm"] = multipletests(df.loc[ok, "earn_p_cw"], method="holm")[1]
    # pooled: cross-pair average of the CW loss differential on each date, one HAC test
    pooled = pd.concat(cws, axis=1).mean(axis=1).dropna()
    t, p = mp.hac_mean_test(pooled)
    df.attrs["pooled"] = (t, p)
    df.round(5).to_csv(out, index=False)
    pd.DataFrame([dict(test="pooled_CW_all_pairs", stat=t, p=p, n_pairs=len(PANEL), n_days=len(pooled))]).round(5).to_csv("results/panel_pooled.csv", index=False)
    print(df.round(4).to_string(index=False))
    print("pooled CW:", round(t, 3), round(p, 4))


if __name__ == "__main__":
    main()
