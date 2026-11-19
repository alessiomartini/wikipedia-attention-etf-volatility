"""Does Wikipedia attention add out-of-sample information about ETF realized volatility?

Target  y_t = log of the 5-day realized volatility (std of log returns) ending at day t.
Features: everything is known at t-1 (strictly lagged).
  baseline (HAR-RV style): log rv5_{t-1}, log rv22_{t-1}, log|r_{t-1}|
  + attention            : dlog(views)_{t-1}, dlog(views)_{t-2}
Validation: expanding-window walk-forward, refit every REFIT days, OLS.
Test: Diebold-Mariano on squared errors with Newey-West (HAC) standard errors,
      because 5-day windows overlap and make the errors autocorrelated.
"""
import time

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
import yfinance as yf
from pathlib import Path
from statsmodels.tsa.stattools import adfuller

CACHE = Path("data")
UA = {"User-Agent": "wikipedia-attention-etf-volatility (research project)"}
MIN_TRAIN, REFIT = 750, 21  # ~3 years of burn-in, refit monthly
ATTN = ["views_l1", "views_l2"]
BASE = ["rv5_l1", "rv22_l1", "absret_l1"]


def fetch_prices(ticker, start, end):
    f = CACHE / f"{ticker}_{start}_{end}.csv"
    if not f.exists():
        CACHE.mkdir(exist_ok=True)
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        df.columns = df.columns.get_level_values(0)  # newer yfinance returns a MultiIndex
        df[["Close"]].to_csv(f)
    return pd.read_csv(f, index_col=0, parse_dates=True)["Close"]


def fetch_views(page, start, end):
    f = CACHE / f"wiki_{page}_{start}_{end}.csv"
    if not f.exists():
        CACHE.mkdir(exist_ok=True)
        s, e = start.replace("-", ""), end.replace("-", "")
        url = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
               f"en.wikipedia/all-access/user/{page}/daily/{s}/{e}")
        for wait in (0, 5, 15, 40, 90):  # Wikimedia rate-limits (429): retry with back-off
            time.sleep(wait)
            r = requests.get(url, headers=UA, timeout=30)
            if r.status_code != 429:
                break
        r.raise_for_status()
        recs = [(i["timestamp"][:8], i["views"]) for i in r.json()["items"]]
        df = pd.DataFrame(recs, columns=["date", "views"])
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
        df.set_index("date").to_csv(f)
    return pd.read_csv(f, index_col=0, parse_dates=True)["views"]


RICH = ["views_ma5_l1", "views_z_l1"]  # extra attention features, only for the ML extension


def build(ticker, page, start, end, rich=False):
    """Returns (X, y) with every feature lagged at least one trading day.
    rich=True adds RICH features (5-day mean of views changes, 60-day z-score of log views)."""
    r = np.log(fetch_prices(ticker, start, end)).diff()
    rv5 = np.log(r.rolling(5).std())
    d = pd.DataFrame({"rv5": rv5, "rv22": np.log(r.rolling(22).std()),
                      "absret": np.log(r.abs() + 1e-6)})
    # Wikipedia has weekend views, ETFs don't: align on trading days, then log-diff
    views = fetch_views(page, start, end).reindex(d.index)
    d["views"] = np.log(views).diff()
    X = pd.DataFrame({f"{c}_l1": d[c].shift(1) for c in ["rv5", "rv22", "absret", "views"]})
    X["views_l2"] = d["views"].shift(2)
    if rich:
        lv = np.log(views)
        X["views_ma5_l1"] = d["views"].rolling(5).mean().shift(1)
        X["views_z_l1"] = ((lv - lv.rolling(60).mean()) / lv.rolling(60).std()).shift(1)
    data = pd.concat([rv5.rename("y"), X], axis=1).dropna()
    return data[BASE + ATTN + (RICH if rich else [])], data["y"]


def walk_forward(X, y, cols, model="ols", min_train=MIN_TRAIN, refit=REFIT):
    """Expanding-window predictions; model fitted on the past only, refit every `refit` days.
    model: "ols" (HAR-style linear) or "gbm" (gradient-boosted trees, non-linear)."""
    pred = pd.Series(np.nan, index=y.index)
    for t0 in range(min_train, len(y), refit):
        te = slice(t0, min(t0 + refit, len(y)))
        if model == "ols":
            m = sm.OLS(y.iloc[:t0], sm.add_constant(X.iloc[:t0][cols])).fit()
            pred.iloc[te] = m.predict(sm.add_constant(X.iloc[te][cols], has_constant="add"))
        else:
            from sklearn.ensemble import HistGradientBoostingRegressor
            m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=150,
                                              random_state=0).fit(X.iloc[:t0][cols], y.iloc[:t0])
            pred.iloc[te] = m.predict(X.iloc[te][cols])
    return pred.dropna()


def hac_mean_test(f, lags=5):
    """One-sided test of E[f] > 0 with Newey-West standard errors. Returns (t-stat, p-value)."""
    r = sm.OLS(np.asarray(f), np.ones(len(f))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    t = float(r.tvalues[0])
    return t, float(r.pvalues[0] / 2 if t > 0 else 1 - r.pvalues[0] / 2)


def compare(y, p_base, p_alt):
    """OOS comparison of an alternative forecast against a baseline (alt nests base).
    DM: E[e_b^2 - e_a^2] > 0.  Clark-West (2007): same, minus the noise term (p_b - p_a)^2 that
    the bigger model adds just by estimating extra parameters; the right test for nested models."""
    yt = y.loc[p_base.index]
    eb, ea = (yt - p_base) ** 2, (yt - p_alt) ** 2
    dm_t, dm_p = hac_mean_test(eb - ea)
    cw_t, cw_p = hac_mean_test(eb - (ea - (p_base - p_alt) ** 2))
    return dict(r2_oos_gain=1 - ea.sum() / eb.sum(), dm_stat=dm_t, p_dm=dm_p, cw_stat=cw_t, p_cw=cw_p), eb - ea


def evaluate(ticker, page, start, end):
    X, y = build(ticker, page, start, end)
    pb, pa = walk_forward(X, y, BASE), walk_forward(X, y, BASE + ATTN)
    yt = y.loc[pb.index]
    res, dl = compare(y, pb, pa)
    eb = (yt - pb) ** 2
    m = sm.OLS(y.iloc[:MIN_TRAIN], sm.add_constant(X.iloc[:MIN_TRAIN][BASE + ATTN])).fit()
    return dict(ticker=ticker, page=page, n_oos=len(yt),
                r2_oos_baseline=1 - eb.sum() / ((yt - yt.expanding().mean().shift(1).bfill()) ** 2).sum(),
                **res,
                adf_p_y=adfuller(y, result_object=True).pvalue,
                adf_p_views=adfuller(X["views_l1"], result_object=True).pvalue,
                coef_views_l1=m.params["views_l1"]), dl


PAIRS = [("QQQ", "Nvidia"), ("SPY", "Federal_Reserve"), ("XLE", "Petroleum"),
         ("XLF", "JPMorgan_Chase"), ("XLK", "Apple_Inc."), ("GLD", "Gold")]

def main(start="2016-01-01", end="2026-10-31"):
    from statsmodels.stats.multitest import multipletests
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Path("results").mkdir(exist_ok=True)
    rows, diffs = [], {}
    for t, p in PAIRS:
        row, dl = evaluate(t, p, start, end)
        rows.append(row)
        diffs[f"{t}/{p}"] = dl
    df = pd.DataFrame(rows)
    df["p_dm_holm"] = multipletests(df["p_dm"], method="holm")[1]
    df["p_cw_holm"] = multipletests(df["p_cw"], method="holm")[1]
    df.round(5).to_csv("results/results.csv", index=False)
    print(df.round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(df["ticker"] + " " + df["page"].str[:8], 100 * df["r2_oos_gain"])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("OOS R$^2$ gain over HAR baseline (%)")
    ax.set_title("Adding Wikipedia attention: out-of-sample gain")
    fig.tight_layout(); fig.savefig("results/oos_gain.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    for k, dl in diffs.items():
        dl.cumsum().plot(ax=ax, label=k, lw=1)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("cum. squared-error diff (>0: attention better)")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig("results/cum_error_diff.png", dpi=150)


if __name__ == "__main__":
    main()
