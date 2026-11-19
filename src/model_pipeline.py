"""Does Wikipedia attention add out-of-sample information about ETF realized volatility?

Target  y_t = log of the 5-day realized volatility (std of log returns) ending at day t.
Features: everything is known at t-1 (strictly lagged).
  baseline (HAR-RV style): log rv5_{t-1}, log rv22_{t-1}, log|r_{t-1}|
  + attention            : dlog(views)_{t-1}, dlog(views)_{t-2}
Validation: expanding-window walk-forward, refit every REFIT days, OLS.
Test: Diebold-Mariano on squared errors with Newey-West (HAC) standard errors,
      because 5-day windows overlap and make the errors autocorrelated.
"""
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
        r = requests.get(url, headers=UA, timeout=30)
        r.raise_for_status()
        recs = [(i["timestamp"][:8], i["views"]) for i in r.json()["items"]]
        df = pd.DataFrame(recs, columns=["date", "views"])
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
        df.set_index("date").to_csv(f)
    return pd.read_csv(f, index_col=0, parse_dates=True)["views"]


def build(ticker, page, start, end):
    """Returns (X, y) with every feature lagged at least one trading day."""
    r = np.log(fetch_prices(ticker, start, end)).diff()
    rv5 = np.log(r.rolling(5).std())
    d = pd.DataFrame({"rv5": rv5, "rv22": np.log(r.rolling(22).std()),
                      "absret": np.log(r.abs() + 1e-6)})
    # Wikipedia has weekend views, ETFs don't: align on trading days, then log-diff
    views = fetch_views(page, start, end).reindex(d.index)
    d["views"] = np.log(views).diff()
    X = pd.DataFrame({f"{c}_l1": d[c].shift(1) for c in ["rv5", "rv22", "absret", "views"]})
    X["views_l2"] = d["views"].shift(2)
    data = pd.concat([rv5.rename("y"), X], axis=1).dropna()
    return data[BASE + ATTN], data["y"]


def walk_forward(X, y, cols):
    """Expanding-window OLS predictions, refit every REFIT days."""
    pred = pd.Series(np.nan, index=y.index)
    for t0 in range(MIN_TRAIN, len(y), REFIT):
        m = sm.OLS(y.iloc[:t0], sm.add_constant(X.iloc[:t0][cols])).fit()
        te = slice(t0, min(t0 + REFIT, len(y)))
        pred.iloc[te] = m.predict(sm.add_constant(X.iloc[te][cols], has_constant="add"))
    return pred.dropna()


def evaluate(ticker, page, start, end):
    X, y = build(ticker, page, start, end)
    pb, pa = walk_forward(X, y, BASE), walk_forward(X, y, BASE + ATTN)
    yt = y.loc[pb.index]
    eb, ea = (yt - pb) ** 2, (yt - pa) ** 2
    dl = eb - ea  # >0: attention model better
    dm = sm.OLS(dl.values, np.ones(len(dl))).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    m = sm.OLS(y.iloc[:MIN_TRAIN], sm.add_constant(X.iloc[:MIN_TRAIN][BASE + ATTN])).fit()
    p = dm.pvalues[0] / 2 if dm.params[0] > 0 else 1 - dm.pvalues[0] / 2
    r2_base = 1 - eb.sum() / ((yt - yt.expanding().mean().shift(1).bfill()) ** 2).sum()
    return dict(ticker=ticker, page=page, n_oos=len(yt),
                r2_oos_baseline=r2_base,  # HAR baseline vs expanding-mean forecast
                r2_oos_gain=1 - ea.sum() / eb.sum(),  # OOS R^2 of attention model vs baseline
                dm_stat=dm.tvalues[0], p_one_sided=p,
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
    df["p_holm"] = multipletests(df["p_one_sided"], method="holm")[1]
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
