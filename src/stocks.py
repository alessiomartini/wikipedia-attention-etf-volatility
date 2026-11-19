"""Single-stock panel: does attention to a company predict the volatility of THAT company's own stock?

Cleaner than ETF/article pairs (an ETF dilutes one company's news) and gives 20 independent earnings
cycles. List fixed before looking at any result. Same method as src/panel.py: HAR-RV on SIP 5-min
realized variance, weekday + own-earnings dummies in both models, Clark-West, Holm, pooled tests.
"""
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from . import intraday as it
from . import model_pipeline as mp
from .panel import earnings_days

# (ticker, Wikipedia article)
STOCKS = [("AAPL", "Apple_Inc."), ("MSFT", "Microsoft"), ("NVDA", "Nvidia"), ("GOOGL", "Alphabet_Inc."),
          ("JPM", "JPMorgan_Chase"), ("BAC", "Bank_of_America"), ("GS", "Goldman_Sachs"),
          ("XOM", "ExxonMobil"), ("CVX", "Chevron_Corporation"), ("WMT", "Walmart"),
          ("KO", "The_Coca-Cola_Company"), ("PFE", "Pfizer"), ("JNJ", "Johnson_&_Johnson"),
          ("DIS", "The_Walt_Disney_Company"), ("NFLX", "Netflix"), ("INTC", "Intel"),
          ("CSCO", "Cisco_Systems"), ("BA", "Boeing"), ("MCD", "McDonald's"), ("PEP", "PepsiCo")]


def load_all(start, end, sub=1):
    """{ticker: (X, y)} plus the cross-sectional mean of the lagged views features."""
    D = {tk: it.build_intraday(tk, pg, start, end, bar="5Min", feed="sip", sub=sub) for tk, pg in STOCKS}
    mean_views = pd.concat({t: D[t][0][it.ATTN_I] for t in D}).groupby(level=1).mean()
    return D, mean_views


def evaluate_stock(tk, page, X, y, mean_views=None):
    """mean_views given -> idiosyncratic attention: views features minus the cross-stock mean on the same day.
    This removes market-wide calendar effects (holidays, summer, Wikipedia-wide traffic shifts)."""
    X = X.copy()
    if mean_views is not None:
        X[it.ATTN_I] = X[it.ATTN_I] - mean_views.reindex(X.index)[it.ATTN_I]
    earn = earnings_days(tk, X.index)
    X["earn"] = earn.astype(float)
    # multi-day window: reaction day and the two following trading days
    win = earn | earn.shift(1, fill_value=False) | earn.shift(2, fill_value=False)
    ctl = it.BASE_I + it.DOW + ["earn"]
    pb = mp.walk_forward(X, y, ctl)
    pa = mp.walk_forward(X, y, ctl + it.ATTN_I)
    yt = y.loc[pb.index]
    res, _ = mp.compare(y, pb, pa)
    cw = (yt - pb) ** 2 - ((yt - pa) ** 2 - (pb - pa) ** 2)
    e1, e3 = earn.loc[pb.index], win.loc[pb.index]
    row = dict(ticker=tk, page=page, n_oos=len(pb), r2_oos_baseline=1 - ((yt - pb) ** 2).sum() / ((yt - yt.expanding().mean().shift(1).bfill()) ** 2).sum(),
               **res, n_earn=int(e1.sum()),
               # sanity check of the earnings dates: RV on reaction days vs all other days
               earn_rv_ratio=float(np.exp(y.loc[earn[earn].index].mean() - y.loc[~earn].mean())))
    for name, m in (("earn1", e1), ("earn3", e3)):
        if m.sum() >= 10:
            r, _ = mp.compare(y, pb[m], pa[m])
            row.update({f"{name}_r2_gain": r["r2_oos_gain"], f"{name}_cw_stat": r["cw_stat"], f"{name}_p_cw": r["p_cw"]})
    return row, cw, e1, e3


def run(D, mean_views, idio):
    rows, cws, e1s, e3s = [], {}, {}, {}
    for tk, page in STOCKS:
        X, y = D[tk]
        row, cw, e1, e3 = evaluate_stock(tk, page, X, y, mean_views if idio else None)
        rows.append(row); cws[tk] = cw; e1s[tk] = e1; e3s[tk] = e3
    df = pd.DataFrame(rows)
    df["p_cw_holm"] = multipletests(df["p_cw"], method="holm")[1]
    for k in ("earn1", "earn3"):
        df[f"{k}_p_cw_holm"] = multipletests(df[f"{k}_p_cw"], method="holm")[1]
    C = pd.concat(cws, axis=1)
    pooled = [dict(test="pooled_all_days", n=len(C), **dict(zip(("stat", "p"), mp.hac_mean_test(C.mean(axis=1).dropna()))))]
    for k, E in (("pooled_earn_reaction_day", e1s), ("pooled_earn_3day_window", e3s)):
        v = pd.concat({t: cws[t][E[t]] for t in cws}).dropna()  # all stock-earnings-days stacked
        pooled.append(dict(test=k, n=len(v), **dict(zip(("stat", "p"), mp.hac_mean_test(v, lags=0)))))
        # earnings days cluster in the same weeks (shared market shocks): average within calendar week, then test
        w = v.groupby([v.index.get_level_values(1).isocalendar().year.values, v.index.get_level_values(1).isocalendar().week.values]).mean()
        pooled.append(dict(test=k + "_week_clustered", n=len(w), **dict(zip(("stat", "p"), mp.hac_mean_test(w, lags=0)))))
    return df, pd.DataFrame(pooled)


def main(start="2016-01-01", end="2026-10-31"):
    D, mv = load_all(start, end)
    for idio, tag in ((False, "raw"), (True, "idio")):
        df, pooled = run(D, mv, idio)
        df.round(5).to_csv(f"results/stocks_results_{tag}.csv", index=False)
        pooled.round(5).to_csv(f"results/stocks_pooled_{tag}.csv", index=False)
        print(tag, "mean R2 gain %.4f" % df.r2_oos_gain.mean(), "| n R2gain<0:", int((df.r2_oos_gain < 0).sum()))
        print(pooled.round(4).to_string(index=False))
        print(df[["ticker", "r2_oos_gain", "p_cw", "p_cw_holm", "earn_rv_ratio"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
