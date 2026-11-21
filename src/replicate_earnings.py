"""Pre-registered replication of the earnings reaction-day effect (see preregistration/earnings-reaction-replication.md).

Runs EXACTLY the analysis of src/stocks.py (idiosyncratic attention, HAR-RV + weekday + own-earnings
controls in both models, expanding walk-forward, Clark-West loss differential) on a list of
(ticker, Wikipedia article) pairs, and computes the one primary test: the loss differential on earnings
reaction days, stacked over stocks, averaged within calendar weeks, one-sided HAC(0) test of mean > 0.

Usage:
  python -m src.replicate_earnings --sample new
  python -m src.replicate_earnings --sample original --events-from 2026-11-23

The pre-registered samples and dates are the constants below; the script refuses to run before the
registered earliest run date (use --allow-early only for tests on synthetic data).
"""
import argparse
import datetime as dt
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from . import intraday as it
from . import model_pipeline as mp
from . import stocks

ALPHA = 0.05  # one-sided, single primary test: no multiplicity correction

# Sample B (primary): ordered candidate list, fixed on 2026-11-21 (approximate market-cap order, S&P 500,
# not in the original 20). The first 20 candidates that pass the exclusion rules form the sample.
CANDIDATES = [("ORCL", "Oracle_Corporation"), ("V", "Visa_Inc."), ("LLY", "Eli_Lilly_and_Company"),
              ("MA", "Mastercard"), ("PG", "Procter_&_Gamble"), ("HD", "Home_Depot"), ("COST", "Costco"),
              ("ABBV", "AbbVie"), ("UNH", "UnitedHealth_Group"), ("MRK", "Merck_&_Co."), ("IBM", "IBM"),
              ("QCOM", "Qualcomm"), ("TXN", "Texas_Instruments"), ("CAT", "Caterpillar_Inc."),
              ("HON", "Honeywell"), ("LMT", "Lockheed_Martin"), ("AMGN", "Amgen"), ("SBUX", "Starbucks"),
              ("NKE", "Nike,_Inc."), ("VZ", "Verizon_Communications"), ("T", "AT&T"), ("CMCSA", "Comcast"),
              ("WFC", "Wells_Fargo"), ("C", "Citigroup"), ("MS", "Morgan_Stanley"),
              ("AXP", "American_Express"), ("LOW", "Lowe's"), ("UPS", "United_Parcel_Service"),
              ("ABT", "Abbott_Laboratories"), ("UNP", "Union_Pacific_Corporation")]
MIN_EVENTS, MIN_RV_RATIO, N_SAMPLE = 25, 1.5, 20

SAMPLES = {  # name -> (pairs, data start, data end, first allowed reaction day, last, earliest run date)
    "new": (CANDIDATES, "2016-01-01", "2026-10-31", None, None, dt.date(2026, 12, 1)),
    "original": (stocks.STOCKS, "2016-01-01", "2027-06-30", "2026-11-23", "2027-06-30", dt.date(2027, 7, 1)),
}


def pooled_reaction_test(cws, e1s, events_from=None, events_to=None):
    """Primary test. cws/e1s: {ticker: loss differential / reaction-day mask}. Returns dict with n events,
    n weeks, t-stat, one-sided p (E[cw] > 0, HAC lags=0 on weekly means)."""
    parts = {}
    for tk in cws:
        m = e1s[tk].reindex(cws[tk].index, fill_value=False)
        if events_from:
            m &= cws[tk].index >= pd.Timestamp(events_from)
        if events_to:
            m &= cws[tk].index <= pd.Timestamp(events_to)
        parts[tk] = cws[tk][m]
    v = pd.concat(parts).dropna()
    if len(v) < 3:
        return dict(n_events=len(v), n_weeks=0, stat=np.nan, p=np.nan)
    d = v.index.get_level_values(1)
    iso = d.isocalendar()
    w = v.groupby([iso.year.values, iso.week.values]).mean()
    t, p = mp.hac_mean_test(w, lags=0)
    return dict(n_events=len(v), n_weeks=len(w), stat=t, p=p)


def analyse(D, pairs, events_from=None, events_to=None):
    """D: {ticker: (X, y)} as returned by intraday.build_intraday. Returns (per-stock DataFrame, primary dict)."""
    mean_views = pd.concat({t: D[t][0][it.ATTN_I] for t in D}).groupby(level=1).mean()
    rows, cws, e1s = [], {}, {}
    for tk, page in pairs:
        row, cw, e1, _ = stocks.evaluate_stock(tk, page, *D[tk], mean_views=mean_views)
        rows.append(row); cws[tk] = cw; e1s[tk] = e1
    df = pd.DataFrame(rows)
    primary = pooled_reaction_test(cws, e1s, events_from, events_to)
    ok = df["earn1_p_cw"].notna()  # per-stock results are descriptive only (Holm reported, no claims)
    df.loc[ok, "earn1_p_cw_holm"] = multipletests(df.loc[ok, "earn1_p_cw"], method="holm")[1]
    primary["mean_stock_r2_gain"] = float(df["earn1_r2_gain"].mean())
    primary["n_stocks_positive"] = int((df["earn1_r2_gain"] > 0).sum())
    primary["n_stocks"] = int(ok.sum())
    primary["success"] = bool(primary["p"] < ALPHA and primary["mean_stock_r2_gain"] > 0)
    return df, primary


def select_new_sample(candidates=CANDIDATES, excluded=()):
    """First N_SAMPLE candidates not in `excluded` (tickers dropped by the written exclusion rules)."""
    return [c for c in candidates if c[0] not in set(excluded)][:N_SAMPLE]


def main(sample="new", out=None, allow_early=False, exclude=()):
    pairs, start, end, ev_from, ev_to, run_from = SAMPLES[sample]
    if dt.date.today() < run_from and not allow_early:
        raise SystemExit(f"Pre-registered earliest run date for sample '{sample}' is {run_from}.")
    if sample == "new":
        pairs = select_new_sample(pairs, exclude)
    D = {tk: it.build_intraday(tk, pg, start, end, bar="5Min", feed="sip") for tk, pg in pairs}
    df, primary = analyse(D, pairs, ev_from, ev_to)
    out = out or f"results/replication_{sample}"
    df.round(5).to_csv(out + "_stocks.csv", index=False)
    pd.DataFrame([primary]).round(5).to_csv(out + "_primary.csv", index=False)
    print(pd.DataFrame([primary]).round(4).to_string(index=False))
    print(df[["ticker", "n_earn", "earn_rv_ratio", "earn1_r2_gain", "earn1_p_cw"]].round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", choices=SAMPLES, default="new")
    ap.add_argument("--exclude", nargs="*", default=[], help="tickers dropped by the written exclusion rules")
    ap.add_argument("--allow-early", action="store_true", help="tests only")
    a = ap.parse_args()
    main(a.sample, exclude=a.exclude, allow_early=a.allow_early)
