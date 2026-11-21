# Pre-registration: replication of the earnings reaction-day effect

Registered: 2026-11-21 (the date of the merge commit of this file on `main` is the timestamp of record).
Before registration, no data for the new tickers (sample B) and no data after 2026-10-31 (sample A) were
downloaded or looked at by the author for this project. The code that runs the plan is
[`src/replicate_earnings.py`](../src/replicate_earnings.py); it reuses `src/stocks.py` unchanged.

## 1. The effect being replicated

Notebook 07 / `src/stocks.py`, 20 large caps, SIP 5-minute realized variance 2016-2026, expanding-window
walk-forward (burn-in 750 days, refit every 21 days), HAR-RV baseline (+ weekday and own-earnings dummies)
versus the same model plus *idiosyncratic* attention (lagged log-view changes minus their cross-stock mean).
Committed results (`results/stocks_pooled_idio.csv`, `results/stocks_results_idio.csv`, README):

| Quantity (idiosyncratic attention) | Value as committed |
|---|---|
| Earnings reaction-day events | 627 (stock-events) |
| Mean OOS R2 gain on reaction days, mean over the 20 stocks | +0.49% (positive for 14 of 20) |
| Pooled Clark-West test, events treated as independent | stat 2.00, p = 0.023 (one-sided) |
| Same, after averaging within calendar weeks (205 weeks) | stat 1.52, p = 0.064 |
| All days, pooled | p = 0.097; mean OOS R2 gain -0.10%, negative for 19 of 20 stocks |
| 3-day window (reaction day + 2), week-clustered | p = 0.24 |

The original authors' reading: a hint (several tests were run, none survives a correction), not a result.
No single stock survives Holm on reaction days (smallest Holm-adjusted p = 0.79). With raw (non-idiosyncratic)
attention the week-clustered reaction-day p is 0.12.

## 2. Hypothesis

H1: on the reaction day of a company's own earnings announcement, adding idiosyncratic Wikipedia attention
(views of the company's article at t-1 and t-2, relative to the cross-stock mean) to the HAR-RV baseline with
weekday and earnings dummies **reduces** out-of-sample squared error of the log realized variance forecast,
i.e. the mean Clark-West loss differential on reaction days is **> 0**.
H0: it is <= 0. One-sided, because the earlier finding and the economic story (attention anticipates news
flow) have a sign; a negative effect is not a replication.

## 3. Primary outcome and test (one test)

Exactly the specification in `src/stocks.py::evaluate_stock` with `mean_views` (idiosyncratic attention):
- target ln(RV_t), RV from SIP 5-minute bars, regular session, no subsampling (`sub=1`), days with fewer than
  10 five-minute returns dropped (as in `intraday.daily_rv`);
- features and controls as in the code (`BASE_I`, `DOW`, `earn`), attention `ATTN_I`; OLS, `MIN_TRAIN=750`, `REFIT=21`;
- loss differential: Clark-West adjusted squared-error difference;
- earnings reaction day: from yfinance dates (after 16:00 New York -> next trading day);
- **primary statistic**: stack the loss differential on reaction days over all stocks in the sample, average
  within ISO calendar weeks, one-sided t-test of mean > 0 with HAC lags=0 (`pooled_reaction_test`).
- alpha = 0.05, one-sided. There is a single primary test per sample, so no multiplicity correction is needed.

Secondary (descriptive, no claims, no correction used for decisions): mean over stocks of the reaction-day
R2 gain and number of positive stocks; per-stock Clark-West p with Holm; the same on all days and on the
3-day window; raw-attention version. These are reported in full, whatever they show.

## 4. Samples

**Sample B (primary): new tickers, same period.** The ordered candidate list in `CANDIDATES`
(`src/replicate_earnings.py`) was written down on 2026-11-21: 30 S&P 500 large caps not among the original 20,
in approximate descending market-cap order from the author's general knowledge. **Caveat: market caps were not
looked up programmatically, the order is approximate**; what is binding is the written list and order. The
sample is the first 20 candidates that survive the exclusion rules of section 5, in list order. Data window
2016-01-01 to 2026-10-31 (same as the original); every reaction day in the out-of-sample period (after the
750-day burn-in) counts. Because the calendar period overlaps with the original analysis, market-wide shocks
are shared; stock-specific data are new. Weekly clustering handles dependence within the sample only.

**Sample A (secondary, out of time): the original 20 stocks, new quarters.** Same pipeline, data 2016-01-01 to
2027-06-30, only reaction days from 2026-11-23 to 2027-06-30 count (the 2026-11-01..2026-11-20 events are
excluded because they may already have been seen). About 20 stocks x 2 quarters = ~40 events: low power;
reported as supporting evidence only and **cannot overturn** the verdict from sample B.

## 5. Exclusion rules (decided now)

Applied before computing any result, using only data-quality information, in list order, replacing with the
next candidate:
1. Wikipedia article title is a redirect, or the page was moved/renamed on or after 2016-01-01 (checked with the
   MediaWiki API move log), or the view history has zero-view days from 2016 (as was done for Amazon and Meta).
2. Fewer than 25 reaction days found in the out-of-sample period in the yfinance earnings dates.
3. Date sanity check: ratio of mean RV on reaction days to other days (`earn_rv_ratio`) below 1.5 (the original
   20 had 2.4-4.1), meaning the dates are wrong.
4. Missing SIP bars for more than 5% of trading days.
Excluded tickers are listed in the report with the rule that excluded them. No other exclusion, no outlier
removal, no winsorising. If fewer than 20 candidates survive, run with all that survive (minimum 15, otherwise
the replication is declared not run).

## 6. Data sources and run date

Wikimedia REST pageviews (`user` agent, all-access, daily); Alpaca SIP 5-minute bars (adjusted, as in
`fetch_bars`); earnings dates from yfinance (unofficial, as in the original). Data for sample B may be pulled and
analysed **no earlier than 2026-12-01**; for sample A **no earlier than 2027-07-01**. The script refuses to run
before these dates. Each sample is run **once** (re-running only for crashes, with identical specification).

## 7. Decision rule

Per sample, the replication **succeeds** if the primary test gives p < 0.05 (one-sided) **and** the mean over
stocks of the reaction-day OOS R2 gain is positive. It **fails** otherwise. Report the primary p whatever the
outcome, with the number of events and weeks. Interpretation fixed now:
- B succeeds: the original hint is supported on new stocks; still a small effect (original size +0.49% R2), not a trading signal.
- B fails: the original hint is treated as a likely false positive of multiple testing; the README will say so.
- Power note: if the true effect equalled the original estimate (stat 1.52 on 205 weeks), power at one-sided 5% would be only about 45%; a failure is therefore not proof of no effect, and with winner's-curse shrinkage power is lower.

## 8. What will NOT be changed afterwards

The sample B list and order, the exclusion rules, the model, features, controls, window, burn-in/refit, the
test and its sidedness, alpha, the week clustering, the choice of idiosyncratic (not raw) attention, the
successful-replication rule, and the run dates. Anything not listed here and done after seeing data is
exploratory and will be labelled so. Deviations (e.g. an API change forcing another data source) are
documented in the report with the reason, not silently applied.
