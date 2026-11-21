# Roadmap

<!--
Read automatically by Projects Overview (https://alessiomartini.github.io/projects-overview/):
every "##" heading is a phase, the "- [ ]" / "- [x]" checkboxes give the progress percentage,
"## Log" is the diary (one line per day, the most recent on top).
-->

## Done
- [x] Data: Wikipedia pageviews and realized volatility of the ETFs
- [x] Analysis and results of the prototype

## Polish
- [x] Reproducible pipeline (yfinance fix, UTF-8 requirements, data cache)
- [x] HAR-RV baseline vs attention model, walk-forward
- [x] Diebold-Mariano HAC test + Holm on 6 pairs, ADF
- [x] README in English with figures and the null result
- [x] Math section in the README (written by Claude at Alessio's request, to be reread)
- [x] Clark-West test (nested models) and size simulation of the tests
- [x] Notebooks 01-04 with financial glossary, visualizations, math
- [x] ML extension (gradient boosting, rich attention features): worse than HAR, no gain
- [x] Critical review in the README: assumptions, alternatives, hardest part, ML/DL ideas
- [x] Intraday extension (Alpaca IEX 15 min, daily RV, HAR-RV) + notebook 05
- [x] Consolidated SIP 5 min data (free with an Alpaca key), panel of 15 prespecified pairs with pooled test, earnings windows + notebook 06
- [x] Panel of 20 single stocks, idiosyncratic attention, 1- and 3-day earnings windows, weekly-cluster test, signature plot and subsampled RV + notebook 07
- [ ] Pre-registered replication of the earnings reaction-day effect (other stocks or next quarters)
  - [x] Written pre-registration (preregistration/earnings-reaction-replication.md) and replication script
  - [ ] Run the replication after 2026-12-01 (new sample; original sample after 2027-07-01) on the new data, without changing the plan

## Decisions and things tried
- On 2026-11-14 an automatic session had added (without real data, API returning 429) a class with Ridge, hand-written ADF/DM, CLI and synthetic tests. On 2026-11-19, with the real data downloadable, I replaced everything with a single pipeline based on statsmodels (HAR OLS, DM-HAC, Holm, ADF with p-values); removed `main.py`, `src/stats_tests.py`, the old tests and `requirements-dev.txt`. New test: `tests/test_no_lookahead.py` (pytest).
- Result: null on all 6 pairs (no significant OOS gain of attention over the HAR baseline). Added Clark-West because DM is too conservative for nested models (simulation: 0.7% vs 4.7% rejections under H0). CW is not used with GBM (not valid for trees). Intraday: IEX covers only 2-3% of the volume and starts in July 2020 (~800 test days, low power); 15 min instead of 5 because IEX is thin on XLE/XLF/GLD. Panel: Amazon and Meta excluded (Wikipedia pages renamed, broken history), filter applied before the results; earnings dates from yfinance (not official). Single stocks: 20 large caps chosen in advance (bias towards famous companies). Limits: 6 hand-picked pairs, linear model, close-to-close volatility.

## Log
- 2026-11-21: 🤖 ROADMAP translated to English (language rule: projects in English or bilingual EN/IT). 🤖 pre-registration of the replication on the earnings reaction day (20 new stocks from a fixed list, single primary test, run from 2026-12-01) and script `src/replicate_earnings.py` with synthetic tests; no new data analyzed
- 2026-11-20: 95% error bars (HAC SE of the R2 gain) in all bar charts; results regenerated, p-values unchanged. No interval excludes zero (except 1 stock out of 20 on the earnings day with raw attention)
- 2026-11-20: User-Agent of the Wikimedia requests = repo URL (no personal contact in the tracked code)
- 2026-11-19: panel of 20 stocks: with raw attention p=2e-5 (false signal from common Wikipedia traffic), with idiosyncratic attention mean gain -0.10% and 19/20 negative; earnings reaction day +0.49% R2, p=0.064 with weekly clusters (hint); earnings dates verified; subsampling changes nothing. SEC EDGAR discarded (needs an email in the User-Agent)
- 2026-11-19: SIP 5 min since 2016 (1951 test days), panel of 15 pairs, pooled test (p=0.025 but mean R2 gain -0.06%), earnings window (Nvidia p=0.18 after Holm): still no usable effect
- 2026-11-19: intraday extension: without the day-of-week control Clark-West+Holm would give QQQ/Nvidia and XLE/Petroleum as significant, with the weekly dummies no pair survives (best Holm p 0.149): weekly seasonality of visits and volatility = false signal
- 2026-11-19: notebooks 01-04, Clark-West + simulation, GBM extension, financial glossary and critical review in the README; XLK/Apple CW p=0.019 raw but 0.11 after Holm
- 2026-11-19: project finished: HAR baseline, DM-HAC+Holm on 6 pairs 2016-2026, null result, README with figures; removed the email from the User-Agent
- 2026-10-18: added the ideas from IDEE.md (single protocol)
- 2026-10-15: created the roadmap for Projects Overview
