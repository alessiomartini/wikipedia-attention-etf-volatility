# Design & research protocol

This document is written **before** the code, and is meant to be frozen before
any result is looked at. Its purpose is to make the project falsifiable: every
choice that could later be tuned to manufacture a positive result is pinned down
here first.

Status of each decision is marked **[DECIDED]**, **[PROPOSED]** (awaiting
confirmation) or **[OPEN]**.

---

## 1. The question

> Does public attention — measured by Wikipedia pageviews across languages —
> carry information about the **next day's** trading activity and realized
> volatility of the companies that attention is directed at, *beyond* what the
> price series already tells us?

Deliberately **not** the question: "can we predict returns". That is tested, but
as an exploratory third target expected to be null (§4).

## 2. Design decisions

### 2.1 Resolution: daily **[DECIDED]**

Wikimedia's per-article Pageviews API is daily-only; hourly per-article data
exists solely in the raw dumps (`dumps.wikimedia.org/other/pageviews/`), at a
cost of ~440 GB/year and a much heavier pipeline. Daily is free, goes back to
2015-07, and covers every language edition.

**Consequence to accept:** no intraday lead-lag. Questions of the form "did
Asian attention move before the European open?" are *not answerable* under this
design. That is the single upgrade path that would justify moving to hourly
later; it is out of scope now.

**Timezone discipline.** Pageview timestamps are UTC day boundaries. European
exchanges close 17:30 CET. The pageview count for UTC day `D` is only published
around `D+1` 05:00–09:00 UTC. Therefore the *earliest* tradable use of day `D`
attention is the open of day `D+1`. Any feature that uses day-`D` attention to
explain day-`D` volatility is a look-ahead bug, regardless of how the join is
written.

### 2.2 Unit of analysis: panel **[DECIDED]**

`N` companies × `T` days, with entity and time fixed effects.

Rationale, and it is the central methodological choice of the project:

- Aggregating to an index destroys the cross-sectional variation, which is where
  the information is. If Kering falls on Gucci news, attention on *Gucci* spikes
  and attention on *Hermès* does not; an index average erases that.
- **Time fixed effects absorb the market/sector factor**, so the estimated
  coefficient answers the sharp question: does *idiosyncratic* attention explain
  *idiosyncratic* volatility?
- A pooled panel is one hypothesis test on `N×T` observations, instead of `N`
  separate tests on `T` observations each. It is simultaneously more powerful
  and more honest (§6).

Standard errors clustered by **entity and by date** (two-way), because attention
shocks are correlated across firms on the same day.

### 2.3 Universe **[DECIDED]**

Two tiers, run in this order and reported separately:

**Tier 1 — pilot, pre-registered primary study.** 49 listed fashion, luxury and
premium-consumer names, where the multilingual angle has a genuine economic
interpretation. The list lives in `config/universe_luxury.yaml`; a sample:

`MC.PA` LVMH · `RMS.PA` Hermès · `KER.PA` Kering · `CFR.SW` Richemont ·
`MONC.MI` Moncler · `BRBY.L` Burberry · `1913.HK` Prada · `CPRI` Capri ·
`RL` Ralph Lauren · `TPR` Tapestry · `EL` Estée Lauder · `NKE` Nike ·
`ADS.DE` adidas · `PUM.DE` Puma · `ZAL.DE` Zalando · `LULU` · `SKX` ·
`9983.T` Fast Retailing · `HM-B.ST` H&M · `ITX.MC` Inditex · `PVH` · `VFC` ·
`DECK` · `ONON` · `BOSS.DE` Hugo Boss

**Tier 2 — generalization test.** STOXX 600 + S&P 500 constituents that have a
Wikidata entity with a stock-exchange listing. Same specification, no retuning.
A Tier-1 effect that does not survive Tier 2 is reported as such.

Infrastructure is built universe-agnostic; the pilot is a config file, not a
code path.

**The universe must be constructed objectively via Wikidata**, not by hand:
`P414` (stock exchange), `P249` (ticker symbol), `P946` (ISIN), `P452`
(industry), plus sitelinks for every language edition. This is the project's
defense against the accusation that the search terms were chosen because they
worked.

### 2.4 Languages: multilingual **[DECIDED]**

Per-entity series for `en`, `it`, `fr`, `de`, `ja`, `ko`, `zh`, `es`.

**Caveat that must not be forgotten:** Wikipedia has been blocked in mainland
China since 2019. `zh.wikipedia` measures Taiwan / Hong Kong / diaspora
readership, **not** mainland Chinese consumers — who are the dominant marginal
buyer in luxury. `zh` is a Greater-China-ex-mainland proxy and must be labelled
as such wherever it appears. `ja` and `ko` are cleaner Asian demand proxies.

Language is a *level* signal at daily resolution, not a *timing* signal. It
supports questions like "is attention on this brand shifting toward Asia?" and
"is this a local event or a global one?", not "which market reacted first".

---

## 3. Attention features

Four entity families, kept as **separate** features rather than summed, because
they are economically different signals:

| Family | Example | Interpretation |
| --- | --- | --- |
| Brand | `Gucci`, `Dior`, `Hermès` | consumer demand |
| Corporate entity | `LVMH`, `Kering` | investor / financial-news attention |
| People | CEOs, creative directors | idiosyncratic / succession risk |
| Themes | `Fashion week`, `Met Gala`, `Quiet luxury`, `Counterfeit` | sector-wide media cycle |

### 3.1 Transforms

Raw pageview counts are non-stationary and have strong weekly seasonality
(weekend traffic differs systematically). Nothing enters in levels.

- `att_i,t = log(views_i,t + 1)` then **abnormal attention**:
  `a_i,t = att_i,t − median(att_i, t-60..t-1)`, scaled by the same window's MAD.
  Median/MAD rather than mean/sd because the series is spike-dominated.
- **Attention share** — the vehicle for the *negative* correlation hypothesis:
  `share_i,t = views_i,t / Σ_j views_j,t` over the sector `j`.
  Relative by construction, mean-reverting, and it encodes "who is stealing the
  spotlight": Hermès' share falls when Gucci is in crisis, with its own absolute
  views unchanged. A negative coefficient found this way has a mechanism; one
  found by dredging does not.
- **Day-of-week adjustment** applied before differencing.

### 3.2 Data-quality handling — non-negotiable

- **Redirect resolution.** A spike may land on `Christian Dior SE` rather than
  `Dior`. Unresolved redirects silently drop a large share of the signal. All
  redirects into a canonical title are resolved and summed.
- **Bot filtering.** Compare `agent=user` against `agent=all-agents`; the filter
  is heuristic, not exact. Large `all-agents − user` divergence is itself a flag.
- **Article renames / page moves** break the series; detected via the MediaWiki
  API move log and stitched.
- **Media contamination.** An actor wearing Armani at the Oscars produces a spike
  with no financial content. The monthly Clickstream dump (referer → article)
  distinguishes search-engine traffic from internal-link traffic and is used as
  a diagnostic, not a feature.

---

## 4. Targets

Three targets **[DECIDED]**, in a strict hierarchy, all at `t+1`:

| # | Target | Role |
| --- | --- | --- |
| 1 | `log` abnormal turnover: `log(vol_t / median(vol_{t-60..t-1}))` | **validation** |
| 2 | `log` realized volatility, Garman–Klass | **primary** |
| 3 | next-day return / sign | **exploratory, expected null** |

**Target 1 is a diagnostic, not a result.** Attention and trading activity are
near-mechanically linked; if no relationship appears there, the pipeline is
broken and nothing downstream should be believed.

**Volatility estimator: Garman–Klass, not squared close-to-close returns.**
A single close-to-close return is an extremely noisy variance estimate. The
Parkinson (1980) high/low range estimator is roughly 5× more efficient and
Garman–Klass (1980), using full OHLC, roughly 7×. OHLC is already downloaded, so
this is free statistical power. With a 7× noisier target the achievable R²
ceiling collapses and a real effect can be mistaken for no effect.

Modelled in **logs**: realized volatility is right-skewed and approximately
log-normal, so `log RV` gives near-Gaussian residuals and a well-specified linear
model.

---

## 5. Baseline — the part that decides whether any result means anything

Realized volatility is autocorrelated at ~0.7–0.9 daily. A model containing
`lag_vol_1` will report a high R² that is entirely the target predicting itself.
The baseline is therefore not an empty model but **HAR** (Corsi, 2009):

```
log RV_{t+1}  ~  log RV_t  +  log RV_{t-5..t}  +  log RV_{t-22..t}
```

daily, weekly and monthly components. For turnover, the analogous
autoregressive baseline.

**The reported metric is incremental out-of-sample R²** of
`HAR + attention` over `HAR` alone — never raw R². Nested-model significance via
**Clark–West**; equal-predictive-accuracy via **Diebold–Mariano**.

Validation is walk-forward with a **purge and embargo** between train and test
folds (López de Prado), because the 5- and 22-day rolling windows in both target
and features overlap the fold boundary and would otherwise leak.

Scaling and every fitted transform live **inside** the pipeline, fitted on the
training slice only.

---

## 6. Multiple testing — the protocol

Screening `N` stocks × `M` features is the fastest available route to a false
result: 600 stocks × 30 features = 18,000 tests, of which ~900 are "significant"
at 5% by chance alone, in both directions, all of them convincing.

The panel is the primary defense: pooling turns `N` tests into **one**
coefficient estimated on `N×T` observations. Heterogeneity becomes a
second-stage question with few tests — does the coefficient vary with market
cap, analyst coverage, retail ownership? — not a per-stock fishing expedition.

Where screening *is* run, as explicit discovery:

1. Universe defined objectively via Wikidata, never hand-picked.
2. **Benjamini–Hochberg FDR** across all tests; **q-values reported, not
   p-values**.
3. **Stationary block bootstrap** (Politis–Romano) for the null distribution.
   Classical t-tests badly overstate significance on autocorrelated series.
4. **Hansen's SPA test** for "is the best of N better than chance?".
5. **Embargoed holdout**: the final 2 years are frozen and evaluated
   **exactly once**, at the end. Every specification choice is made without
   seeing them.
6. Exploratory features are declared exploratory in this document *before* being
   run, and reported separately from the pre-registered specification.

---

## 7. Explicit non-goals

- **Order-book / microstructure correlation.** There is no real-time Wikipedia
  *reading* signal. The finest reading resolution is hourly with ~1h lag; the
  only truly live stream (EventStreams SSE) carries *edits*, and a page like
  `Giorgio Armani` is edited a few times a month — empty as a high-frequency
  signal. This question is not answerable with Wikipedia data and is dropped.
- **A trading strategy.** No transaction costs, no execution model, no P&L.
  Forecasting volatility is not the same as monetizing it.
- **Causal claims.** The design identifies association with lagged ordering, not
  causation. Attention and volatility plausibly share a common driver (news).

---

## 8. Known limitations, stated up front

- Wikipedia attention is a *proxy* for a proxy: it captures encyclopedic
  curiosity, which correlates imperfectly with investor attention.
- Pageviews start 2015-07 — ~10 years, one full cycle, one crisis (2020).
- Survivorship bias in Tier 2 if constituents are taken as of today; historical
  index membership is needed for a clean test.
- `yfinance` is unstable and recent versions auto-adjust prices by default,
  which changes the meaning of the OHLC columns used for Garman–Klass.
- Some luxury names are thinly traded outside their primary listing; use the
  primary line only.

---

## 9. Data sources **[DECIDED]**

All free, no API key anywhere in the pipeline.

| Layer | Source | Role |
| --- | --- | --- |
| Attention | Wikimedia Pageviews API | daily per-article, per-language series |
| Article identity | MediaWiki Action API | canonical titles, redirect graph, page moves, QIDs |
| Universe | Wikidata SPARQL | listings (P414), tickers (P249), ISIN (P946), sitelinks |
| Market | `yfinance` | primary daily OHLCV |
| Market | Stooq CSV | independent cross-check |

**Two free price sources rather than one paid one.** `yfinance` scrapes an
undocumented endpoint and breaks periodically; the usual remedy is a vendor
subscription. But the failure mode that actually threatens the result is not
downtime — it is a silently wrong bar. An unadjusted split, a stale close or a
zero-volume placeholder raises nothing and simply changes the answer. Two
independent sources that disagree make that visible, which no single source of
any price can. Stooq is free, needs no API key, and covers European venues with
decades of history, so the cross-check costs nothing.

**The adjustment trap.** Garman–Klass uses only within-day ratios (H/L, C/O), so
a corporate-action factor applied to all four prices of a day cancels out. The
danger is a bar adjusted *inconsistently*: with `auto_adjust=False`, Yahoo
returns raw OHLC alongside a separately adjusted close, and mixing the raw high
with the adjusted close produces a meaningless range on every split day. The
ingestion layer always requests consistently adjusted OHLC, and a test asserts
the invariance directly rather than trusting the claim.

---

## 10. Open items

- **[OPEN]** Whether Tier 2 uses point-in-time index membership. Until it does,
  Tier 2 results carry survivorship bias and must be reported with it stated.
- **[OPEN]** Reconstructing the *historical* redirect graph rather than using
  the current one.
- **[OPEN]** Whether theme-article attention enters as a common factor or as a
  per-company exposure estimated in a first stage.
