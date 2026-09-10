# Wikipedia attention and next-day trading activity

**Does public attention to a company — measured by Wikipedia pageviews across
eight languages — carry information about its next-day trading activity and
realized volatility, beyond what the price series already says?**

The interesting part of this project is not the answer. It is the set of
decisions taken *before* looking at any result, each of which exists to stop the
study from fooling itself. They are recorded in **[DESIGN.md](DESIGN.md)**, and
this README explains the reasoning behind the ones that shaped the code.

> **Status: ingestion stage complete, modelling stage not started.**
> What exists is the data layer — universe construction, multilingual attention
> series, cross-validated prices, and the volatility estimators — with 49 tests.
> The panel builder, the HAR baseline and the significance machinery described
> below are specified but not yet written. Nothing in this repository has
> produced a research result, and no result should be inferred from it.

---

## Why the design looks like this

### The unit of analysis is a panel, not an index

The obvious move, when one stock looks too noisy, is to aggregate into an index.
That is half right: it reduces noise, and it destroys the information.

If Kering falls on bad Gucci news, attention on *Gucci* spikes and attention on
*Hermès* does not. An index average erases exactly that difference. So the study
is a firm-level panel — ~49 companies × ~2,500 days ≈ 120,000 observations —
with **entity and time fixed effects**. The time effects absorb the market and
sector factor, which sharpens the question into the one worth asking: does
*idiosyncratic* attention explain *idiosyncratic* volatility?

The panel also happens to be the answer to the multiple-testing problem, which
is the subtler reason for it. See below.

### Daily resolution, and what that forecloses

Wikimedia publishes per-article traffic **daily only**. Hourly per-article data
exists solely in the raw dumps, at ~440 GB/year. Daily is free, complete, and
reaches back to 2015-07-01.

This decides what the study can ask. *Answerable:* does yesterday's attention
inform today's volatility. *Not answerable:* which geography reacted first
within a day, or anything about order-book microstructure — there is no
real-time Wikipedia *reading* signal at all, and the only live stream
(EventStreams) carries page *edits*, which for a company article number a few
per month. That question is listed as a non-goal rather than left to resurface.

The timezone rule that follows is strict, and subtler than it first looks.
Pageview days are UTC and the count for day `D` is published around `D+1`
05:00–09:00 UTC — but Tokyo opens at 00:00 UTC and Hong Kong at 01:30, *before
the data exists*, while London and the continental venues open at 08:00, inside
the publication window. Only New York, at 14:30, is safely after it.

An earlier version of this design said "the open of day `D+1`" for everyone.
That would have handed three quarters of the universe a number that was not
available when its session opened — a bug that produces a **better** backtest
rather than an error. The lag is therefore derived per venue from its opening
hour, and defaults to two days everywhere except the US.

### Eight languages, with one large caveat

The same company is one Wikidata item with a different article title per
language, so the multilingual series come for free once the sitelinks are
resolved. The language split is a *level* signal — is attention on this brand
shifting toward Asia? — not a timing signal, which daily resolution cannot
support.

**Wikipedia has been blocked in mainland China since 2019.** `zh.wikipedia`
therefore measures Taiwan, Hong Kong and diaspora readership, **not** the
mainland luxury consumer who is the sector's dominant marginal buyer. Building
a "Chinese demand" thesis on `zh` would be measuring something else entirely.
It is labelled a Greater-China-ex-mainland proxy everywhere it appears; `ja`
and `ko` are the cleaner Asian signals.

### Three targets, in a hierarchy

| # | Target | Role |
|---|---|---|
| 1 | log abnormal turnover, `log(vol_t / median(vol_{t-60..t-1}))` | **validation** |
| 2 | log realized variance (Garman–Klass + overnight gap), `t+1` | **primary** |
| 3 | next-day return / sign | **exploratory, expected null** |

Target 1 is a *diagnostic, not a result*: attention and trading activity are
near-mechanically linked, so if nothing shows up there the pipeline is broken
and nothing downstream should be believed. Target 3 is expected to be null and
will be reported as null; attention is symmetric, and good news and bad news
both generate it.

### Garman–Klass, not squared close-to-close returns

A single close-to-close return is one draw, and as a variance estimate it is
extremely noisy. Parkinson's high/low range estimator is roughly **5× more
efficient**, and Garman–Klass, using all four prices, roughly **7×**. The OHLC
bar is already being downloaded, so this is free statistical power — and it is
decisive, because with a 7× noisier target a real effect at this sample size is
indistinguishable from no effect.

Garman–Klass measures *intraday* variance and ignores the overnight gap. For an
attention study that omission is material, since company news mostly arrives
while the market is shut, so `realized_variance` adds the gap back explicitly
and requires **both** components to be defined — a halted day yields `NaN`
rather than a gap-only number that would sit in the same column as a
full-day measurement while meaning something different.

### The baseline is HAR, and the metric is incremental R²

Realized volatility is autocorrelated at ~0.7–0.9 daily. A model containing
`lag_vol_1` reports a high R² that is *entirely the target predicting itself*.
So the baseline is not an empty model but **HAR** (Corsi, 2009):

```
log RV_{t+1}  ~  log RV_t  +  log RV_{t-5..t}  +  log RV_{t-22..t}
```

and the reported number is the **incremental out-of-sample R²** of
`HAR + attention` over `HAR` alone, never a raw R², with Clark–West for the
nested comparison. Walk-forward validation uses a purge and embargo between
folds, because the 5- and 22-day windows in both target and features straddle
the boundary and would otherwise leak.

### Screening, and why the panel is the defence against it

Screening 600 stocks × 30 features is 18,000 tests, of which ~900 look
significant at 5% *by chance alone* — in both directions, all of them
convincing. This is the fastest available route to a false result.

Pooling is the fix: the panel turns `N` per-stock tests into **one** coefficient
estimated on `N×T` observations, which is simultaneously more honest and more
powerful. Cross-sectional heterogeneity then becomes a second-stage question
with few tests — does the coefficient vary with market cap, analyst coverage,
retail ownership? — rather than a per-stock fishing expedition.

Negative relationships are *constructed*, not dredged. The vehicle is
**attention share**, `share_i,t = views_i,t / Σ_j views_j,t`: when Gucci is in
crisis, Hermès' share falls with its own absolute views unchanged. A negative
coefficient found that way has a mechanism behind it; one found by screening
does not.

Where screening is run as explicit discovery, the protocol is Benjamini–Hochberg
FDR with q-values reported, a stationary block bootstrap for the null (classical
t-tests badly overstate significance on autocorrelated series), Hansen's SPA
test for "is the best of N better than chance", and a final **2-year embargoed
holdout evaluated exactly once**.

### The universe comes from Wikidata, not from a hand-written list

This is the defence against the most damaging criticism an attention study can
attract: *you picked the terms that worked*. Membership is defined by Wikidata
properties — `P414` listing, `P249` ticker, `P946` ISIN, `P452` industry —
decided in advance and reproducible by anyone. The shipped YAML is a seed for
the pilot; `wikidata.discover_listed_companies()` is the authoritative version,
and where they disagree the query wins.

Delisted companies are **kept**, with their end date. Dropping a name because
its series ends early is textbook survivorship bias: the survivors are, by
construction, the firms whose attention shocks did not kill them. The panel is
unbalanced on purpose.

---

## Data sources

Everything used here is free and requires no API key.

| Layer | Source | Granularity | History | Notes |
|---|---|---|---|---|
| Attention | Wikimedia Pageviews API | daily, per article, per language | 2015-07 → | ~24–48h latency |
| Article identity | MediaWiki Action API | live | — | redirects, page moves, QIDs |
| Universe | Wikidata SPARQL | live | — | listings, tickers, sitelinks |
| Market | `yfinance` | daily OHLCV | decades | primary, no key, no quota |
| Company | `yfinance` | earnings dates, splits, share counts, holders | varies | same source, no key |
| Market | Twelve Data / Alpha Vantage | daily OHLCV | decades | optional cross-check, free API key |
| Macro | FRED `fredgraph.csv` | daily / monthly | decades | controls, **no API key** |

**Why the price data is cross-checked at all.** The failure mode that threatens
the result is not downtime — it is a silently wrong bar. An unadjusted split, a
stale close or a zero-volume placeholder raises nothing; it just changes the
answer. Two independent sources disagreeing is the only way to see one.

Stooq filled that role until it put a JavaScript anti-bot wall in front of every
request. The free API tiers that could replace it all publish coverage claims
that are hard to pin down *for the tier that costs nothing*, and this universe is
two-thirds non-US — Euronext Paris, Borsa Italiana, LSE, XETRA, SIX, BME, Nasdaq
Stockholm and Copenhagen, Tokyo, Hong Kong. "Free" and "covers Borsa Italiana on
the free plan" are different claims, and only the second one matters here.

So each candidate is an adapter behind one interface, and the question is
settled empirically rather than from a vendor's marketing page:

```bash
attention-panel check-sources config/universe_luxury.yaml
```

It probes one representative ticker per exchange — coverage gaps are per-venue,
not per-company — and prints which source actually returned data for which
venue. Set `TWELVEDATA_API_KEY` or `ALPHAVANTAGE_API_KEY` to include a source;
with neither set, everything still runs on the primary source alone.

**Source-independent structural checks do the real work either way**, and the
first live run proved it: with no second source they found four to five bad bars
per Hong Kong ticker plus single bad bars in four European names. Four checks —
internal consistency (is the high really the day's maximum?), zero range
(halts), stale bars (all four prices repeating the previous day), extreme moves
(probable unadjusted splits) — and each one **excludes** the bar rather than
merely counting it. A bar whose reported high sits below its close still yields
a finite `ln(H/L)`, so counting it and using it anyway would put a plausible,
fabricated variance in the target column on a day that looks entirely ordinary.

**Redirects are summed into the canonical article.** Each redirect title carries
its own independent pageview counter, and readers arrive via whichever alias a
headline used — so the redirect share of traffic is *highest on exactly the
spike days that carry the signal*. Ignoring redirects does not add symmetric
noise; it attenuates the events the study is about.

---

## Layout

```
DESIGN.md                        the research protocol, written before the code
config/universe_luxury.yaml      Tier-1 pilot universe (49 names, unverified until validated)
src/attention_panel/
  config.py                      universe loading; a universe is data, never a code branch
  httpcache.py                   polite HTTP: descriptive UA, backoff, settled-vs-volatile cache
  wikidata.py                    objective universe construction, cross-language sitelinks
  mediawiki.py                   canonical titles, redirect graph, page moves, QIDs
  pageviews.py                   the attention series, redirects summed, zero days filled
  sources.py                     every price vendor behind one interface, probed not assumed
  market.py                      bar validation; Garman-Klass, Parkinson, turnover
  macro.py                       FRED controls, aligned with an explicit publication lag
  features.py                    attention transforms; the per-venue availability lag
  fundamentals.py                earnings dates, corporate actions, share counts
  cli.py                         plan / validate-universe / fetch-attention / fetch-market
tests/                           106 tests, none of which touch the network
```

Not yet written: the panel assembly, the HAR baseline, and the significance
machinery. They are specified in DESIGN.md sections 3–6.

---

## Running it

### Setup

```bash
git clone https://github.com/alessiomartini/wikipedia-attention-etf-volatility.git
cd wikipedia-attention-etf-volatility
git checkout claude/attention-stock-correlation-b8nrjv

python -m venv .venv
# bash / zsh:   source .venv/bin/activate
# PowerShell:   .venv\Scripts\Activate.ps1
# Windows CMD:  .venv\Scripts\activate.bat

pip install -e ".[dev]"
```

`pip install -e .` is the step that makes everything below work: it puts the
package on the import path, so `python -m attention_panel.cli` runs from any
directory, and installs an `attention-panel` command as well. Without it,
Python cannot find the package, because the source lives under `src/`.

The modelling libraries are a separate extra, `pip install -e ".[model]"`. They
are not needed yet, and on a freshly released Python they are usually the last
packages to ship wheels — keeping them out of the default install means one
missing wheel cannot block a stage that does not use them.

### The User-Agent

Wikimedia requires a descriptive User-Agent with a contact address. Anonymous
clients are throttled, and the symptom is not an error — it is a run that
quietly returns fewer series than it asked for. Set it once per shell:

```bash
# bash / zsh (Linux, macOS, Git Bash, WSL)
export ATTENTION_PANEL_UA="attention-panel/0.1 (https://github.com/<you>/<repo>; <you>@example.com)"
```

```powershell
# PowerShell
$env:ATTENTION_PANEL_UA = "attention-panel/0.1 (https://github.com/<you>/<repo>; <you>@example.com)"
```

```bat
:: Windows CMD -- no quotes, and no spaces around the '='
set ATTENTION_PANEL_UA=attention-panel/0.1 (https://github.com/<you>/<repo>; <you>@example.com)
```

Or skip the variable and pass `--user-agent "..."` on each command.

### The commands

```bash
# 1. What would a full fetch cost? Offline, issues no requests.
attention-panel plan config/universe_luxury.yaml

# 2. RUN THIS FIRST. The shipped universe was written without network access,
#    so its tickers and article titles are unverified by construction.
attention-panel validate-universe config/universe_luxury.yaml

# 2b. Optional: which price sources actually cover this universe's exchanges?
attention-panel check-sources config/universe_luxury.yaml

# 2c. FRED control series and yfinance company data. No API key needed.
attention-panel fetch-macro
attention-panel fetch-fundamentals config/universe_luxury.yaml

# 3. Fetch. Settled history is cached permanently, so only the first run pays.
attention-panel fetch-attention config/universe_luxury.yaml
attention-panel fetch-market    config/universe_luxury.yaml

pytest
```

`python -m attention_panel.cli <command>` is equivalent to `attention-panel
<command>` everywhere. Paths are relative to the current directory, so run
these from the repository root.

The pilot is 155 distinct articles × 8 languages = 1,240 series, and redirects
multiply the request count by roughly 5–15×. That is a large but courteous
first run; every subsequent run is served almost entirely from cache, because
Wikimedia computes a settled day's pageviews once and never revises them.

---

## Honest limitations

- **Nothing has been run against the live APIs from this repository yet.** The
  code was written in an environment with no access to `wikimedia.org`,
  `query.wikidata.org` or `stooq.com`. The tests are thorough and run against
  recorded payloads, but the first live run will find things.
- **The shipped universe is unverified.** Tickers and article titles are
  plausible, not checked. `validate-universe` exists for exactly this.
- **The redirect graph is the current one.** A title that redirected here in
  2019 but was retargeted since is missed; one created last month is credited
  for its whole history. Small relative to ignoring redirects, but real.
- **The bot filter is a heuristic.** `agent=user` is Wikimedia's classification,
  not ground truth. Its divergence from `all-agents` is recorded as a quality
  flag and never used as a feature.
- **Tier 2 membership is point-in-time-unaware**, so a Tier 2 result carries
  survivorship bias until that is fixed. Tier 1 keeps its delisted names.
- **Wikipedia attention is a proxy for a proxy.** It captures encyclopedic
  curiosity, which correlates imperfectly with investor attention.
- **No causal claim is available.** Attention and volatility plausibly share a
  common driver — news — and lagged ordering identifies association, not cause.
- **No strategy, no costs, no P&L.** Forecasting volatility is not the same as
  monetizing it.
