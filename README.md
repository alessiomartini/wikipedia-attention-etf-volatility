# Alternative Data Alpha — Does Public Attention Predict ETF Volatility?

A small quantitative research prototype that asks one question and tries to
answer it without fooling itself:

> **Does a spike in public attention to a company predict the realized
> volatility of a related ETF the next day?**

Attention is proxied by **daily Wikipedia pageviews**; volatility is measured on
**daily ETF price data**. The point of the project is less the answer than the
discipline used to reach it: a non-stationary target is transformed into a
stationary one, features are lagged so no information from the future can leak
in, and validation is walk-forward rather than shuffled.

> **Status: single-file prototype.** Everything runs from one class in
> `src/model_pipeline.py`, roughly 100 lines. There is no saved output, no test
> suite and no CLI. Read the *Honest limitations* section before drawing any
> conclusion from a result it prints.

## The method

### 1. Data

| Layer | Source | Retrieved by |
| --- | --- | --- |
| Market | Daily OHLCV for one ETF ticker | `yfinance` |
| Alternative | Daily pageviews for one English Wikipedia article | Wikimedia REST API, `/metrics/pageviews/per-article/` |

The two series are joined on the date index with an inner join, so only days
present in both survive.

### 2. Making the series stationary

Prices are integrated of order one, $I(1)$; regressing one non-stationary series
on another produces spurious relationships and unstable coefficients. So nothing
enters the model in levels:

- prices become **daily log returns**, $r_t = \ln(P_t / P_{t-1})$;
- the **target** is realized volatility — the rolling standard deviation of
  those log returns over a 5-day window;
- pageviews enter as **percentage first differences**, which isolates the change
  in interest from the article's secular popularity trend.

### 3. Features, lagged on purpose

The design matrix has four columns, and every one of them is lagged:

```
lag_views_1, lag_views_2    attention change at t-1 and t-2
lag_vol_1,   lag_vol_2      realized volatility at t-1 and t-2
```

Using today's attention to predict today's volatility would be a look-ahead bug
dressed up as a result. Only $t-1$ and $t-2$ are available to the model.

### 4. Walk-forward validation

Standard $k$-fold cross-validation shuffles time and lets the model train on the
future to predict the past. This uses scikit-learn's `TimeSeriesSplit` with five
folds instead, so training always precedes testing:

```
Fold 1:  train [t0 … t1]  →  test (t1 … t2]
Fold 2:  train [t0 … t2]  →  test (t2 … t3]
Fold 3:  train [t0 … t3]  →  test (t3 … t4]
…
```

Scaling sits **inside** an `sklearn.pipeline.Pipeline` together with the model,
so the `StandardScaler` is fitted on each fold's training slice only. Scaling
before splitting would leak the test set's mean and variance into training — the
most common way a walk-forward backtest is quietly invalidated.

The estimator is **Ridge** regression ($\alpha = 1.0$). Out-of-sample **RMSE**
and **R²** are printed per fold.

## Repository layout

```
src/model_pipeline.py   The whole project: the AlternativeDataAlphaEngine class
src/__init__.py         Empty
main.py                 Unused — still the PyCharm "print_hi" starter stub
requirements.txt        Pinned dependency versions (see the note below)
```

## Running it

```bash
pip install yfinance pandas numpy scikit-learn requests
python -m src.model_pipeline
```

The `__main__` block at the bottom of `model_pipeline.py` runs the worked
example: predicting **QQQ** volatility from interest in the **Nvidia** Wikipedia
article, over 2025-01-01 → 2026-06-01. Change the four constructor arguments to
test another pair:

```python
engine = AlternativeDataAlphaEngine(
    ticker="QQQ",          # any Yahoo Finance ticker
    wiki_page="Nvidia",    # any en.wikipedia article title
    start_date="2025-01-01",
    end_date="2026-06-01",
)
X, y = engine.construct_design_matrix()
engine.run_backtest_pipeline(X, y)
```

**`requirements.txt` is encoded as UTF-16** (it was redirected from a PowerShell
`pip freeze`), which `pip install -r` cannot read. Either install the five
packages by name as above, or convert the file first:

```bash
iconv -f UTF-16 -t UTF-8 requirements.txt > requirements.utf8.txt
```

## Honest limitations

Worth stating plainly, because the framing above is the kind that invites
over-reading a result:

- **One ticker and one article per run.** There is no cross-sectional study and
  no portfolio; the engine handles a single pair at a time.
- **No stationarity test is actually run.** Log returns and first differences are
  applied because theory says they should be; no Augmented Dickey–Fuller test
  verifies it on the data at hand.
- **No baseline comparison.** Realized volatility is strongly autocorrelated, so
  `lag_vol_1` alone will already predict it well. Without a lags-only baseline,
  the R² reported here says nothing about whether the *attention* features add
  anything. This is the single most important missing piece.
- **No hyperparameter search and no significance test.** `alpha=1.0` is a
  default, not a choice, and a positive fold R² is not evidence of an
  exploitable effect.
- **Wikipedia pageviews are noisy as an attention proxy** — bot traffic, news
  cycles unrelated to the asset, and article renames all show up in the series.
- **Data sources are unstable.** The Wikimedia API expects a descriptive
  `User-Agent` (one is set) and rate-limits; `yfinance` breaks periodically, and
  recent versions auto-adjust prices by default, which changes the meaning of the
  `Adj Close` column this code reads.
- **No transaction costs, no trading rule, no strategy.** Predicting volatility
  is not the same as making money from it.
