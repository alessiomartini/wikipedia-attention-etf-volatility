# Alternative Data Alpha — Does Public Attention Predict ETF Volatility?

A small quantitative research prototype that asks one question and tries to
answer it without fooling itself:

> **Does a spike in public attention to a company predict the realized
> volatility of a related ETF the next day?**

Attention is proxied by **daily Wikipedia pageviews**; volatility is measured on
**daily ETF price data**. The point of the project is less the answer than the
discipline used to reach it: a non-stationary target is transformed into a
stationary one (WHAT DOES IT MEAN?), features are lagged so no information from the future can leak
in, and validation is walk-forward rather than shuffled.

> **Status: small prototype, now with tests, a CLI and a baseline.** Everything
> runs from one class in `src/model_pipeline.py`. `main.py` is a small CLI, the
> backtest also reports a persistence baseline and a volatility-lags-only Ridge,
> and `tests/` holds offline pytest tests on synthetic data. No real-data result
> is saved in the repository. Read the *Honest limitations* section before
> drawing any conclusion from a result it prints.

## The method

### 1. Data

| Layer | Source | Retrieved by |
| --- | --- | --- |
| Market | Daily OHLCV (Open, High, Low, Close, and Volume) for one ETF ticker | `yfinance` |
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
dressed up as a result. Only $t-1$ and $t-2$ are available to the model. In principle, what we want to do is to use today's data to predict tomorrow's volatility (implied volatility) and compare it with realized volatility of tomorrow using tomorrow's data.

### 4. Walk-forward validation

Standard $k$-fold cross-validation shuffles time and lets the model train on the
future to predict the past. (THIS IS STANDARD FOR TIME SERIES AND FORECASTING). This uses scikit-learn's `TimeSeriesSplit` with five
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

The estimator is **Ridge** regression ($\alpha = 1.0$). (We didn't use **Lasso** just because this is an experiment. It would be an improvement to implement the combination of the two of them). Out-of-sample **RMSE**
and **R²** are printed per fold, next to two baselines evaluated on the same
folds: a **persistence** forecast ($\hat\sigma_t = \sigma_{t-1}$, nothing fitted)
and a **vol-only** Ridge that sees only the volatility lags. The attention
features are worth something only if the full model beats both.

### 5. Stationarity and significance tests

Two tests are implemented by hand in `src/stats_tests.py` (numpy/scipy only, no
`statsmodels`):

- **Augmented Dickey–Fuller** (constant, lag order chosen by AIC) on the series
  that enter the model: log returns, realized volatility and the pageview
  percentage change. The statistic is compared with MacKinnon's asymptotic
  critical values (1%: −3.43, 5%: −2.86, 10%: −2.57); p-values are not computed.
- **Diebold–Mariano** on the concatenated walk-forward test predictions, with
  squared-error loss, the Harvey–Leybourne–Newbold small-sample correction and a
  Newey–West variance with $h-1$ lags ($h=1$). It compares the full model with the
  vol-only Ridge and with persistence. A **negative** statistic means the full
  model has the smaller squared error.

## Repository layout

```
src/model_pipeline.py   The AlternativeDataAlphaEngine class (data, features, backtest, baselines)
src/stats_tests.py      ADF and Diebold-Mariano tests (numpy/scipy)
src/__init__.py         Empty
main.py                 Command-line entry point (argparse)
tests/                  Offline pytest tests on synthetic data (no network)
requirements.txt        Pinned dependency versions (see the note below)
requirements-dev.txt    Test dependency (pytest)
```

## Running it

```bash
pip install yfinance pandas numpy scikit-learn requests
python main.py                       # QQQ vs the Nvidia article, 2025-01-01 -> 2026-06-01
python main.py --ticker SPY --article Apple_Inc. --start 2024-01-01 --end 2025-01-01 --out folds.csv
```

The defaults reproduce the worked example: predicting **QQQ** volatility from
interest in the **Nvidia** Wikipedia article. The CLI also prints the ADF and
Diebold–Mariano tables. `--out` writes the per-fold metrics
(model, persistence and vol-only RMSE and R²) to a CSV, and the Diebold–Mariano
table to a second CSV with a `_dm` suffix (`folds.csv` → `folds_dm.csv`). The engine can also be
used from Python, and accepts already-loaded data so that nothing is downloaded:

```python
engine = AlternativeDataAlphaEngine(
    ticker="QQQ", wiki_page="Nvidia",
    start_date="2025-01-01", end_date="2026-06-01",
    prices=None,   # optional DataFrame with an 'Adj Close' column
    views=None,    # optional DataFrame with a 'views' column
)
X, y = engine.construct_design_matrix()
results = engine.run_backtest_pipeline(X, y)   # DataFrame, one row per fold
```

### Tests

```bash
pip install pytest        # or: pip install -r requirements-dev.txt
python -m pytest
```

The tests use synthetic data only. They check the stationarity transforms, that
every feature is lagged (including a case where the target depends only on
same-day information and the model must not be able to predict it), the ordering
of the walk-forward folds, the baseline columns, and the ADF and Diebold–Mariano
tests (white noise and AR(1) reject a unit root, a random walk does not;
identical forecasts give NaN; the sign convention is checked).

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
- **The ADF test is only a check.** It is run on each series, but it has low
  power against persistent alternatives (realized volatility is close to a unit
  root), uses asymptotic critical values without p-values, and says nothing about
  structural breaks.
- **The Diebold–Mariano test is approximate here.** The full model and the
  vol-only Ridge are nested, for which the DM null distribution is only an
  approximation, and the 5-day volatility windows overlap, so forecast errors are
  autocorrelated even at $h=1$; the $h-1$ lag Newey–West variance does not
  correct for that. Read the p-values with caution.
- **No hyperparameter search.** `alpha=1.0` is a default, not a choice, and a
  positive fold R² is not evidence of an exploitable effect.
- **Wikipedia pageviews are noisy as an attention proxy** — bot traffic, news
  cycles unrelated to the asset, and article renames all show up in the series.
- **Data sources are unstable.** The Wikimedia API expects a descriptive
  `User-Agent` (one is set) and rate-limits; `yfinance` breaks periodically, and
  recent versions auto-adjust prices by default, which changes the meaning of the
  `Adj Close` column this code reads.
- **No transaction costs, no trading rule, no strategy.** Predicting volatility
  is not the same as making money from it.

## Notes on the questions in the text

Short answers to the questions written in capitals above.

**What does "stationary" mean?** A series is (weakly) stationary if its mean,
variance and autocovariances do not depend on time. Prices wander: their mean and
variance change over time (a random walk, $I(1)$), so a model fitted on one period
describes another period badly, and regressing one trending series on another
gives high R² and "significant" coefficients even when the two are unrelated
(spurious regression). Differencing the log price gives returns, which are
approximately stationary, and the same idea applies to pageviews via percentage
changes. "Approximately" matters: volatility clusters, so returns are not
independent, and stationarity should be checked (e.g. with an ADF test), not
assumed.

**Is walk-forward validation the standard for forecasting?** Yes. Shuffled
$k$-fold puts future observations in the training set and past ones in the test
set. With autocorrelated data (overlapping 5-day windows make neighbouring
volatility values share most of their inputs) a test point has near-duplicates in
the training set, so the score is inflated and says nothing about real
forecasting. Walk-forward (expanding window, as in `TimeSeriesSplit`) always
trains on the past and tests on the future, as in real use. With overlapping
targets one may also add a gap between train and test.

**Ridge, Lasso, Elastic Net.** All three are linear regression with a penalty on
the coefficients. Ridge penalizes the sum of squares ($L_2$): it shrinks
coefficients and handles correlated features well (here the two volatility lags
are strongly correlated), but never sets them exactly to zero. Lasso penalizes
the sum of absolute values ($L_1$): it can zero out coefficients, i.e. select
features, but picks arbitrarily among correlated ones. Elastic Net mixes the two
penalties, with a weight between them. With four features there is little to
select, so the choice matters less than the baseline comparison; the penalty
strength (and the mixing weight) should be chosen by cross-validation inside the
training window, not fixed by hand.
