import numpy as np
import pandas as pd
import pytest

from src.model_pipeline import AlternativeDataAlphaEngine


def make_engine(prices, views):
    return AlternativeDataAlphaEngine("TEST", "Test", "2020-01-01", "2021-01-01",
                                      prices=prices, views=views)


def synthetic(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    prices = pd.DataFrame({'Adj Close': 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))}, index=idx)
    views = pd.DataFrame({'views': rng.integers(1000, 2000, n)}, index=idx)
    return prices, views


def test_stationarity_transforms():
    prices, views = synthetic()
    eng = make_engine(prices, views)
    m = eng.fetch_market_data()
    r = np.log(prices['Adj Close']).diff().dropna()
    assert np.allclose(m['log_return'], r.loc[m.index])
    assert np.allclose(m['realized_volatility'], r.rolling(5).std().loc[m.index])
    v = eng.fetch_wikipedia_views()
    assert np.allclose(v['views_diff'], views['views'].pct_change().dropna())


def test_all_features_are_lagged():
    prices, views = synthetic()
    eng = make_engine(prices, views)
    X, y = eng.construct_design_matrix()
    market, alt = eng.fetch_market_data(), eng.fetch_wikipedia_views()
    # lag_views_k at date t equals views_diff at the k-th previous row, never same-day
    merged = market.join(alt, how='inner')
    for lag in (1, 2):
        expected = merged['views_diff'].shift(lag).loc[X.index]
        assert np.allclose(X[f'lag_views_{lag}'], expected)
        expected = merged['realized_volatility'].shift(lag).loc[X.index]
        assert np.allclose(X[f'lag_vol_{lag}'], expected)
    assert not np.allclose(X['lag_vol_1'], y)


def test_same_day_information_is_not_leaked():
    """Target is a perfect function of same-day views_diff; lagged features must not recover it."""
    rng = np.random.default_rng(1)
    n = 800
    idx = pd.bdate_range("2018-01-01", periods=n)
    # iid views -> views_diff has no useful lag structure that predicts a same-day function of itself
    views = pd.DataFrame({'views': np.exp(rng.normal(7, 0.3, n))}, index=idx)
    vd = views['views'].pct_change()
    prices = pd.DataFrame({'Adj Close': 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))}, index=idx)
    eng = make_engine(prices, views)
    X, _ = eng.construct_design_matrix()
    y = (vd.loc[X.index] ** 2) * 100  # perfectly predictable from same-day info only
    res = eng.run_backtest_pipeline(X, y)
    assert (res['r2_model'] < 0.3).all()
    assert res['r2_model'].mean() < 0.1


def test_lagged_signal_is_picked_up():
    """Sanity check the other way: a target that depends on yesterday's views_diff is predictable."""
    rng = np.random.default_rng(2)
    n = 800
    idx = pd.bdate_range("2018-01-01", periods=n)
    views = pd.DataFrame({'views': np.exp(rng.normal(7, 0.3, n))}, index=idx)
    prices = pd.DataFrame({'Adj Close': 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))}, index=idx)
    eng = make_engine(prices, views)
    X, _ = eng.construct_design_matrix()
    y = X['lag_views_1'] * 0.01 + 0.02
    res = eng.run_backtest_pipeline(X, y)
    assert (res['r2_model'] > 0.9).all()
    assert (res['r2_vol_only'] < res['r2_model']).all()


def test_fold_ordering_and_baseline_columns():
    prices, views = synthetic()
    eng = make_engine(prices, views)
    X, y = eng.construct_design_matrix()
    res = eng.run_backtest_pipeline(X, y)
    assert len(res) == 5
    assert (res['train_end'] < res['test_start']).all()
    assert res['test_start'].is_monotonic_increasing
    assert res['n_train'].is_monotonic_increasing
    for name in ('model', 'persistence', 'vol_only'):
        assert f'rmse_{name}' in res.columns and f'r2_{name}' in res.columns
        assert res[f'rmse_{name}'].notna().all()


def test_persistence_baseline_is_lag_vol_1():
    prices, views = synthetic()
    eng = make_engine(prices, views)
    X, y = eng.construct_design_matrix()
    res = eng.run_backtest_pipeline(X, y)
    first = res.iloc[0]
    test = slice(int(first['n_train']), int(first['n_train'] + first['n_test']))
    expected = np.sqrt(np.mean((y.iloc[test] - X['lag_vol_1'].iloc[test]) ** 2))
    assert first['rmse_persistence'] == pytest.approx(expected)


# ---- statistical tests (synthetic, fixed seeds) ----

from src.stats_tests import adf_test, diebold_mariano


def test_adf_rejects_white_noise_and_ar():
    rng = np.random.default_rng(10)
    noise = rng.normal(size=1500)
    assert adf_test(noise)['reject_5pct']
    ar = np.zeros(1500)
    e = rng.normal(size=1500)
    for t in range(1, 1500):
        ar[t] = 0.5 * ar[t - 1] + e[t]
    assert adf_test(ar)['reject_5pct']


def test_adf_does_not_reject_random_walk():
    rng = np.random.default_rng(20)
    rw = np.cumsum(rng.normal(size=1500))
    res = adf_test(rw)
    assert not res['reject_5pct']
    assert res['stat'] > res['crit_5']


def test_dm_identical_forecasts_is_nan_not_error():
    rng = np.random.default_rng(12)
    y = rng.normal(size=200)
    p = y + rng.normal(size=200)
    res = diebold_mariano(y, p, p.copy())
    assert np.isnan(res['stat']) and np.isnan(res['p_value'])


def test_dm_sign_convention_better_first_forecast():
    rng = np.random.default_rng(13)
    y = rng.normal(size=500)
    good = y + rng.normal(0, 0.2, 500)
    bad = y + rng.normal(0, 1.0, 500)
    res = diebold_mariano(y, good, bad)
    assert res['stat'] < 0 and res['p_value'] < 0.01
    flipped = diebold_mariano(y, bad, good)
    assert flipped['stat'] > 0
    assert flipped['stat'] == pytest.approx(-res['stat'])


def test_dm_equal_accuracy_not_significant_and_hac_runs():
    rng = np.random.default_rng(14)
    y = rng.normal(size=600)
    a = y + rng.normal(size=600)
    b = y + rng.normal(size=600)
    assert diebold_mariano(y, a, b)['p_value'] > 0.01
    assert np.isfinite(diebold_mariano(y, a, b, h=3)['stat'])


def test_pipeline_reports_adf_and_dm():
    prices, views = synthetic()
    eng = make_engine(prices, views)
    X, y = eng.construct_design_matrix()
    adf = eng.stationarity_tests()
    assert list(adf['series']) == ['log_return', 'realized_volatility', 'views_diff']
    eng.run_backtest_pipeline(X, y)
    dm = eng.significance_tests()
    assert list(dm['comparison']) == ['model vs vol_only', 'model vs persistence']
    assert (dm['n'] == len(eng.predictions_)).all()
    # predictions are the concatenated test folds: strictly later than the first training slice
    assert eng.predictions_.index.is_monotonic_increasing
