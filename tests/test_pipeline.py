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
