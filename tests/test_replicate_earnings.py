"""Synthetic-data tests for the pre-registered replication script (no network, no real data)."""
import numpy as np
import pandas as pd
import pytest
from src import intraday as it
from src import replicate_earnings as rep
from src import stocks


def synth(seed, n=1500, effect=0.0):
    """HAR-like log-RV; attention changes matter on earnings days by `effect` (0 = null)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-01-04", periods=n)
    earn = np.zeros(n, bool); earn[::63] = True
    views = rng.normal(size=n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.8 * y[t - 1] + rng.normal(scale=0.3) + 1.0 * earn[t] + effect * earn[t] * views[t - 1]
    y = pd.Series(y, index=idx)
    X = pd.DataFrame({"rv_d_l1": y.shift(1), "rv_w_l1": y.rolling(5).mean().shift(1),
                      "rv_m_l1": y.rolling(22).mean().shift(1),
                      "views_l1": pd.Series(views, idx).shift(1), "views_l2": pd.Series(views, idx).shift(2)})
    for k in range(1, 5):
        X[f"dow_{k}"] = (idx.dayofweek == k).astype(float)
    d = pd.concat([y.rename("y"), X], axis=1).dropna()
    return d[it.BASE_I + it.ATTN_I + it.DOW], d["y"], pd.Series(earn, idx)


def run(effect, monkeypatch, n_stocks=6, **kw):
    pairs, D, E = [(f"S{i}", f"P{i}") for i in range(n_stocks)], {}, {}
    for i, (tk, _) in enumerate(pairs):
        X, y, e = synth(i, effect=effect)
        D[tk], E[tk] = (X, y), e
    monkeypatch.setattr(stocks, "earnings_days", lambda tk, index: E[tk].reindex(index, fill_value=False))
    return rep.analyse(D, pairs, **kw)


def test_planted_effect_is_detected(monkeypatch):
    df, prim = run(1.0, monkeypatch)
    assert prim["success"] and prim["p"] < 0.01
    assert prim["n_stocks"] == 6 and prim["n_events"] > 60 and prim["n_weeks"] <= prim["n_events"]
    assert (df["earn1_r2_gain"] > 0).all()


def test_null_has_no_effect_and_events_window_filters(monkeypatch):
    _, prim = run(0.0, monkeypatch)
    assert prim["p"] > 0.01  # no planted signal
    _, late = run(0.0, monkeypatch, events_from="2020-01-01")
    assert 0 < late["n_events"] < prim["n_events"]


def test_sample_selection_and_run_lock(monkeypatch):
    s = rep.select_new_sample(excluded=["ORCL", "V"])
    assert len(s) == rep.N_SAMPLE and s[0][0] == "LLY" and "ORCL" not in dict(s)
    originals = {t for t, _ in stocks.STOCKS}
    assert not originals & {t for t, _ in rep.CANDIDATES}
    import datetime as dt
    monkeypatch.setitem(rep.SAMPLES, "new", (*rep.SAMPLES["new"][:5], dt.date(2999, 1, 1)))
    with pytest.raises(SystemExit):
        rep.main("new")  # before the registered run date: refuses to run
