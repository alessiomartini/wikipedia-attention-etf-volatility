"""Smoke check: features at day t must not change if data after t-1 is altered."""
import numpy as np
import pandas as pd
from src import model_pipeline as mp


def _fake(seed, tail_shift=0.0):
    idx = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(seed)
    px = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))), idx)
    views = pd.Series(rng.integers(1000, 2000, 300), idx).astype(float)
    px.iloc[-1] *= np.exp(tail_shift)  # perturb only the last day
    views.iloc[-1] *= 1 + tail_shift * 10
    return px, views


def test_features_ignore_future(monkeypatch):
    out = []
    for shift in (0.0, 0.5):
        px, v = _fake(0, shift)
        monkeypatch.setattr(mp, "fetch_prices", lambda *a, px=px: px)
        monkeypatch.setattr(mp, "fetch_views", lambda *a, v=v: v)
        X, y = mp.build("X", "Y", "2020-01-01", "2021-01-01")
        out.append(X)
    # the perturbed last day may only change the last row of X (t-1 features of day t+1 do not exist)
    pd.testing.assert_frame_equal(out[0].iloc[:-1], out[1].iloc[:-1])
