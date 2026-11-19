"""Daily RV aggregation on synthetic intraday bars with a known answer."""
import numpy as np
import pandas as pd
from src.intraday import daily_rv


def test_daily_rv_known_value():
    idx = pd.date_range("2024-03-04 09:30", periods=26, freq="15min", tz="America/New_York")
    step = 0.001  # every 15-min log return is exactly +0.1%
    close = pd.Series(100 * np.exp(step * np.arange(26)), idx)
    rv = daily_rv(close)
    assert len(rv) == 1
    assert abs(rv.iloc[0] - 25 * step**2) < 1e-12  # 25 returns, first bar has none (no overnight)
