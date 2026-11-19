"""Earnings reaction-day mapping: after-close -> next trading day; before-open -> same day; weekend -> Monday."""
import pandas as pd
from src import model_pipeline as mp
from src.panel import earnings_days


def test_reaction_days(tmp_path, monkeypatch):
    monkeypatch.setattr(mp, "CACHE", tmp_path)
    pd.Series(["2024-03-07 16:00:00-05:00",   # Thu after close -> Fri 8th
               "2024-03-12 07:00:00-04:00",   # Tue before open -> Tue 12th
               "2024-03-15 16:00:00-04:00"]).to_csv(tmp_path / "earnings_XX.csv", index=False)  # Fri after close -> Mon 18th
    idx = pd.bdate_range("2024-03-04", "2024-03-22")
    e = earnings_days("XX", idx)
    assert list(e[e].index.strftime("%m-%d")) == ["03-08", "03-12", "03-18"]
