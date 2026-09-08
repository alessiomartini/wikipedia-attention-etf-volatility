"""Shared fixtures. No test in this suite is allowed to touch the network.

Ingestion code that can only be tested against a live API is untestable in
practice: the APIs rate-limit, they change, and a red test that means "Wikimedia
is slow today" trains everyone to ignore red tests. Every test here therefore
runs against recorded payloads through `FakeClient`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeClient:
    """Stands in for `HttpClient`, serving canned responses and recording calls.

    Matching is by substring so a test can key on the meaningful part of a URL
    (an article title, a SPARQL fragment) without restating query-string order.
    """

    def __init__(self, responses: dict[str, object] | None = None) -> None:
        self.responses = responses or {}
        self.calls: list[tuple[str, dict]] = []

    def _match(self, url: str, params: dict | None):
        haystack = url + "|" + "|".join(f"{k}={v}" for k, v in sorted((params or {}).items()))
        for needle, payload in self.responses.items():
            if needle in haystack:
                return payload
        return None

    def get_json(self, url, params=None, policy=None, headers=None, allow_404=False):
        self.calls.append((url, dict(params or {})))
        return self._match(url, params)

    def get_text(self, url, params=None, policy=None, headers=None, allow_404=False):
        self.calls.append((url, dict(params or {})))
        return self._match(url, params)


@pytest.fixture
def fake_client():
    return FakeClient


@pytest.fixture
def ohlcv():
    """A small, deliberately awkward price frame.

    It contains a normal bar, a zero-range (halted) bar, and a bar whose open
    and close sit at the extremes of the range -- the three cases the
    volatility estimators have to handle differently.
    """
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    return pd.DataFrame(
        {
            "open": [100.0, 105.0, 50.0, 60.0],
            "high": [110.0, 106.0, 50.0, 61.0],
            "low": [90.0, 104.0, 50.0, 59.0],
            "close": [105.0, 104.5, 50.0, 59.5],
            "volume": [1_000_000.0, 900_000.0, 0.0, 1_200_000.0],
        },
        index=pd.DatetimeIndex(index, name="date"),
    )
