"""Tests for the shared HTTP client.

The User-Agent checks are not pedantry: Wikimedia throttles or blocks anonymous
and generic agents, and the symptom is not an exception but a run that quietly
returns fewer series than it asked for.
"""

from __future__ import annotations

import time

import pytest

from attention_panel.httpcache import CachePolicy, HttpClient

GOOD_UA = "attention-panel/0.1 (https://github.com/example/repo; researcher@example.com)"


@pytest.mark.parametrize(
    "bad_agent",
    [
        "",
        "bot",
        "python-requests/2.31.0",
        "Mozilla/5.0 (Windows NT 10.0)",
        "attention-panel/0.1 no contact",
    ],
)
def test_agents_that_would_get_us_throttled_are_rejected_at_construction(bad_agent):
    with pytest.raises(ValueError):
        HttpClient(bad_agent, cache_dir=None)


def test_a_descriptive_agent_is_accepted(tmp_path):
    client = HttpClient(GOOD_UA, cache_dir=tmp_path)
    assert client._session.headers["User-Agent"] == GOOD_UA


def test_cache_round_trip(tmp_path):
    client = HttpClient(GOOD_UA, cache_dir=tmp_path)
    key = "abc123"
    client._write_cache(key, CachePolicy.IMMUTABLE, 200, '{"x": 1}', "http://example/x")
    entry = client._read_cache(key, CachePolicy.IMMUTABLE)
    assert entry is not None and entry["body"] == '{"x": 1}'


def test_an_expired_volatile_entry_is_a_miss(tmp_path):
    client = HttpClient(GOOD_UA, cache_dir=tmp_path)
    key = "expiring"
    client._write_cache(key, CachePolicy.VOLATILE, 200, "stale", "http://example/x")
    # Backdate the entry past a one-second TTL.
    path = client._cache_path(key)
    entry = path.read_text().replace(f'"fetched_at": {time.time():.0f}', '"fetched_at": 0')
    path.write_text(entry.replace('"fetched_at"', '"fetched_at"'))
    assert client._read_cache(key, CachePolicy(ttl_seconds=-1)) is None


def test_immutable_entries_never_expire(tmp_path):
    """Settled pageview days are computed once by Wikimedia and never revised."""
    client = HttpClient(GOOD_UA, cache_dir=tmp_path)
    client._write_cache("k", CachePolicy.IMMUTABLE, 200, "body", "http://example/x")
    assert client._read_cache("k", CachePolicy.IMMUTABLE) is not None


def test_a_corrupt_cache_file_is_treated_as_a_miss_not_a_crash(tmp_path):
    """An interrupted write must never be able to fail a later run."""
    client = HttpClient(GOOD_UA, cache_dir=tmp_path)
    path = client._cache_path("broken")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ truncated")
    assert client._read_cache("broken", CachePolicy.IMMUTABLE) is None


def test_caching_can_be_disabled_entirely(tmp_path):
    client = HttpClient(GOOD_UA, cache_dir=tmp_path)
    client._write_cache("k", CachePolicy.NONE, 200, "body", "http://example/x")
    assert client._read_cache("k", CachePolicy.NONE) is None


def test_cache_keys_separate_different_query_parameters():
    from attention_panel.httpcache import _cache_key

    a = _cache_key("http://x/api", {"titles": "Gucci"})
    b = _cache_key("http://x/api", {"titles": "Prada"})
    assert a != b
    # Parameter order must not change the key, or the cache would never hit.
    assert _cache_key("http://x", {"a": 1, "b": 2}) == _cache_key("http://x", {"b": 2, "a": 1})
