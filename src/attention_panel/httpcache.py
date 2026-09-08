"""A polite, cached HTTP client shared by every ingestion module.

WHY THIS IS NOT JUST `requests.get`

Three properties are needed by all three upstreams (Wikimedia AQS, the
MediaWiki Action API, Wikidata SPARQL), and getting any of them wrong produces
either a ban or a silently truncated dataset:

1. IDENTIFICATION. Wikimedia's API etiquette requires a descriptive
   User-Agent with a contact address. Anonymous or library-default agents are
   rate-limited aggressively and may be blocked outright. This is a hard
   requirement, not a nicety, so the client refuses to start without one.

2. BACKOFF. A 429 or a 503 that is treated as a permanent failure turns into a
   gap in a time series. A gap in a time series is not a visible error -- it is
   a slightly different result. Every retryable status is retried with
   exponential backoff and jitter, honouring `Retry-After` when present.

3. CACHING, with the right invalidation rule. This is the important one.
   Pageview counts for a day in the past are IMMUTABLE: Wikimedia computes them
   once and never revises them. Counts for the last day or two can still be
   revised as late-arriving logs are processed. So the cache is permanent for
   settled data and short-lived for recent data, which is what makes it
   acceptable to re-run the full pilot (1240 article-language series) as often
   as development requires while issuing almost no repeat traffic.

The cache is on disk and content-addressed, so it survives process restarts and
is safe to share between the CLI and the tests.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Mapping

import requests

log = logging.getLogger(__name__)

# Statuses worth retrying. 404 is deliberately absent: for the pageviews API a
# 404 is a meaningful answer ("this article has no recorded traffic"), not a
# transient failure, and retrying it would waste the rate-limit budget.
RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})

DEFAULT_CACHE_DIR = Path(".httpcache")


class RateLimitError(RuntimeError):
    """Raised when an upstream kept refusing after every retry was exhausted."""


@dataclass(frozen=True)
class CachePolicy:
    """How long a response may be reused.

    `ttl_seconds=None` means "forever": use it only for data the upstream
    guarantees never changes. Everything else must carry a finite TTL, so that
    a stale value cannot quietly become a research finding.
    """

    ttl_seconds: float | None

    # Declared as ClassVar so the dataclass does not turn these into fields.
    IMMUTABLE: ClassVar["CachePolicy"]
    VOLATILE: ClassVar["CachePolicy"]
    NONE: ClassVar["CachePolicy"]


#: Settled pageview days: Wikimedia computes them once and never revises them.
CachePolicy.IMMUTABLE = CachePolicy(ttl_seconds=None)
#: Recent days, and anything editable (article titles, redirects, Wikidata).
CachePolicy.VOLATILE = CachePolicy(ttl_seconds=6 * 3600)
#: Bypass the cache entirely.
CachePolicy.NONE = CachePolicy(ttl_seconds=0.0)


class _TokenBucket:
    """Minimum-interval limiter, shared across threads.

    Wikimedia publishes generous limits, but the courteous ceiling for an
    unauthenticated research client is far below them. Pacing requests costs a
    few minutes on a full fetch and removes any risk of being throttled
    halfway through, which would leave a partially-filled panel.
    """

    def __init__(self, min_interval: float) -> None:
        self._min_interval = min_interval
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._next_allowed = now + self._min_interval


class HttpClient:
    """Cached, rate-limited, retrying HTTP client.

    Args:
        user_agent: descriptive agent WITH a contact address, e.g.
            ``"attention-panel/0.1 (https://github.com/<user>/<repo>; you@example.com)"``.
            Required by Wikimedia; the client refuses a generic one.
        cache_dir: on-disk cache location, or ``None`` to disable caching.
        min_interval: seconds between requests to the same client.
        max_retries: attempts per request before giving up.
    """

    def __init__(
        self,
        user_agent: str,
        cache_dir: str | Path | None = DEFAULT_CACHE_DIR,
        min_interval: float = 0.12,
        max_retries: int = 5,
        timeout: float = 60.0,
    ) -> None:
        _validate_user_agent(user_agent)
        self.user_agent = user_agent
        self.timeout = timeout
        self.max_retries = max_retries
        self._bucket = _TokenBucket(min_interval)
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": user_agent,
                # Wikimedia serves gzip; the series are highly compressible and
                # this cuts a full pilot fetch by roughly an order of magnitude.
                "Accept-Encoding": "gzip",
            }
        )
        self.stats = {"hits": 0, "misses": 0, "retries": 0}

    # -- public API ---------------------------------------------------------

    def get_json(
        self,
        url: str,
        params: Mapping[str, Any] | None = None,
        policy: CachePolicy = CachePolicy.VOLATILE,
        headers: Mapping[str, str] | None = None,
        allow_404: bool = False,
    ) -> Any | None:
        """GET a URL and parse JSON.

        Returns ``None`` on a 404 when ``allow_404`` is set. That case is not
        an error for the pageviews API -- it means the article genuinely has no
        recorded traffic in the requested window -- and conflating it with a
        failure would either crash a full run or, worse, encourage a bare
        `except` that also swallows real failures.
        """
        body = self.get_text(url, params=params, policy=policy, headers=headers, allow_404=allow_404)
        if body is None:
            return None
        return json.loads(body)

    def get_text(
        self,
        url: str,
        params: Mapping[str, Any] | None = None,
        policy: CachePolicy = CachePolicy.VOLATILE,
        headers: Mapping[str, str] | None = None,
        allow_404: bool = False,
    ) -> str | None:
        cache_key = _cache_key(url, params)
        cached = self._read_cache(cache_key, policy)
        if cached is not None:
            self.stats["hits"] += 1
            return None if cached["status"] == 404 else cached["body"]

        self.stats["misses"] += 1
        status, body = self._request_with_retries(url, params, headers, allow_404=allow_404)
        # A 404 is cached too, and under the *same* policy. Re-asking Wikimedia
        # 1240 times per run about articles that have no traffic is exactly the
        # behaviour the etiquette guidelines ask clients not to have.
        self._write_cache(cache_key, policy, status, body, url)
        return None if status == 404 else body

    # -- internals ----------------------------------------------------------

    def _request_with_retries(
        self,
        url: str,
        params: Mapping[str, Any] | None,
        headers: Mapping[str, str] | None,
        allow_404: bool,
    ) -> tuple[int, str]:
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            self._bucket.acquire()
            try:
                response = self._session.get(
                    url, params=params, headers=dict(headers or {}), timeout=self.timeout
                )
            except requests.RequestException as exc:
                # Connection resets and read timeouts are indistinguishable
                # from throttling at this layer, so they take the same path.
                last_error = exc
                self._sleep_backoff(attempt, None)
                self.stats["retries"] += 1
                continue

            if response.status_code == 404 and allow_404:
                return 404, ""

            if response.status_code in RETRYABLE_STATUSES:
                self.stats["retries"] += 1
                retry_after = _parse_retry_after(response.headers.get("Retry-After"))
                log.warning(
                    "%s from %s (attempt %d/%d)", response.status_code, url, attempt + 1, self.max_retries
                )
                self._sleep_backoff(attempt, retry_after)
                last_error = RateLimitError(f"HTTP {response.status_code} from {url}")
                continue

            response.raise_for_status()
            return response.status_code, response.text

        raise RateLimitError(f"giving up on {url} after {self.max_retries} attempts") from last_error

    def _sleep_backoff(self, attempt: int, retry_after: float | None) -> None:
        if retry_after is not None:
            time.sleep(min(retry_after, 120.0))
            return
        # Exponential with full jitter. Without jitter, a batch of parallel
        # fetchers that hit a 429 together would retry together forever.
        delay = min(2.0**attempt, 60.0)
        time.sleep(random.uniform(0.0, delay))

    def _cache_path(self, key: str) -> Path:
        assert self._cache_dir is not None
        # Two-level fan-out: a flat directory with tens of thousands of entries
        # is slow to stat on most filesystems.
        return self._cache_dir / key[:2] / f"{key}.json"

    def _read_cache(self, key: str, policy: CachePolicy) -> dict | None:
        if self._cache_dir is None or policy.ttl_seconds == 0.0:
            return None
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # A truncated cache file (interrupted write, full disk) must never
            # be able to fail a run; treat it as a miss and overwrite it.
            return None
        if policy.ttl_seconds is not None and time.time() - entry["fetched_at"] > policy.ttl_seconds:
            return None
        return entry

    def _write_cache(self, key: str, policy: CachePolicy, status: int, body: str, url: str) -> None:
        if self._cache_dir is None or policy.ttl_seconds == 0.0:
            return
        path = self._cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"url": url, "status": status, "body": body, "fetched_at": time.time()}
        # Write-then-rename, so an interrupted run cannot leave a half-written
        # entry that a later run would read as a valid response.
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(entry), encoding="utf-8")
        tmp.replace(path)


def _cache_key(url: str, params: Mapping[str, Any] | None) -> str:
    canonical = url + "?" + json.dumps(dict(sorted((params or {}).items())), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        # The HTTP-date form of Retry-After is rare here; falling back to
        # exponential backoff is safe and avoids a date-parsing dependency.
        return None


def _validate_user_agent(user_agent: str) -> None:
    """Reject agents that would get the project throttled or blocked.

    Wikimedia asks for a descriptive agent identifying the tool and a way to
    contact its operator. Failing fast here is far better than discovering
    mid-fetch that half the series are missing.
    """
    if not user_agent or len(user_agent) < 15:
        raise ValueError(
            "A descriptive User-Agent with contact details is required by Wikimedia, e.g. "
            "'attention-panel/0.1 (https://github.com/<user>/<repo>; you@example.com)'"
        )
    if "@" not in user_agent and "http" not in user_agent:
        raise ValueError(
            "User-Agent must contain a contact address or project URL so Wikimedia "
            "operators can reach you if the client misbehaves."
        )
    banned = ("python-requests", "curl/", "Mozilla/")
    if any(token in user_agent for token in banned):
        raise ValueError(f"User-Agent must not impersonate a generic client: {user_agent!r}")
