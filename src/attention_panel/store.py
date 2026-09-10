"""An on-disk dataset store, so nothing is downloaded twice.

TWO CACHES, DOING DIFFERENT JOBS

`httpcache` stores raw HTTP responses and stops the same request being made
again. This module stores assembled *datasets* -- a ticker's price history, a
language's pageview matrix -- and stops them being re-fetched and re-derived at
all. Both are needed, and one does not replace the other:

  * Some sources never touch the HTTP cache. yfinance does its own networking,
    so before this module existed every run re-downloaded every price and every
    earnings date, no matter how many times it had already seen them.
  * A stored dataset is inspectable. A directory of CSVs a person can open is a
    different kind of artefact from a directory of hashed response bodies.

INCREMENTAL BY DEFAULT

A stored series that already runs to last Tuesday only needs Wednesday onward,
so a second run costs a few days of data rather than ten years of it.

THE TRAP THAT MAKES NAIVE APPENDING DANGEROUS

Adjusted prices are not append-only. When a company splits, **every historical
price in the series is retroactively divided by the split factor**. A cache that
appends new rows to old ones would then hold pre-split prices before the event
and post-split prices after it -- a series with a fabricated jump at the join,
and one whose Garman-Klass estimates straddle two different adjustment
conventions. That is precisely the failure `market.py` warns about, arriving
through the back door of a cache rather than through a mixed-column read.

So `merge_incremental` compares the overlapping region instead of trusting it.
If stored and fresh disagree there, the series was re-adjusted and the answer is
a full re-fetch, never an append. Detecting it costs one overlapping day; not
detecting it costs the study.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_ROOT = Path("data/store")

#: Days at the end of a stored series that are always re-fetched.
#: Pageview counts for the last few days are still being revised, and a price
#: bar can be corrected after the close. Re-asking for a short tail is cheap;
#: trusting a provisional value forever is not.
REVISABLE_TAIL_DAYS = 5


@dataclass
class MergeOutcome:
    """The result of folding fresh data into stored data."""

    frame: pd.DataFrame
    #: "appended", "replaced" (a retroactive re-adjustment was detected),
    #: "unchanged", or "created".
    action: str
    detail: str = ""


class DataStore:
    """CSV-backed store of assembled datasets, with a manifest.

    CSV rather than parquet on purpose: it adds no binary dependency, it is
    readable by anything, and at this size the difference is irrelevant. A
    research dataset a collaborator cannot open without installing something is
    a dataset that gets re-derived instead of reused.
    """

    def __init__(self, root: str | Path = DEFAULT_ROOT) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.root / "manifest.json"

    # -- paths --------------------------------------------------------------

    def path_for(self, kind: str, key: str) -> Path:
        # Tickers contain characters that are illegal in filenames on Windows
        # ("HM-B.ST" is fine, but a slash would not be), so keys are sanitised
        # rather than trusted.
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in key)
        directory = self.root / kind
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{safe}.csv.gz"

    # -- read / write -------------------------------------------------------

    def load(self, kind: str, key: str) -> pd.DataFrame | None:
        path = self.path_for(kind, key)
        if not path.exists():
            return None
        try:
            frame = pd.read_csv(path, index_col=0, parse_dates=True, compression="gzip")
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            # A truncated file from an interrupted write must never fail a run;
            # treat it as absent and let it be rebuilt.
            log.warning("unreadable store entry %s (%s); refetching", path, exc)
            return None
        return frame.rename_axis("date").sort_index()

    def save(self, kind: str, key: str, frame: pd.DataFrame, **meta) -> Path:
        path = self.path_for(kind, key)
        # Write-then-rename: an interrupted save cannot leave a half-written
        # file that a later run would read as complete.
        tmp = path.with_suffix(".tmp")
        frame.to_csv(tmp, compression="gzip")
        tmp.replace(path)
        self._record(kind, key, frame, meta)
        return path

    def coverage(self, kind: str, key: str) -> tuple[dt.date, dt.date] | None:
        frame = self.load(kind, key)
        if frame is None or frame.empty:
            return None
        return frame.index.min().date(), frame.index.max().date()

    # -- manifest -----------------------------------------------------------

    def _record(self, kind: str, key: str, frame: pd.DataFrame, meta: dict) -> None:
        """Note what was stored and when.

        Reproducibility needs the fetch date, not just the data: a result that
        cannot say which vintage of a revised series produced it cannot be
        rerun.
        """
        manifest = self.manifest()
        manifest[f"{kind}/{key}"] = {
            "rows": int(len(frame)),
            "columns": list(map(str, frame.columns)),
            "first": frame.index.min().date().isoformat() if len(frame) else None,
            "last": frame.index.max().date().isoformat() if len(frame) else None,
            "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            **{k: str(v) for k, v in meta.items()},
        }
        tmp = self._manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self._manifest_path)

    def manifest(self) -> dict:
        if not self._manifest_path.exists():
            return {}
        try:
            return json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def summary(self) -> pd.DataFrame:
        """The manifest as a table, for `attention-panel store-status`."""
        manifest = self.manifest()
        if not manifest:
            return pd.DataFrame(columns=["dataset", "rows", "first", "last", "fetched_at"])
        rows = [{"dataset": name, **entry} for name, entry in sorted(manifest.items())]
        frame = pd.DataFrame(rows)
        return frame[[c for c in ("dataset", "rows", "first", "last", "fetched_at") if c in frame]]

    # -- incremental fetch --------------------------------------------------

    def missing_window(
        self,
        kind: str,
        key: str,
        start: dt.date,
        end: dt.date,
        revisable_tail_days: int = REVISABLE_TAIL_DAYS,
    ) -> tuple[dt.date, dt.date] | None:
        """The date range still to fetch, or None when the store already has it.

        The stored end is pulled back by `revisable_tail_days` because the last
        few observations of a series are provisional -- pageview counts are
        still being revised, and a price bar can be corrected after the close.
        Re-asking for a short tail every run is cheap; trusting a provisional
        value forever is not.

        A stored series that does not reach far enough back is refetched from
        `start`, not patched at the front: a gap in the middle of a time series
        is worse than a redundant download.
        """
        existing = self.coverage(kind, key)
        if existing is None:
            return (start, end)

        stored_start, stored_end = existing
        if stored_start > start:
            return (start, end)

        resume = stored_end - dt.timedelta(days=revisable_tail_days)
        if resume >= end:
            return None
        return (max(resume, start), end)


def merge_incremental(
    stored: pd.DataFrame | None,
    fresh: pd.DataFrame,
    price_column: str | None = "close",
    tolerance: float = 1e-4,
) -> MergeOutcome:
    """Fold a fresh fetch into stored data, refusing to append across a split.

    WHY THE OVERLAP IS CHECKED RATHER THAN TRUSTED

    Adjusted price series are not append-only. A split retroactively divides
    every historical price by the split factor, so appending fresh rows to
    stored ones would produce a series holding two different adjustment
    conventions either side of the join -- with a fabricated jump at the seam
    and Garman-Klass estimates that are meaningless across it.

    Comparing the overlapping dates catches it: if the same day has a different
    price in the two frames, the history was re-adjusted and the whole series
    must be replaced, not extended. The check costs one overlapping day.

    `price_column=None` disables the check, for datasets that genuinely are
    append-only, such as pageview counts, which Wikimedia never revises once a
    day has settled.
    """
    if stored is None or stored.empty:
        return MergeOutcome(fresh.sort_index(), "created", f"{len(fresh)} rows")
    if fresh.empty:
        return MergeOutcome(stored, "unchanged", "fetch returned nothing")

    overlap = stored.index.intersection(fresh.index)
    if price_column and price_column in stored.columns and price_column in fresh.columns and len(overlap):
        old = stored.loc[overlap, price_column].astype("float64")
        new = fresh.loc[overlap, price_column].astype("float64")
        both = old.notna() & new.notna()
        if both.any():
            relative = ((old[both] - new[both]).abs() / new[both].abs().replace(0, np.nan)).max()
            if pd.notna(relative) and relative > tolerance:
                return MergeOutcome(
                    fresh.sort_index(),
                    "replaced",
                    f"history re-adjusted (max relative change {relative:.1%} on "
                    f"{len(overlap)} overlapping days); appending would have mixed "
                    "two adjustment conventions",
                )

    # Fresh wins on overlapping dates: it is the later vintage of a revisable
    # observation.
    combined = pd.concat([stored[~stored.index.isin(fresh.index)], fresh])
    added = len(combined) - len(stored)
    return MergeOutcome(combined.sort_index(), "appended", f"{added} new rows")
