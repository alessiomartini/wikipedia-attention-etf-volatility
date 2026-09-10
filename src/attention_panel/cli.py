"""Command line entry points for the ingestion stage.

    python -m attention_panel.cli plan               config/universe_luxury.yaml
    python -m attention_panel.cli validate-universe  config/universe_luxury.yaml
    python -m attention_panel.cli fetch-attention    config/universe_luxury.yaml
    python -m attention_panel.cli fetch-market       config/universe_luxury.yaml

`plan` works offline. The other three need network access and a contact
address in the User-Agent, which Wikimedia requires -- set it once:

    export ATTENTION_PANEL_UA="attention-panel/0.1 (https://github.com/<you>/<repo>; <you>@example.com)"

RUN `validate-universe` FIRST. The shipped universe file was written without
access to any of these APIs, so its tickers and article titles are unverified
by construction; the validator is what turns them from plausible into checked.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import requests

from .config import load_universe
from .httpcache import HttpClient, RateLimitError
from .market import (
    build_market_features,
    cross_check,
    fetch_stooq,
    fetch_yfinance,
    to_stooq_symbol,
    validate_ohlcv,
)
from .mediawiki import find_title_collisions, resolve_titles, search_titles
from .pageviews import EARLIEST_DATE, article_frame, bot_divergence

log = logging.getLogger("attention_panel")

DATA_DIR = Path("data")
UA_ENV_VAR = "ATTENTION_PANEL_UA"


EXAMPLE_UA = "attention-panel/0.1 (https://github.com/<you>/<repo>; <you>@example.com)"


def _user_agent_help() -> str:
    """Show how to set the variable in the shell the user is actually in.

    `export` is bash syntax and fails on Windows, where the two native shells
    disagree with each other as well. Printing all three costs three lines and
    removes a guaranteed stumble on the very first command anyone runs.
    """
    return (
        f"A descriptive User-Agent with a contact address is required by Wikimedia.\n"
        f"Anonymous clients get throttled, and the symptom is not an error -- it is a run\n"
        f"that quietly returns fewer series than it asked for.\n\n"
        f"Set it for your shell:\n\n"
        f"  bash / zsh (Linux, macOS, Git Bash, WSL)\n"
        f'    export {UA_ENV_VAR}="{EXAMPLE_UA}"\n\n'
        f"  PowerShell\n"
        f'    $env:{UA_ENV_VAR} = "{EXAMPLE_UA}"\n\n'
        f"  Windows CMD  (no quotes, and no spaces around the '=')\n"
        f"    set {UA_ENV_VAR}={EXAMPLE_UA}\n\n"
        f"Or skip the variable entirely and pass it per command:\n"
        f'    attention-panel --user-agent "{EXAMPLE_UA}" plan config/universe_luxury.yaml'
    )


def make_client(args) -> HttpClient:
    user_agent = args.user_agent or os.environ.get(UA_ENV_VAR)
    if not user_agent:
        sys.exit("\n" + _user_agent_help())
    try:
        return HttpClient(user_agent, cache_dir=args.cache_dir)
    except ValueError as exc:
        # The agent was set but is not usable -- generic, too short, or with no
        # way to contact its operator.
        sys.exit(f"\n{exc}\n\n{_user_agent_help()}")


# ---------------------------------------------------------------------------
# plan -- offline
# ---------------------------------------------------------------------------


def cmd_plan(args) -> int:
    """Print what a full fetch would cost, without issuing a single request.

    Worth running before every change to a universe file: the pilot is a few
    thousand requests, which is polite, while a careless Tier 2 configuration
    is a few hundred thousand, which is not.
    """
    universe = load_universe(args.universe)
    titles = universe.article_titles

    print(f"universe          {universe.name}")
    print(f"companies         {len(universe.companies)}")
    print(f"delisted (kept)   {sum(1 for c in universe.companies if c.delisted_on)}")
    print(f"distinct articles {len(titles)}")
    print(f"languages         {', '.join(universe.languages)}")
    print(f"article-language series {universe.fetch_plan_size()}")
    print()
    print("Requests are dominated by redirects: each article needs one pageviews")
    print("call per redirect per language, so expect roughly 5-15x the series count.")
    print("Settled history is cached permanently, so only the first run pays this.")
    return 0


# ---------------------------------------------------------------------------
# validate-universe
# ---------------------------------------------------------------------------


def cmd_validate_universe(args) -> int:
    """Check that every article title and every ticker actually resolves.

    Reports rather than fixes. Silently correcting a universe would hide the
    fact that it was wrong, and a universe that changed after the results were
    seen is no longer a pre-registered universe.

    Exits non-zero if anything needs attention, so it can gate a fetch.
    """
    universe = load_universe(args.universe)
    client = make_client(args)
    failures: list[str] = []

    # -- articles -----------------------------------------------------------

    print(f"Resolving {len(universe.article_titles)} titles on en.wikipedia ...")
    report = resolve_titles(client, "en", list(universe.article_titles), with_redirects=False)
    print(f"  {report.summary()}")

    if report.missing:
        print("\nTITLES THAT DO NOT EXIST")
        print("Candidates from the wiki's own search; verify one and put it in the YAML.")
        print("Do not guess -- a plausible but wrong title resolves silently to another")
        print("entity's traffic, which is worse than a missing one.\n")
        for title in sorted(report.missing):
            print(f"  {title}")
            for hit in search_titles(client, "en", title, limit=4):
                print(f"      -> {hit['title']}")
            failures.append(f"missing title: {title}")

    # -- collisions: the check that matters most -----------------------------

    collisions = find_title_collisions(report.resolved)
    if collisions:
        owner = _title_owners(universe)
        print("\nCOLLISIONS: distinct entries resolving to the SAME article")
        print("Each group below would produce one identical pageview series under several")
        print("names. Across companies that gives two panel entities the same regressor;")
        print("within a company it collapses the brand/corporate distinction the feature")
        print("design depends on. Both must be fixed before fetching.\n")
        for canonical, titles in sorted(collisions.items()):
            print(f"  {canonical}")
            for title in titles:
                print(f"      <- {title}   [{', '.join(owner.get(title, ['theme']))}]")
            failures.append(f"collision on {canonical}")

    if report.redirected:
        print("\nTitles that are redirects (resolved automatically).")
        print("Skim these anyway: a redirect can land on a DIFFERENT entity, which no")
        print("automated check can catch.\n")
        for requested, canonical in sorted(report.redirected):
            print(f"  {requested} -> {canonical}")

    # -- tickers ------------------------------------------------------------

    print(f"\nChecking {len(universe.companies)} tickers against yfinance and Stooq ...")
    start_date = dt.date.today() - dt.timedelta(days=365)
    end_date = dt.date.today()
    stooq_reasons: list[str] = []

    for company in universe.companies:
        try:
            yahoo = fetch_yfinance(company.ticker, start_date, end_date)
        except Exception as exc:  # yfinance raises a different type every release
            print(f"  {company.ticker:12s} yfinance ERROR: {exc}")
            failures.append(f"{company.ticker}: yfinance error")
            continue

        if yahoo.empty:
            # A delisted name having no recent data is expected, not a problem.
            if company.delisted_on is not None:
                print(f"  {company.ticker:12s} no recent data (delisted {company.delisted_on}, expected)")
            else:
                print(f"  {company.ticker:12s} NO DATA -- delisted? add `delisted_on:` to the YAML")
                failures.append(f"{company.ticker}: no data")
            continue

        checks = validate_ohlcv(yahoo, company.ticker)
        line = (
            f"  {company.ticker:12s} {checks['rows']:4d} rows  "
            f"{checks['first_date']}..{checks['last_date']}"
        )

        stooq = fetch_stooq(client, company.ticker, start_date, end_date, args.stooq_user_agent)
        if stooq.ok:
            agreement = cross_check(yahoo, stooq.frame)
            line += f"  stooq ok, mismatches={agreement['mismatches']}"
            if agreement["mismatches"]:
                failures.append(f"{company.ticker}: sources disagree")
        else:
            line += "  stooq FAILED"
            stooq_reasons.append(stooq.reason)

        for flag in ("high_below_others", "low_above_others", "extreme_moves", "unusable_bars"):
            if checks.get(flag):
                line += f"  {flag}={checks[flag]}"
        print(line)

    if stooq_reasons:
        print(f"\nSTOOQ CROSS-CHECK UNAVAILABLE for {len(stooq_reasons)} tickers.")
        print("Stooq answers nearly everything with HTTP 200, so the body is the only")
        print("evidence of what went wrong. Distinct responses seen:\n")
        for reason in sorted({_strip_symbol(r) for r in stooq_reasons})[:5]:
            print(f"  {reason}")
        print("\nThis degrades the cross-check but does not block anything: yfinance is")
        print("the primary source and the structural checks above still run. Try a")
        print("different agent with --stooq-user-agent if the body suggests one.")

    print("\n" + "=" * 70)
    if failures:
        print(f"{len(failures)} problems need attention:")
        for problem in failures[:20]:
            print(f"  - {problem}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
        print("\nFix the YAML before fetching: an unvalidated universe is not pre-registered.")
        return 1
    print("Universe validated: every title resolves uniquely and every ticker has data.")
    return 0


def _title_owners(universe) -> dict[str, list[str]]:
    """Map each article title to the `ticker/family` entries that requested it."""
    owners: dict[str, list[str]] = {}
    for company in universe.companies:
        for article in company.articles:
            owners.setdefault(article.title, []).append(f"{company.ticker}/{article.family.value}")
    return owners


def _strip_symbol(reason: str) -> str:
    """Drop the leading `symbol: ` so identical failures collapse into one line."""
    _, sep, rest = reason.partition(": ")
    return rest if sep else reason


# ---------------------------------------------------------------------------
# fetch-attention
# ---------------------------------------------------------------------------


def cmd_fetch_attention(args) -> int:
    universe = load_universe(args.universe)
    client = make_client(args)
    start, end = _date_range(args)
    out_dir = Path(args.out) / "attention"
    out_dir.mkdir(parents=True, exist_ok=True)

    quality: list[dict] = []

    for lang in universe.languages:
        titles = list(universe.article_titles)
        log.info("resolving %d titles on %s.wikipedia", len(titles), lang)
        report = resolve_titles(client, lang, titles, with_redirects=True)

        frames = {}
        for requested, resolved in report.resolved.items():
            if resolved.canonical is None:
                # No article on this wiki. A missing series and a zero series
                # are different things; the panel builder must see the
                # difference, so nothing is written for this title.
                continue
            frame = article_frame(client, lang, resolved, start, end)
            if frame.empty:
                continue
            frames[requested] = frame["views"]
            quality.append(
                {
                    "language": lang,
                    "requested_title": requested,
                    "canonical_title": resolved.canonical,
                    "qid": resolved.qid,
                    "redirects_summed": len(resolved.redirects),
                    "days": int(len(frame)),
                    "total_views": int(frame["views"].sum()),
                    "bot_divergence": round(bot_divergence(frame), 4),
                }
            )

        if frames:
            wide = pd.DataFrame(frames).rename_axis("date").sort_index()
            path = out_dir / f"pageviews_{lang}.csv.gz"
            wide.to_csv(path, compression="gzip")
            log.info("wrote %s (%d days x %d articles)", path, len(wide), wide.shape[1])

    quality_path = Path(args.out) / "attention_quality.csv"
    pd.DataFrame(quality).to_csv(quality_path, index=False)
    print(f"\nwrote {quality_path}")
    print(f"cache: {client.stats['hits']} hits, {client.stats['misses']} misses, "
          f"{client.stats['retries']} retries")

    # The bot classifier is a heuristic, so a high divergence is a flag for the
    # data-quality table, never a reason to silently drop an article.
    if quality:
        suspicious = [q for q in quality if q["bot_divergence"] > 0.6]
        if suspicious:
            print(f"\n{len(suspicious)} article-language series are >60% bot traffic. "
                  "Inspect before using; they are NOT dropped automatically.")
    return 0


# ---------------------------------------------------------------------------
# fetch-market
# ---------------------------------------------------------------------------


def cmd_fetch_market(args) -> int:
    universe = load_universe(args.universe)
    client = make_client(args)
    start, end = _date_range(args)
    out_dir = Path(args.out) / "market"
    out_dir.mkdir(parents=True, exist_ok=True)

    quality: list[dict] = []
    for company in universe.companies:
        try:
            yahoo = fetch_yfinance(company.ticker, start, end)
        except Exception as exc:
            log.warning("%s: yfinance failed: %s", company.ticker, exc)
            quality.append({"ticker": company.ticker, "error": str(exc)})
            continue

        if yahoo.empty:
            quality.append({"ticker": company.ticker, "error": "no data"})
            continue

        checks = validate_ohlcv(yahoo, company.ticker)
        stooq = fetch_stooq(client, company.ticker, start, end, args.stooq_user_agent)
        checks["stooq_reason"] = stooq.reason
        checks.update({f"xcheck_{k}": v for k, v in cross_check(yahoo, stooq.frame).items()})
        checks.pop("extreme_move_dates", None)
        checks.pop("xcheck_mismatch_dates", None)
        quality.append(checks)

        features = build_market_features(yahoo)
        features.to_csv(out_dir / f"{company.ticker.replace('/', '_')}.csv.gz", compression="gzip")

    quality_path = Path(args.out) / "market_quality.csv"
    pd.DataFrame(quality).to_csv(quality_path, index=False)
    print(f"wrote {quality_path}")
    print(json.dumps({"tickers": len(universe.companies), "written": len(quality)}, indent=2))
    return 0


def _date_range(args) -> tuple[dt.date, dt.date]:
    start = dt.date.fromisoformat(args.start) if args.start else EARLIEST_DATE
    end = dt.date.fromisoformat(args.end) if args.end else dt.date.today()
    return max(start, EARLIEST_DATE), end


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="attention_panel", description=__doc__)
    parser.add_argument("--user-agent", default=None, help=f"overrides ${UA_ENV_VAR}")
    parser.add_argument("--cache-dir", default=".httpcache")
    parser.add_argument(
        "--stooq-user-agent",
        default=None,
        help="agent to send to Stooq only; Wikimedia's contact requirement does not "
             "apply to an unrelated CSV host, so the two are set separately",
    )
    parser.add_argument("--out", default=str(DATA_DIR))
    parser.add_argument("--start", default=None, help="YYYY-MM-DD (default: 2015-07-01)")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD (default: today)")
    parser.add_argument("-v", "--verbose", action="store_true")

    sub = parser.add_subparsers(dest="command", required=True)
    for name, handler, help_text in (
        ("plan", cmd_plan, "print the fetch plan without touching the network"),
        ("validate-universe", cmd_validate_universe, "check every title and ticker resolves"),
        ("fetch-attention", cmd_fetch_attention, "download the multilingual pageview series"),
        ("fetch-market", cmd_fetch_market, "download and cross-check daily OHLCV"),
    ):
        sub_parser = sub.add_parser(name, help=help_text)
        sub_parser.add_argument("universe", help="path to a universe YAML file")
        sub_parser.set_defaults(handler=handler)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )
    universe_path = Path(getattr(args, "universe", ""))
    if not universe_path.exists():
        print(f"universe file not found: {universe_path}", file=sys.stderr)
        print(
            "\nPaths are relative to the current directory, so this usually means the shell "
            "is not\nin the repository. Change into your clone first:\n"
            "\n    cd wikipedia-attention-etf-volatility\n"
            f"\nThen re-run. Available universe files:",
            file=sys.stderr,
        )
        for candidate in sorted(Path("config").glob("*.yaml")) or ["  (none found here either)"]:
            print(f"    {candidate}", file=sys.stderr)
        return 2

    try:
        return args.handler(args)
    except KeyboardInterrupt:
        # A partial fetch is not a problem: the cache makes a resumed run pick
        # up where this one stopped, and settled history is never re-requested.
        print("\ninterrupted; cached progress is kept, re-run to resume", file=sys.stderr)
        return 130
    except (RateLimitError, requests.RequestException) as exc:
        # A wall of urllib3 traceback tells the reader nothing useful. The two
        # causes that matter are an unreachable network and being throttled,
        # and both have the same remedy: wait, then re-run against the cache.
        print(f"\nnetwork failure talking to an upstream API:\n  {exc}", file=sys.stderr)
        print(
            "\nCheck connectivity to wikimedia.org, query.wikidata.org and stooq.com. "
            "Cached progress is kept, so re-running resumes rather than restarts.",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
