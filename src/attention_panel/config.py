"""Universe definitions, loaded from YAML rather than hard-coded.

WHY THIS MODULE EXISTS
The central design decision (DESIGN.md section 2.2) is a firm-level *panel*
rather than a single index. The reason is statistical: aggregating brands into
an index destroys the cross-sectional variation, which is exactly where the
information is. If Kering falls on Gucci news, attention on Gucci spikes and
attention on Hermes does not; an index average erases that difference.

A panel only pays off if the universe can grow without touching the code, so a
universe is *data*. The pilot (Tier 1, ~49 fashion and luxury names) and the
generalization test (Tier 2, STOXX 600 + S&P 500) differ only by which YAML
file is passed in.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import yaml


class Family(str, Enum):
    """Which kind of attention an article measures.

    These are kept as separate features rather than summed, because they are
    economically different signals and mixing them destroys the distinction
    (DESIGN.md section 3):

    CORPORATE  the listed entity ("LVMH")      -> investor / financial-news attention
    BRAND      a product line ("Louis Vuitton") -> consumer demand
    PERSON     CEOs, creative directors         -> idiosyncratic / succession risk
    THEME      "Met Gala", "Quiet luxury"       -> sector-wide media cycle

    THEME articles are attached to the universe, not to any single company, so
    that a Met Gala spike is absorbed as a sector-wide event instead of being
    attributed to whichever brand happens to be most read that week.
    """

    CORPORATE = "corporate"
    BRAND = "brand"
    PERSON = "person"
    THEME = "theme"


@dataclass(frozen=True)
class ArticleRef:
    """A Wikipedia article title, before redirect resolution.

    `title` is the title as a human wrote it in the YAML. It is NOT assumed to
    be canonical: `mediawiki.resolve_titles` maps it to the article it actually
    redirects to, and `pageviews` sums the traffic of every redirect pointing
    at that article. Skipping that step silently discards a large share of the
    signal, because news coverage and search engines send readers to whichever
    alias is in the headline (DESIGN.md section 3.2).
    """

    title: str
    family: Family


@dataclass(frozen=True)
class Company:
    """One panel entity: a listed company and the articles that describe it."""

    ticker: str
    name: str
    articles: tuple[ArticleRef, ...]
    delisted_on: dt.date | None = None

    @property
    def corporate_article(self) -> str:
        """The article used to resolve this company's Wikidata QID.

        The QID is looked up from this title at runtime rather than written in
        the YAML: a mistyped QID resolves to a real but unrelated entity, and
        no schema validation would catch it. A mistyped article title, by
        contrast, simply fails to resolve and is reported.
        """
        for article in self.articles:
            if article.family is Family.CORPORATE:
                return article.title
        raise ValueError(f"{self.ticker} has no article of family 'corporate'")


@dataclass(frozen=True)
class Universe:
    """A set of companies to build a panel over, plus the languages to fetch."""

    name: str
    description: str
    languages: tuple[str, ...]
    companies: tuple[Company, ...]
    theme_articles: tuple[ArticleRef, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        tickers = [c.ticker for c in self.companies]
        duplicates = {t for t in tickers if tickers.count(t) > 1}
        if duplicates:
            # A duplicate ticker would enter the panel twice and be counted
            # twice by every clustered standard error, inflating significance.
            raise ValueError(f"duplicate tickers in universe {self.name!r}: {sorted(duplicates)}")

    @property
    def article_titles(self) -> tuple[str, ...]:
        """Every distinct title in the universe, company and theme alike."""
        seen: dict[str, None] = {}
        for company in self.companies:
            for article in company.articles:
                seen[article.title] = None
        for article in self.theme_articles:
            seen[article.title] = None
        return tuple(seen)

    def fetch_plan_size(self) -> int:
        """Number of (article, language) series a full fetch will request.

        Worth printing before a run: the pilot is a few thousand requests,
        which is polite; a careless Tier 2 configuration is a few hundred
        thousand, which is not.
        """
        return len(self.article_titles) * len(self.languages)


def load_universe(path: str | Path) -> Universe:
    """Read a universe YAML file into typed objects.

    Validation is strict and fails loudly. An ingestion bug that silently drops
    a company produces a smaller panel and a *more* significant-looking result,
    so there is no safe way to be permissive here.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")

    for key in ("name", "languages", "companies"):
        if key not in raw:
            raise ValueError(f"{path}: missing required key {key!r}")

    languages = tuple(raw["languages"])
    if not languages:
        raise ValueError(f"{path}: 'languages' must list at least one wiki language code")

    companies = tuple(_parse_company(entry, path) for entry in raw["companies"])
    themes = tuple(
        ArticleRef(title=str(t), family=Family.THEME) for t in raw.get("theme_terms", ())
    )

    return Universe(
        name=str(raw["name"]),
        description=str(raw.get("description", "")).strip(),
        languages=languages,
        companies=companies,
        theme_articles=themes,
    )


def _parse_company(entry: dict, path: Path) -> Company:
    if "ticker" not in entry:
        raise ValueError(f"{path}: a company entry is missing 'ticker'")
    ticker = str(entry["ticker"])

    articles_block = entry.get("articles") or {}
    articles: list[ArticleRef] = []

    corporate = articles_block.get("corporate")
    if not corporate:
        raise ValueError(f"{path}: {ticker} must define articles.corporate")
    articles.append(ArticleRef(title=str(corporate), family=Family.CORPORATE))

    for title in articles_block.get("brands", ()) or ():
        articles.append(ArticleRef(title=str(title), family=Family.BRAND))
    for title in articles_block.get("people", ()) or ():
        articles.append(ArticleRef(title=str(title), family=Family.PERSON))

    delisted = entry.get("delisted_on")
    if delisted is not None and not isinstance(delisted, dt.date):
        delisted = dt.date.fromisoformat(str(delisted))

    return Company(
        ticker=ticker,
        name=str(entry.get("name", ticker)),
        articles=tuple(articles),
        # Delisted companies are KEPT (DESIGN.md section 2.3). Dropping a name
        # because its series ends early is the textbook route to survivorship
        # bias: the survivors are by construction the firms whose attention
        # shocks did not kill them. The panel is unbalanced on purpose.
        delisted_on=delisted,
    )
