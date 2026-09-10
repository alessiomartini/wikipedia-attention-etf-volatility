"""MediaWiki Action API: canonical titles, redirects, QIDs and page moves.

WHY THIS MODULE EXISTS AT ALL

The pageviews API counts traffic *per requested title*. A redirect is a title
in its own right, with its own separate counter. So a reader who arrives at
"Christian Dior SE" and a reader who arrives at "Dior" are recorded against two
different series, and asking only for the canonical title returns a fraction of
the real attention.

This matters more than it sounds, and it matters asymmetrically: news coverage
and search engines send readers to whichever alias is in the headline, so the
share of traffic arriving via redirects is HIGHEST exactly on the spike days
that carry the signal. Ignoring redirects therefore does not add symmetric
noise -- it attenuates the events the study is about. (DESIGN.md section 3.2.)

The module also resolves each company's Wikidata QID from its article title,
which is why no QIDs appear in the universe YAML: a mistyped QID resolves to a
real but unrelated entity and no schema check would catch it, whereas a
mistyped article title simply fails to resolve and is reported.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .httpcache import CachePolicy, HttpClient

log = logging.getLogger(__name__)

# The Action API accepts 50 titles per request for unauthenticated clients.
# Batching matters: the pilot has 155 titles across 8 languages, which is 25
# requests batched versus 1240 unbatched.
MAX_TITLES_PER_REQUEST = 50


def api_url(lang: str) -> str:
    return f"https://{lang}.wikipedia.org/w/api.php"


@dataclass
class ResolvedArticle:
    """A title after normalisation and redirect resolution.

    Attributes:
        requested: the title as written in the universe YAML.
        canonical: the article the request lands on, or None if it does not exist.
        qid: Wikidata item id, used to link the same entity across languages.
        redirects: every title that currently redirects INTO `canonical`.
            Their pageviews must be summed with the canonical article's.
        missing: True when the title does not exist on this wiki. Common and
            benign -- most brands have no article on ko.wikipedia -- and it is
            reported rather than raised so a full run is not blocked by it.
    """

    requested: str
    canonical: str | None
    qid: str | None = None
    redirects: tuple[str, ...] = ()
    missing: bool = False

    @property
    def all_titles(self) -> tuple[str, ...]:
        """Canonical plus redirects: the complete set to fetch pageviews for."""
        if self.canonical is None:
            return ()
        return (self.canonical, *self.redirects)


@dataclass
class ResolutionReport:
    """What a resolution pass found, for the run log and the data-quality table."""

    resolved: dict[str, ResolvedArticle] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    redirected: list[tuple[str, str]] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{len(self.resolved)} resolved, {len(self.missing)} missing, "
            f"{len(self.redirected)} were redirects"
        )


def resolve_titles(
    client: HttpClient,
    lang: str,
    titles: list[str],
    *,
    with_redirects: bool = True,
) -> ResolutionReport:
    """Normalise titles, follow redirects, and fetch Wikidata QIDs.

    Args:
        with_redirects: also collect every title redirecting *into* each
            article. Costs one extra request per article but is what makes the
            pageview sums correct; only turn it off for a quick existence check.
    """
    report = ResolutionReport()

    for batch in _chunks(titles, MAX_TITLES_PER_REQUEST):
        payload = client.get_json(
            api_url(lang),
            params={
                "action": "query",
                "format": "json",
                "formatversion": "2",
                "titles": "|".join(batch),
                # `redirects=1` makes the API follow redirects server-side and
                # report the mapping, so a single call gives both the canonical
                # title and the fact that the input was an alias.
                "redirects": "1",
                "prop": "pageprops",
                "ppprop": "wikibase_item",
            },
            # Titles and redirects are editable at any moment, so this is never
            # cached permanently -- unlike settled pageview counts.
            policy=CachePolicy.VOLATILE,
        )
        if payload is None:
            log.warning("no response resolving titles on %s.wikipedia", lang)
            continue

        query = payload.get("query", {})
        # `normalized` covers case and underscore fixes; `redirects` covers real
        # redirects. Both map an input title onto the title the API actually
        # returned, and both must be composed to get back to what was asked for.
        forward: dict[str, str] = {}
        for item in query.get("normalized", ()):
            forward[item["from"]] = item["to"]
        redirect_map: dict[str, str] = {}
        for item in query.get("redirects", ()):
            redirect_map[item["from"]] = item["to"]

        pages_by_title = {page.get("title"): page for page in query.get("pages", ())}

        for requested in batch:
            after_norm = forward.get(requested, requested)
            after_redirect = redirect_map.get(after_norm, after_norm)
            page = pages_by_title.get(after_redirect)

            if page is None or page.get("missing"):
                report.resolved[requested] = ResolvedArticle(
                    requested=requested, canonical=None, missing=True
                )
                report.missing.append(requested)
                continue

            if after_redirect != after_norm:
                report.redirected.append((requested, after_redirect))

            report.resolved[requested] = ResolvedArticle(
                requested=requested,
                canonical=page["title"],
                qid=(page.get("pageprops") or {}).get("wikibase_item"),
            )

    if with_redirects:
        for requested, article in report.resolved.items():
            if article.canonical is None:
                continue
            article.redirects = incoming_redirects(client, lang, article.canonical)

    return report


def incoming_redirects(client: HttpClient, lang: str, title: str) -> tuple[str, ...]:
    """Every main-namespace title that currently redirects into `title`.

    LIMITATION worth stating: this is the redirect graph *as of now*. A title
    that redirected here in 2019 but was retargeted since will be missed, and
    one created last month is credited for its whole history. Both effects are
    small relative to simply ignoring redirects, but they are real, and a
    rigorous version would reconstruct the graph from the page-move and edit
    history instead.
    """
    collected: list[str] = []
    continuation: dict[str, str] = {}

    while True:
        payload = client.get_json(
            api_url(lang),
            params={
                "action": "query",
                "format": "json",
                "formatversion": "2",
                "titles": title,
                "prop": "redirects",
                "rdlimit": "max",
                "rdnamespace": "0",  # article space only: talk pages are not readers
                **continuation,
            },
            policy=CachePolicy.VOLATILE,
        )
        if payload is None:
            break

        for page in payload.get("query", {}).get("pages", ()):
            for redirect in page.get("redirects", ()) or ():
                collected.append(redirect["title"])

        cont = payload.get("continue")
        if not cont:
            break
        # A handful of very popular articles have hundreds of redirects; without
        # following continuation the list is silently truncated at 500.
        continuation = {k: v for k, v in cont.items() if k != "continue"}

    return tuple(collected)


def page_moves(client: HttpClient, lang: str, title: str) -> list[dict]:
    """Move-log entries for an article, newest first.

    A page move breaks a pageview series: traffic before the move is recorded
    under the old title and after it under the new one. Detecting moves is how
    an apparent 'collapse in attention' is distinguished from a rename
    (DESIGN.md section 3.2).
    """
    payload = client.get_json(
        api_url(lang),
        params={
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "list": "logevents",
            "letype": "move",
            "letitle": title,
            "lelimit": "max",
        },
        policy=CachePolicy.VOLATILE,
    )
    if payload is None:
        return []
    return list(payload.get("query", {}).get("logevents", ()))


def _chunks(items: list[str], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def search_titles(client: HttpClient, lang: str, query: str, limit: int = 5) -> list[dict]:
    """Search a wiki for candidate articles matching a title.

    Used by `validate-universe` to propose replacements for titles that do not
    exist. The point is to keep a human in the loop without making them guess:
    the API proposes, the author confirms, and the universe file records a
    title someone actually verified.

    Guessing replacements is what produced the broken titles in the first
    place, and a plausible-but-wrong title is worse than a missing one -- it
    resolves silently to another entity's traffic.
    """
    payload = client.get_json(
        api_url(lang),
        params={
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "list": "search",
            "srsearch": query,
            "srlimit": str(limit),
            "srnamespace": "0",
            "srprop": "snippet",
        },
        policy=CachePolicy.VOLATILE,
    )
    if payload is None:
        return []
    return list(payload.get("query", {}).get("search", ()))


def find_title_collisions(resolved: dict[str, ResolvedArticle]) -> dict[str, list[str]]:
    """Group requested titles by the canonical article they land on.

    Returns only the groups with more than one member -- the collisions.

    WHY THIS IS A HARD ERROR AND NOT A WARNING

    Two distinct entries in a universe that resolve to the same article produce
    the SAME pageview series under two names. Live validation found exactly
    this: "Christian Dior (fashion house)" (a brand of LVMH) and
    "Christian Dior SE" (the corporate article of a separately listed company)
    both redirect to "Dior" on en.wikipedia. Left undetected, two entities of
    the panel would carry a numerically identical regressor, and every
    clustered standard error would treat them as independent evidence.

    The same failure inside one company is subtler and just as wrong: a brand
    article redirecting to its parent ("Levi's 501" -> "Levi Strauss & Co.")
    silently double-counts corporate attention as brand attention, collapsing
    a distinction the feature design depends on.
    """
    groups: dict[str, list[str]] = {}
    for requested, article in resolved.items():
        if article.canonical is None:
            continue
        groups.setdefault(article.canonical, []).append(requested)
    return {canonical: sorted(titles) for canonical, titles in groups.items() if len(titles) > 1}
