"""Wikidata: objective universe construction and cross-language article mapping.

WHY THE UNIVERSE COMES FROM WIKIDATA AND NOT FROM A HAND-WRITTEN LIST

This is the project's defence against the single most damaging criticism an
attention study can attract: *you picked the companies and search terms that
worked*. With 600 stocks and 30 features there are 18,000 tests, of which
roughly 900 look significant at 5% by chance alone -- in both directions, all
of them convincing (DESIGN.md section 6). A universe selected after seeing the
data is that problem in its purest form.

So membership is defined by properties on Wikidata, decided before any result
is seen and reproducible by anyone:

    P414  stock exchange the company is listed on
    P249  ticker symbol (usually a qualifier on the P414 statement)
    P946  ISIN
    P452  industry

WHY SITELINKS MATTER

The same company is one Wikidata item with a different article title in every
language: Q193592 is "LVMH" on en.wikipedia and "LVMH" on ja.wikipedia but
"LVMH集团" would be wrong to guess. Sitelinks give the true titles, which is
what makes the multilingual design (DESIGN.md section 2.4) possible without
hand-maintaining eight titles per company.

CAVEAT ON zh: Wikipedia has been blocked in mainland China since 2019, so
zh.wikipedia readership is Greater-China-ex-mainland, NOT the mainland luxury
consumer. Labelled as such everywhere it is used.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .httpcache import CachePolicy, HttpClient

log = logging.getLogger(__name__)

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

#: Wikidata QIDs for the exchanges Tier 2 would be drawn from.
EXCHANGES = {
    "NYSE": "Q13677",
    "NASDAQ": "Q82059",
    "Euronext Paris": "Q2385849",
    "Borsa Italiana": "Q1585225",
    "London Stock Exchange": "Q171240",
    "Frankfurt Stock Exchange": "Q151139",
    "SIX Swiss Exchange": "Q652941",
    "Tokyo Stock Exchange": "Q217475",
    "Hong Kong Stock Exchange": "Q496672",
    "Bolsa de Madrid": "Q1770211",
    "Nasdaq Stockholm": "Q1163801",
    "Nasdaq Copenhagen": "Q1163803",
}

#: Industry QIDs describing the fashion / luxury sector. Subclasses are
#: traversed at query time, so a company tagged with a narrower industry
#: (e.g. "shoe industry") is still captured.
FASHION_INDUSTRY_QIDS = ("Q1520670", "Q3251801", "Q1725664", "Q1479677")


@dataclass(frozen=True)
class WikidataEntity:
    qid: str
    label: str
    tickers: tuple[str, ...] = ()
    exchanges: tuple[str, ...] = ()
    isin: str | None = None


def run_sparql(client: HttpClient, query: str, policy: CachePolicy = CachePolicy.VOLATILE) -> list[dict]:
    """Execute a SPARQL query and return its bindings as plain dicts.

    The Wikidata Query Service enforces a 60-second timeout and, like every
    Wikimedia endpoint, expects a descriptive User-Agent. Broad discovery
    queries do time out; when that happens, split them by exchange rather than
    retrying the same query, since retrying a query that is too expensive just
    times out again more slowly.
    """
    payload = client.get_json(
        SPARQL_ENDPOINT,
        params={"query": query, "format": "json"},
        headers={"Accept": "application/sparql-results+json"},
        policy=policy,
    )
    if payload is None:
        return []
    rows = []
    for binding in payload.get("results", {}).get("bindings", ()):
        rows.append({key: value.get("value") for key, value in binding.items()})
    return rows


def sitelinks(client: HttpClient, qids: list[str], languages: list[str]) -> dict[str, dict[str, str]]:
    """Map each QID to ``{language: article title}``.

    Returns only the languages that actually have an article; a company with no
    ko.wikipedia page simply has no "ko" key, which the caller treats as a
    missing series rather than a zero one. Those are different things: a zero
    means nobody read the article, a missing means there is nothing to read.
    """
    if not qids:
        return {}

    values = " ".join(f"wd:{qid}" for qid in qids)
    lang_filter = ", ".join(f'"{lang}"' for lang in languages)
    query = f"""
    SELECT ?item ?lang ?title WHERE {{
      VALUES ?item {{ {values} }}
      ?sitelink schema:about ?item ;
                schema:inLanguage ?lang ;
                schema:name ?title ;
                schema:isPartOf ?wiki .
      FILTER(STRENDS(STR(?wiki), ".wikipedia.org/"))
      FILTER(?lang IN ({lang_filter}))
    }}
    """
    out: dict[str, dict[str, str]] = {}
    for row in run_sparql(client, query):
        qid = row["item"].rsplit("/", 1)[-1]
        out.setdefault(qid, {})[row["lang"]] = row["title"]
    return out


def discover_listed_companies(
    client: HttpClient,
    exchange_qids: list[str],
    industry_qids: list[str] | None = None,
    limit: int = 5000,
) -> list[WikidataEntity]:
    """Every company listed on the given exchanges, optionally filtered by industry.

    This is the Tier 2 universe builder, and the authoritative version of Tier 1:
    when it disagrees with the hand-written YAML seed, the query wins and the
    disagreement is itself worth reporting (a company the author forgot is
    exactly the kind of omission that biases a hand-picked universe).

    NOTE ON POINT-IN-TIME MEMBERSHIP: this returns *current* listings. A study
    that treats today's constituents as the historical universe has survivorship
    bias, because firms that delisted after a bad shock are absent. DESIGN.md
    section 9 keeps this open; until it is resolved, results on Tier 2 must be
    reported with the bias stated.
    """
    exchange_values = " ".join(f"wd:{qid}" for qid in exchange_qids)

    industry_clause = ""
    if industry_qids:
        industry_values = " ".join(f"wd:{qid}" for qid in industry_qids)
        # `wdt:P452/wdt:P279*` walks up the subclass tree, so a company tagged
        # only with a narrow industry still matches a broad filter.
        industry_clause = f"""
      VALUES ?industryRoot {{ {industry_values} }}
      ?item wdt:P452/wdt:P279* ?industryRoot .
        """

    query = f"""
    SELECT DISTINCT ?item ?itemLabel ?ticker ?exchange ?isin WHERE {{
      VALUES ?exchange {{ {exchange_values} }}
      ?item p:P414 ?listing .
      ?listing ps:P414 ?exchange .
      OPTIONAL {{ ?listing pq:P249 ?ticker . }}
      OPTIONAL {{ ?item wdt:P946 ?isin . }}
      {industry_clause}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT {limit}
    """

    merged: dict[str, dict] = {}
    for row in run_sparql(client, query):
        qid = row["item"].rsplit("/", 1)[-1]
        entry = merged.setdefault(
            qid, {"label": row.get("itemLabel", qid), "tickers": set(), "exchanges": set(), "isin": None}
        )
        if row.get("ticker"):
            entry["tickers"].add(row["ticker"])
        if row.get("exchange"):
            entry["exchanges"].add(row["exchange"].rsplit("/", 1)[-1])
        if row.get("isin"):
            entry["isin"] = row["isin"]

    return [
        WikidataEntity(
            qid=qid,
            label=entry["label"],
            tickers=tuple(sorted(entry["tickers"])),
            exchanges=tuple(sorted(entry["exchanges"])),
            isin=entry["isin"],
        )
        for qid, entry in sorted(merged.items())
    ]
