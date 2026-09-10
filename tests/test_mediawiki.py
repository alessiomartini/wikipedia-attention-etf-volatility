"""Tests for title resolution.

Redirect resolution is the step that decides whether the attention series
measures the event or a fraction of it, so it is tested against a payload that
contains all four cases at once: a normalisation, a real redirect, a canonical
title and a missing page.
"""

from __future__ import annotations

from attention_panel import mediawiki

PAYLOAD = {
    "query": {
        "normalized": [{"from": "gucci", "to": "Gucci"}],
        "redirects": [{"from": "Dior", "to": "Christian Dior (fashion house)"}],
        "pages": [
            {
                "title": "Gucci",
                "pageprops": {"wikibase_item": "Q189141"},
            },
            {
                "title": "Christian Dior (fashion house)",
                "pageprops": {"wikibase_item": "Q542767"},
            },
            {"title": "No Such Brand", "missing": True},
        ],
    }
}


def test_normalisation_and_redirects_are_composed(fake_client):
    """'gucci' -> 'Gucci' is a normalisation; 'Dior' -> ... is a redirect.

    Both map an input title onto what the API actually returned, and only
    applying both gets each requested title back to its own result.
    """
    client = fake_client({"api.php": PAYLOAD})
    report = mediawiki.resolve_titles(
        client, "en", ["gucci", "Dior", "No Such Brand"], with_redirects=False
    )

    assert report.resolved["gucci"].canonical == "Gucci"
    assert report.resolved["Dior"].canonical == "Christian Dior (fashion house)"
    assert ("Dior", "Christian Dior (fashion house)") in report.redirected


def test_a_missing_page_is_reported_not_raised(fake_client):
    """Most brands have no article on most wikis. That is data, not a failure."""
    client = fake_client({"api.php": PAYLOAD})
    report = mediawiki.resolve_titles(client, "en", ["No Such Brand"], with_redirects=False)

    article = report.resolved["No Such Brand"]
    assert article.missing is True
    assert article.canonical is None
    assert article.all_titles == ()  # nothing to fetch pageviews for
    assert "No Such Brand" in report.missing


def test_qids_come_from_the_article_not_from_hand_written_yaml(fake_client):
    """A mistyped QID resolves to a real but unrelated entity, uncaught.

    Resolving it from a title the author can read and verify removes that
    entire class of error, which is why the universe YAML contains no QIDs.
    """
    client = fake_client({"api.php": PAYLOAD})
    report = mediawiki.resolve_titles(client, "en", ["gucci"], with_redirects=False)
    assert report.resolved["gucci"].qid == "Q189141"


def test_titles_are_batched_to_stay_within_the_api_limit(fake_client):
    """155 titles across 8 languages is 25 requests batched, 1240 unbatched."""
    client = fake_client({"api.php": {"query": {"pages": []}}})
    titles = [f"Article {i}" for i in range(120)]
    mediawiki.resolve_titles(client, "en", titles, with_redirects=False)
    assert len(client.calls) == 3  # ceil(120 / 50)
    for _, params in client.calls:
        assert len(params["titles"].split("|")) <= mediawiki.MAX_TITLES_PER_REQUEST


def test_incoming_redirects_follows_continuation(fake_client):
    """Without this, a popular article's redirect list is silently truncated."""

    class Paging:
        def __init__(self):
            self.n = 0
            self.calls = []

        def get_json(self, url, params=None, **kwargs):
            self.calls.append(params)
            self.n += 1
            if self.n == 1:
                return {
                    "query": {"pages": [{"redirects": [{"title": "Alias A"}]}]},
                    "continue": {"rdcontinue": "next", "continue": "||"},
                }
            return {"query": {"pages": [{"redirects": [{"title": "Alias B"}]}]}}

    client = Paging()
    assert mediawiki.incoming_redirects(client, "en", "Gucci") == ("Alias A", "Alias B")
    assert client.calls[1]["rdcontinue"] == "next"
    # The bare `continue` key must not be echoed back as a query parameter.
    assert "continue" not in client.calls[1]


def test_collisions_are_detected_when_two_titles_share_one_article():
    """The failure live validation found: two universe entries, one series.

    'Christian Dior (fashion house)' is a brand of LVMH and 'Christian Dior SE'
    is the corporate article of a separately listed company, but en.wikipedia
    redirects both to 'Dior'. Undetected, two panel entities would carry an
    identical regressor while every clustered standard error treated them as
    independent evidence.
    """
    from attention_panel.mediawiki import ResolvedArticle, find_title_collisions

    resolved = {
        "Christian Dior (fashion house)": ResolvedArticle("Christian Dior (fashion house)", "Dior"),
        "Christian Dior SE": ResolvedArticle("Christian Dior SE", "Dior"),
        "Gucci": ResolvedArticle("Gucci", "Gucci"),
        "No Such Brand": ResolvedArticle("No Such Brand", None, missing=True),
    }
    collisions = find_title_collisions(resolved)

    assert collisions == {"Dior": ["Christian Dior (fashion house)", "Christian Dior SE"]}
    assert "Gucci" not in collisions          # a unique title is not a collision
    # A missing title has no canonical article, so it cannot collide with
    # another missing one -- they must not be grouped together under None.
    assert None not in collisions


def test_search_returns_candidates_for_a_title_that_does_not_exist(fake_client):
    """The validator proposes; the author confirms. Nobody guesses."""
    from attention_panel.mediawiki import search_titles

    client = fake_client(
        {"list=search": {"query": {"search": [{"title": "Ugg boots", "snippet": "..."}]}}}
    )
    results = search_titles(client, "en", "Ugg (brand)")
    assert results[0]["title"] == "Ugg boots"
