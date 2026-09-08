"""Tests for universe loading.

A universe loader that silently drops a company produces a smaller panel and a
*more* significant-looking result, so every check here is deliberately strict.
"""

from __future__ import annotations

import datetime as dt

import pytest
import yaml

from attention_panel.config import ArticleRef, Company, Family, Universe, load_universe

from conftest import REPO_ROOT


def test_the_shipped_pilot_universe_loads():
    universe = load_universe(REPO_ROOT / "config" / "universe_luxury.yaml")
    assert universe.name == "luxury_fashion_tier1"
    assert len(universe.companies) >= 40
    assert "en" in universe.languages and "zh" in universe.languages
    # Every company must expose a corporate article: it is what the Wikidata
    # QID is resolved from, so a missing one is a hard error, not a warning.
    for company in universe.companies:
        assert company.corporate_article


def test_delisted_companies_are_kept_with_their_end_date():
    """Dropping them would be textbook survivorship bias (DESIGN.md 2.3)."""
    universe = load_universe(REPO_ROOT / "config" / "universe_luxury.yaml")
    delisted = [c for c in universe.companies if c.delisted_on]
    assert delisted, "the pilot deliberately includes at least one delisted name"
    assert all(isinstance(c.delisted_on, dt.date) for c in delisted)


def test_duplicate_tickers_are_rejected():
    """A duplicated entity is counted twice by every clustered standard error."""
    article = (ArticleRef("X", Family.CORPORATE),)
    with pytest.raises(ValueError, match="duplicate tickers"):
        Universe(
            name="broken",
            description="",
            languages=("en",),
            companies=(
                Company("AAA", "A", article),
                Company("AAA", "A again", article),
            ),
        )


def test_a_company_without_a_corporate_article_is_rejected(tmp_path):
    path = tmp_path / "u.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": "u",
                "languages": ["en"],
                "companies": [{"ticker": "AAA", "articles": {"brands": ["Only a brand"]}}],
            }
        )
    )
    with pytest.raises(ValueError, match="articles.corporate"):
        load_universe(path)


def test_article_families_are_preserved(tmp_path):
    """Corporate, brand and person attention are different signals (DESIGN.md 3)."""
    path = tmp_path / "u.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": "u",
                "languages": ["en"],
                "theme_terms": ["Met Gala"],
                "companies": [
                    {
                        "ticker": "AAA",
                        "articles": {
                            "corporate": "Acme Inc.",
                            "brands": ["Acme Shoes"],
                            "people": ["Wile E. Coyote"],
                        },
                    }
                ],
            }
        )
    )
    universe = load_universe(path)
    families = {a.family for a in universe.companies[0].articles}
    assert families == {Family.CORPORATE, Family.BRAND, Family.PERSON}
    assert universe.theme_articles[0].family is Family.THEME
    # Themes belong to the universe, not to a company, so a sector-wide media
    # event is absorbed as a common factor rather than attributed to one brand.
    assert all(a.family is not Family.THEME for a in universe.companies[0].articles)


def test_fetch_plan_size_counts_every_article_language_pair():
    universe = load_universe(REPO_ROOT / "config" / "universe_luxury.yaml")
    assert universe.fetch_plan_size() == len(universe.article_titles) * len(universe.languages)
