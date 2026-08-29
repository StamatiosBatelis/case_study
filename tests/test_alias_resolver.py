"""Tests for AliasResolver — alias resolution, domain extraction, deduplication."""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.resolution.alias_resolver import (
    AliasResolver,
    _domain_to_company,
    _local_to_name,
    _normalize,
)


# ---------------------------------------------------------------------------
# Pure function tests — no DB needed
# ---------------------------------------------------------------------------

class TestLocalToName:
    def test_dot_separated(self) -> None:
        assert _local_to_name("marcus.vane") == "Marcus Vane"

    def test_hyphen_separated(self) -> None:
        assert _local_to_name("j-chen") == "J Chen"

    def test_single_part(self) -> None:
        assert _local_to_name("elena") == "Elena"

    def test_underscore_separated(self) -> None:
        assert _local_to_name("james_chen") == "James Chen"


class TestDomainToCompany:
    def test_hyphenated_domain(self) -> None:
        assert _domain_to_company("shell-corp.io") == "Shell Corp"

    def test_single_word_domain(self) -> None:
        assert _domain_to_company("privatenet.com") == "Privatenet"

    def test_multi_hyphen(self) -> None:
        assert _domain_to_company("freight-anon.com") == "Freight Anon"

    def test_clearbank(self) -> None:
        assert _domain_to_company("clearbank-demo.com") == "Clearbank Demo"


# ---------------------------------------------------------------------------
# AliasResolver tests — DB mocked
# ---------------------------------------------------------------------------

def _make_resolver_no_db() -> AliasResolver:
    """Resolver with empty lookup tables (no DB needed)."""
    with patch.object(AliasResolver, "_build", lambda self, _: None):
        return AliasResolver(db_path=Path("/nonexistent.db"))


class TestResolverHeuristics:
    def test_resolve_person_heuristic(self) -> None:
        r = _make_resolver_no_db()
        assert r.resolve_person("marcus.vane@shell-corp.io") == "Marcus Vane"

    def test_resolve_person_passthrough_for_names(self) -> None:
        r = _make_resolver_no_db()
        assert r.resolve_person("Elena Ross") == "Elena Ross"

    def test_resolve_company_free_domain_returns_none(self) -> None:
        r = _make_resolver_no_db()
        assert r.resolve_company("j.chen@freemail.org") is None

    def test_resolve_company_org_domain(self) -> None:
        r = _make_resolver_no_db()
        assert r.resolve_company("marcus.vane@shell-corp.io") == "Shell Corp"

    def test_resolve_company_privatenet(self) -> None:
        r = _make_resolver_no_db()
        assert r.resolve_company("elena.ross@privatenet.com") == "Privatenet"

    def test_resolve_returns_tuple(self) -> None:
        r = _make_resolver_no_db()
        person, company = r.resolve("marcus.vane@shell-corp.io")
        assert person == "Marcus Vane"
        assert company == "Shell Corp"

    def test_resolve_free_email_no_company(self) -> None:
        r = _make_resolver_no_db()
        _, company = r.resolve("j.chen@freemail.org")
        assert company is None


class TestNormalize:
    def test_strips_legal_suffix(self) -> None:
        assert _normalize("Freight Anon BV") == "freight anon"

    def test_strips_ltd(self) -> None:
        assert _normalize("Shell Corp IO Ltd") == "shell corp io"

    def test_no_suffix(self) -> None:
        assert _normalize("Harrington Capital Partners") == "harrington capital partners"


class TestDeduplication:
    def test_freight_anon_matches_freight_anon_bv(self) -> None:
        r = _make_resolver_no_db()
        r._company_entities = [("Freight Anon BV", _normalize("Freight Anon BV"))]
        result = r._deduplicate_company("Freight Anon")
        assert result == "Freight Anon BV"

    def test_shell_corp_matches_shell_corp_io_ltd(self) -> None:
        r = _make_resolver_no_db()
        r._company_entities = [("Shell Corp IO Ltd", _normalize("Shell Corp IO Ltd"))]
        result = r._deduplicate_company("Shell Corp")
        assert result == "Shell Corp IO Ltd"

    def test_no_match_below_threshold(self) -> None:
        r = _make_resolver_no_db()
        r._company_entities = [("Harrington Capital Partners", _normalize("Harrington Capital Partners"))]
        result = r._deduplicate_company("Freight Anon")
        assert result is None

    def test_resolve_company_returns_canonical_via_dedup(self) -> None:
        r = _make_resolver_no_db()
        r._company_entities = [("Freight Anon BV", _normalize("Freight Anon BV"))]
        # freight-anon.com → heuristic "Freight Anon" → deduped → "Freight Anon BV"
        assert r.resolve_company("logistics@freight-anon.com") == "Freight Anon BV"

    def test_resolve_company_new_entity_when_no_match(self) -> None:
        r = _make_resolver_no_db()
        r._company_entities = []  # empty store — no known companies
        assert r.resolve_company("logistics@freight-anon.com") == "Freight Anon"


class TestResolverDBLookup:
    def test_exact_email_match_preferred_over_heuristic(self) -> None:
        r = _make_resolver_no_db()
        r._email_to_entity["elena.ross@privatenet.com"] = "Elena Ross"
        assert r.resolve_person("elena.ross@privatenet.com") == "Elena Ross"

    def test_exact_domain_match(self) -> None:
        r = _make_resolver_no_db()
        r._domain_to_entity["shell-corp.io"] = "Shell Corp IO Ltd"
        assert r.resolve_company("some.person@shell-corp.io") == "Shell Corp IO Ltd"
