"""
Canonical alias resolution for cross-source entity linking.

Resolves raw identifiers (email addresses) to the canonical entity names
established by document extraction, and extracts organisational affiliation
from email domains.

The key deduplication problem: the same entity appears as multiple strings
across sources — "Freight Anon BV" (document), "Freight Anon" (email domain
heuristic), "Shell Corp IO Ltd" (document), "Shell Corp" (heuristic).

Resolution strategy:
  1. Exact email match from document extraction context
  2. Legal-suffix normalisation + fuzzy match against known entities
     (strips BV, Ltd, LLC etc. before comparison)
  3. Heuristic: email local-part → person name, domain → company name
  4. Raw identifier as last resort

The resolver is deterministic and built entirely from the entity store.
It runs at graph construction time so ETL stores raw identifiers and
resolution can be improved without re-running LLM calls.
"""

import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

from src.storage import entity_store

FREE_DOMAINS: frozenset[str] = frozenset({
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
    "live.com", "icloud.com", "me.com", "protonmail.com",
    "freemail.org", "mail.com", "gmx.com", "yandex.com",
    "bank-notify.net",
})

# Legal-form suffixes stripped from the END of a company name before comparison.
# "corp" is intentionally excluded — it appears mid-name (e.g. "Shell Corp IO")
# and stripping it everywhere produces worse matches than leaving it.
_LEGAL_SUFFIXES: frozenset[str] = frozenset({
    "ltd", "limited", "bv", "nv", "inc", "incorporated",
    "llc", "llp", "lp", "gmbh", "ag", "sa", "srl", "plc",
})

_EMAIL_RE = re.compile(r"[\w.\-+]+@[\w.\-]+\.[a-z]{2,}", re.IGNORECASE)

# Fuzzy match threshold for entity deduplication.
# Above this → treat as the same entity and use the canonical name.
# Below this → treat as a distinct entity (new node).
_DEDUP_THRESHOLD = 0.82


def _local_to_name(local: str) -> str:
    """'marcus.vane' → 'Marcus Vane'"""
    parts = re.split(r"[._\-]", local)
    return " ".join(p.capitalize() for p in parts if p)


def _domain_to_company(domain: str) -> str:
    """
    'shell-corp.io'      → 'Shell Corp'
    'freight-anon.com'   → 'Freight Anon'
    'clearbank-demo.com' → 'Clearbank Demo'
    'privatenet.com'     → 'Privatenet'
    """
    base = domain.rsplit(".", 1)[0]
    parts = re.split(r"[\-.]", base)
    return " ".join(p.capitalize() for p in parts if p)


def _normalize(name: str) -> str:
    """
    Strip trailing legal suffixes and lowercase for fuzzy comparison.
    'Freight Anon BV'   → 'freight anon'
    'Shell Corp IO Ltd' → 'shell corp io'

    Suffixes are stripped only from the end so that words like 'Corp'
    appearing mid-name are preserved.
    """
    tokens = name.lower().split()
    while tokens and tokens[-1] in _LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


class AliasResolver:
    """
    Cross-source entity linker built deterministically from the entity store.

    Maintains:
      _email_to_entity  : email      → canonical entity name
      _domain_to_entity : domain     → canonical entity name
      _company_entities : list of (canonical_name, normalized_name) for fuzzy match
    """

    def __init__(self, db_path: Path = entity_store.DB_PATH) -> None:
        self._email_to_entity: dict[str, str] = {}
        self._domain_to_entity: dict[str, str] = {}
        self._company_entities: list[tuple[str, str]] = []
        self._build(db_path)

    def _build(self, db_path: Path) -> None:
        # Collect company entity names for fuzzy deduplication
        for row in entity_store.all_entities(db_path):
            if row["entity_type"].lower() in ("company", "organisation", "organization"):
                name = row["name"]
                self._company_entities.append((name, _normalize(name)))

        # Scan relation contexts for email addresses alongside known entity names
        for row in entity_store.all_relations(db_path):
            context = row["context"] or ""
            for email in _EMAIL_RE.findall(context):
                email_lower = email.lower()
                domain = email_lower.split("@")[1]
                for candidate in (row["source_entity"], row["target_entity"]):
                    if "@" not in candidate:
                        self._email_to_entity.setdefault(email_lower, candidate)
                        if domain not in FREE_DOMAINS:
                            self._domain_to_entity.setdefault(domain, candidate)

    def _deduplicate_company(self, candidate: str) -> Optional[str]:
        """
        Fuzzy-match a candidate company name against all known company entities.

        Normalises both sides (strips legal suffixes) before comparison.
        Returns the canonical entity name if similarity ≥ threshold, else None.

        Example:
          candidate = 'Freight Anon'        (from domain heuristic)
          known     = 'Freight Anon BV'     (from document extraction)
          normalized: 'freight anon' vs 'freight anon' → 1.0 → canonical returned
        """
        norm_candidate = _normalize(candidate)
        best_name: Optional[str] = None
        best_score = 0.0

        for canonical, norm_canonical in self._company_entities:
            score = _similarity(norm_candidate, norm_canonical)
            if score > best_score:
                best_score = score
                best_name = canonical

        if best_score >= _DEDUP_THRESHOLD:
            return best_name
        return None

    def resolve_person(self, raw: str) -> str:
        """
        Resolve a raw email to a canonical person name.
        Pass-through for inputs that contain no '@' (already a name).
        """
        if "@" not in raw:
            return raw

        key = raw.lower()

        if key in self._email_to_entity:
            return self._email_to_entity[key]

        local = key.split("@")[0]
        return _local_to_name(local) or raw

    def resolve_company(self, raw: str) -> Optional[str]:
        """
        Extract and deduplicate the organisational affiliation from an email domain.

        Returns None for free/consumer providers.
        Returns the canonical entity name when the heuristic name fuzzy-matches
        a known company, otherwise returns the heuristic name (new entity).
        """
        if "@" not in raw:
            return None

        domain = raw.lower().split("@")[1]
        if domain in FREE_DOMAINS:
            return None

        # Exact domain → entity mapping from document extraction
        if domain in self._domain_to_entity:
            return self._domain_to_entity[domain]

        # Heuristic name, then try to deduplicate against known companies
        heuristic = _domain_to_company(domain)
        canonical = self._deduplicate_company(heuristic)
        return canonical if canonical is not None else heuristic

    def resolve(self, raw: str) -> tuple[str, Optional[str]]:
        """Convenience: returns (person_node, company_node) in one call."""
        return self.resolve_person(raw), self.resolve_company(raw)
