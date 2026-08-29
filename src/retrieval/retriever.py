"""
Hybrid Graph-RAG retriever.

Three retrieval modes, selected automatically or explicitly:

  structured  → Cypher query on KuzuDB
                Best for: entity lookups, money path tracing, graph traversals
                e.g. "trace all funds from ACC-4471 to Bluewater Ventures"

  semantic    → Vector similarity search on ChromaDB
                Best for: intent/narrative questions, latent evidence
                e.g. "find communications suggesting fear of detection"

  hybrid      → Graph-first: expand entity subgraph in KuzuDB
                Entity-anchor: collect entity names from graph results
                Vector-filter: search ChromaDB filtered by those entity names
                Best for: complex cross-source questions
                e.g. "what connects Elena Ross to the Rotterdam shipment?"

The retriever does NOT call the LLM — it prepares a RetrievalContext that
the agent reasoning tier consumes. Clean separation of retrieval and reasoning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import kuzu

import chromadb

from src.resolution.alias_resolver import _normalize
from src.storage import kuzu_store, vector_store

# ---------------------------------------------------------------------------
# Keywords that steer the automatic query classifier
# ---------------------------------------------------------------------------

_STRUCTURED_SIGNALS = {
    "transferred", "sent", "received", "wired", "amount", "payment",
    "trace", "path", "connected", "linked", "between", "flow",
    "account", "iban", "txn", "transaction", "flagged",
    "director", "owns", "beneficial owner", "holds",
}

_SEMANTIC_SIGNALS = {
    "suggest", "indicate", "evidence", "describe", "explain",
    "what happened", "why", "intent", "suspicious", "pattern",
    "fear", "evasion", "hide", "launder", "unusual", "concern",
}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class RetrievalContext:
    query: str
    mode: str
    entities_found: list[str] = field(default_factory=list)
    graph_results: list[dict] = field(default_factory=list)
    semantic_results: list[dict] = field(default_factory=list)

    def to_context_string(self) -> str:
        """
        Render retrieval results as a structured string for the LLM context window.
        Keeps graph and semantic evidence clearly attributed for traceability.
        """
        sections: list[str] = [f"[Query mode: {self.mode}]"]

        if self.entities_found:
            sections.append(f"[Entities identified: {', '.join(self.entities_found)}]")

        if self.graph_results:
            sections.append("\n--- Graph Evidence ---")
            for r in self.graph_results[:20]:
                sections.append(str(r))

        if self.semantic_results:
            sections.append("\n--- Document / Comms Evidence ---")
            for r in self.semantic_results:
                meta = r.get("metadata", {})
                source = meta.get("doc_id") or meta.get("event_id", "unknown")
                sections.append(f"[source: {source}] {r['text']}")

        return "\n".join(sections)


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Stateless retrieval layer over KuzuDB (graph) and ChromaDB (vector).
    Instantiate once per pipeline run; thread-safe for read queries.
    """

    def __init__(
        self,
        kuzu_conn: kuzu.Connection,
        chroma_client: chromadb.ClientAPI,
        n_semantic_results: int = 5,
    ) -> None:
        self._conn = kuzu_conn
        self._chroma = chroma_client
        self._n = n_semantic_results
        # Cache entity names for fast in-query entity spotting
        self._entity_names: list[str] = kuzu_store.all_entity_names(kuzu_conn)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, mode: str = "auto") -> RetrievalContext:
        """
        Main entry point. Returns a RetrievalContext ready to be passed to the LLM.
        """
        if mode == "auto":
            mode = self._classify(query)

        entities = self._spot_entities(query)
        ctx = RetrievalContext(query=query, mode=mode, entities_found=entities)

        if mode == "structured":
            ctx.graph_results = self._graph_retrieve(query, entities)

        elif mode == "semantic":
            ctx.semantic_results = self._semantic_retrieve(query)

        else:  # hybrid
            ctx.graph_results = self._graph_retrieve(query, entities)
            # Entity-anchor: use names from graph results to filter vector search
            anchor_entities = entities + self._extract_names_from_graph(ctx.graph_results)
            ctx.semantic_results = self._semantic_retrieve(query, entity_filter=anchor_entities or None)

        return ctx

    # ------------------------------------------------------------------
    # Query classifier
    # ------------------------------------------------------------------

    def _classify(self, query: str) -> str:
        tokens = set(re.findall(r"\w+", query.lower()))
        structured_score = len(tokens & _STRUCTURED_SIGNALS)
        semantic_score   = len(tokens & _SEMANTIC_SIGNALS)
        if structured_score > semantic_score:
            return "structured"
        if semantic_score > structured_score:
            return "semantic"
        return "hybrid"

    # ------------------------------------------------------------------
    # Entity spotting (deterministic — no LLM)
    # ------------------------------------------------------------------

    # Minimum token length for an entity name to be used in spotting.
    # Filters noisy LLM-extracted abbreviations (RO, NL, UK, BVI, JC…)
    _MIN_SPOT_LEN = 4

    def _spot_entities(self, query: str) -> list[str]:
        """
        Return known entity names referenced in the query.

        Two-pass matching:
          1. Exact word-boundary substring match ("Northstar Trading Ltd" in query).
          2. Normalised match — legal suffixes stripped from both the entity name
             and the query before comparison, so "Northstar Trading" in a query
             matches canonical "Northstar Trading Ltd" in the graph.

        Deduplication keeps the longest (most specific) form when multiple
        candidates refer to the same entity (e.g. a bare name and its suffixed form).
        """
        q_lower = " " + query.lower() + " "
        # Also build a suffix-stripped version of the query for normalised matching
        q_norm  = " " + _normalize(query.lower()) + " "

        found: dict[str, str] = {}  # canonical_name → match_key (for dedup)

        for name in self._entity_names:
            if len(name) < self._MIN_SPOT_LEN:
                continue
            n_lower = name.lower()
            n_norm  = _normalize(n_lower)

            # Pass 1: exact word-boundary match
            exact = bool(re.search(
                r'(?<![a-z0-9])' + re.escape(n_lower) + r'(?![a-z0-9])', q_lower
            ))
            # Pass 2: normalised match (suffix-stripped both sides)
            if n_norm and len(n_norm) >= self._MIN_SPOT_LEN:
                normed = bool(re.search(
                    r'(?<![a-z0-9])' + re.escape(n_norm) + r'(?![a-z0-9])', q_norm
                ))
            else:
                normed = False

            if exact or normed:
                found[name] = n_norm or n_lower

        # Prefer longer (more specific) names — deduplicate by normalised form
        candidates = sorted(found.keys(), key=len, reverse=True)
        deduped: list[str] = []
        seen_norms: set[str] = set()
        for name in candidates:
            norm = found[name]
            # Skip if a longer canonical form sharing the same normalised root is kept
            if not any(norm in s for s in seen_norms):
                deduped.append(name)
                seen_norms.add(norm)
        return deduped

    # ------------------------------------------------------------------
    # Graph retrieval (KuzuDB)
    # ------------------------------------------------------------------

    def _graph_retrieve(self, query: str, entities: list[str]) -> list[dict]:
        results = []

        if entities:
            # Expand 2-hop subgraph around each spotted entity
            for name in entities[:3]:  # cap at 3 anchors to control context size
                subgraph = kuzu_store.entity_subgraph(self._conn, name)
                results.extend(subgraph)
        else:
            # No entity anchor — fallback: return flagged financial edges
            results = kuzu_store.cypher(
                self._conn,
                "MATCH (a:Entity)-[r:FINANCIAL_FLOW]->(b:Entity) "
                "WHERE r.flagged = true "
                "RETURN a.name AS sender, r.amount AS amount, r.currency AS currency, "
                "b.name AS receiver, r.timestamp AS timestamp "
                "ORDER BY r.timestamp",
            )
        return results

    # ------------------------------------------------------------------
    # Semantic retrieval (ChromaDB)
    # ------------------------------------------------------------------

    def _semantic_retrieve(
        self,
        query: str,
        entity_filter: Optional[list[str]] = None,
    ) -> list[dict]:
        docs = vector_store.search(
            self._chroma, vector_store.DOCS_COLL, query,
            n_results=self._n, entity_filter=entity_filter,
        )
        comms = vector_store.search(
            self._chroma, vector_store.COMMS_COLL, query,
            n_results=self._n, entity_filter=entity_filter,
        )
        # Interleave by distance (lower = more similar)
        combined = sorted(docs + comms, key=lambda r: r["distance"])
        return combined[:self._n * 2]

    # ------------------------------------------------------------------
    # Helper: extract entity names from graph result dicts
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_names_from_graph(graph_results: list[dict]) -> list[str]:
        names = set()
        for row in graph_results:
            for v in row.values():
                if isinstance(v, str) and 2 < len(v) < 60:
                    names.add(v)
        return list(names)
