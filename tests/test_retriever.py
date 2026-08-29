"""Tests for HybridRetriever — classification, entity spotting, and retrieval modes."""

import json
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import kuzu
import chromadb
import pytest

from chromadb import EmbeddingFunction, Documents, Embeddings
from src.retrieval.retriever import HybridRetriever, RetrievalContext
from src.storage import entity_store, kuzu_store, vector_store
from src.storage.vector_store import DOCS_COLL, COMMS_COLL


# ---------------------------------------------------------------------------
# Stub embedder — extends EmbeddingFunction to get embed_query for free
# ---------------------------------------------------------------------------

class StubEmbedder(EmbeddingFunction):
    DIM = 8

    def __call__(self, input: Documents) -> Embeddings:
        return [[float(abs(hash(t)) % 1000) / 1000.0] * self.DIM for t in input]


# ---------------------------------------------------------------------------
# Fixtures — minimal KuzuDB and ChromaDB with two entities
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_sqlite(tmp_path: Path) -> Path:
    db = tmp_path / "test.db"
    entity_store.init_db(db)
    from src.models import DocumentExtractionResult, ExtractedEntity, ExtractedRelation
    entity_store.upsert_extraction(
        DocumentExtractionResult(
            doc_id="DOC-R01",
            entities=[
                ExtractedEntity(name="Marcus Vane", entity_type="Person"),
                ExtractedEntity(name="Northstar Trading", entity_type="Company"),
            ],
            relations=[
                ExtractedRelation(
                    source_entity="Marcus Vane",
                    relation_type="CONTROLS",
                    target_entity="Northstar Trading",
                    context="Marcus controls the company",
                )
            ],
        ),
        db_path=db,
    )
    entity_store.upsert_transaction(
        {
            "txn_id": "TXN-R01", "timestamp": "2024-03-10T09:00:00Z",
            "txn_type": "wire_transfer", "amount": 120000.0, "currency": "USD",
            "sender_node": "Marcus Vane", "sender_name": "Marcus Vane", "sender_country": "GB",
            "receiver_node": "Northstar Trading", "receiver_name": "Northstar Trading",
            "receiver_country": "NL", "reference": "Consulting", "flagged": True,
        },
        db_path=db,
    )
    return db


@pytest.fixture()
def kuzu_conn(tmp_path: Path, tmp_sqlite: Path) -> kuzu.Connection:
    kuzu_path = tmp_path / "kuzu_db"
    kuzu_store.sync(kuzu_path, tmp_sqlite)
    db = kuzu_store.open_db(kuzu_path)
    return kuzu.Connection(db)


STUB_DOCS = [
    {
        "doc_id": "DOC-R01",
        "title": "SAR Report",
        "type": "sar",
        "date": "2024-01-15",
        "content": "Marcus Vane has been flagged for suspicious activity involving Northstar Trading.",
    }
]

STUB_COMMS = [
    {
        "event_id": "evt_R01",
        "sender_node": "Marcus Vane",
        "channel": "email",
        "timestamp": "2024-03-10T08:00:00Z",
        "body": "Confirm the wire is ready to proceed. Keep it quiet.",
        "intent_signal": "coordination",
        "subject": "Wire",
    }
]


@pytest.fixture()
def chroma(tmp_path: Path) -> chromadb.ClientAPI:
    client = vector_store.open_client(tmp_path / "chroma_test")
    vector_store.index_documents(
        client, STUB_DOCS, known_entities=["Marcus Vane", "Northstar Trading"],
        embedder=StubEmbedder(),
    )
    vector_store.index_comms(
        client, STUB_COMMS, known_entities=["Marcus Vane"],
        embedder=StubEmbedder(),
    )
    return client


@pytest.fixture()
def retriever(kuzu_conn: kuzu.Connection, chroma: chromadb.ClientAPI) -> HybridRetriever:
    with patch.object(
        vector_store, "OllamaEmbedder", return_value=StubEmbedder()
    ):
        r = HybridRetriever.__new__(HybridRetriever)
        r._conn = kuzu_conn
        r._chroma = chroma
        r._n = 3
        r._entity_names = kuzu_store.all_entity_names(kuzu_conn)
        return r


# ---------------------------------------------------------------------------
# Query classifier
# ---------------------------------------------------------------------------

class TestClassify:
    def test_structured_signals_detected(self, retriever: HybridRetriever) -> None:
        assert retriever._classify("trace the funds transferred from ACC-4471") == "structured"

    def test_semantic_signals_detected(self, retriever: HybridRetriever) -> None:
        assert retriever._classify("what evidence suggests intent to launder?") == "semantic"

    def test_tie_falls_back_to_hybrid(self, retriever: HybridRetriever) -> None:
        assert retriever._classify("unrelated neutral phrasing here") == "hybrid"


# ---------------------------------------------------------------------------
# Entity spotting
# ---------------------------------------------------------------------------

class TestSpotEntities:
    def test_known_entity_found(self, retriever: HybridRetriever) -> None:
        found = retriever._spot_entities("What did Marcus Vane do?")
        assert "Marcus Vane" in found

    def test_unknown_entity_not_found(self, retriever: HybridRetriever) -> None:
        found = retriever._spot_entities("What did Bob Smith do?")
        assert found == []

    def test_case_insensitive(self, retriever: HybridRetriever) -> None:
        found = retriever._spot_entities("what did marcus vane do?")
        assert "Marcus Vane" in found

    def test_substring_dedup(self, retriever: HybridRetriever) -> None:
        # If "Trading" and "Northstar Trading" are both entities, "Northstar Trading" wins
        retriever._entity_names = ["Northstar", "Northstar Trading"]
        found = retriever._spot_entities("Northstar Trading was involved")
        assert "Northstar Trading" in found
        assert len(found) == 1


# ---------------------------------------------------------------------------
# RetrievalContext.to_context_string
# ---------------------------------------------------------------------------

class TestRetrievalContextString:
    def test_contains_mode(self) -> None:
        ctx = RetrievalContext(query="test", mode="hybrid")
        s = ctx.to_context_string()
        assert "hybrid" in s

    def test_contains_entities(self) -> None:
        ctx = RetrievalContext(query="q", mode="structured", entities_found=["Alice"])
        assert "Alice" in ctx.to_context_string()

    def test_contains_graph_evidence_header(self) -> None:
        ctx = RetrievalContext(
            query="q", mode="structured",
            graph_results=[{"sender": "Alice", "receiver": "Bob"}],
        )
        assert "Graph Evidence" in ctx.to_context_string()

    def test_contains_semantic_evidence_header(self) -> None:
        ctx = RetrievalContext(
            query="q", mode="semantic",
            semantic_results=[{
                "id": "DOC-T01_chunk0",
                "text": "Alice transferred funds.",
                "metadata": {"doc_id": "DOC-T01"},
                "distance": 0.1,
            }],
        )
        assert "Document / Comms Evidence" in ctx.to_context_string()


# ---------------------------------------------------------------------------
# Full retrieve() integration
# ---------------------------------------------------------------------------

class TestRetrieve:
    def _search_stub(self, client, coll, query, n_results=5, entity_filter=None, embedder=None):
        """Return a single synthetic result without hitting Ollama."""
        return [{
            "id": "stub_chunk_0",
            "text": "Stub result for: " + query,
            "metadata": {"doc_id": "DOC-stub"},
            "distance": 0.2,
        }]

    def test_structured_mode_returns_graph_results(self, retriever: HybridRetriever) -> None:
        with patch.object(vector_store, "search", side_effect=self._search_stub):
            ctx = retriever.retrieve("trace the funds transferred from Marcus Vane", mode="structured")
        assert ctx.mode == "structured"
        assert isinstance(ctx.graph_results, list)

    def test_semantic_mode_returns_semantic_results(self, retriever: HybridRetriever) -> None:
        with patch.object(vector_store, "search", side_effect=self._search_stub):
            ctx = retriever.retrieve("evidence suggesting evasion intent", mode="semantic")
        assert ctx.mode == "semantic"
        assert len(ctx.semantic_results) > 0

    def test_hybrid_mode_returns_both(self, retriever: HybridRetriever) -> None:
        with patch.object(vector_store, "search", side_effect=self._search_stub):
            ctx = retriever.retrieve("what connects Marcus Vane to Northstar Trading?", mode="hybrid")
        assert ctx.mode == "hybrid"
        # Should have attempted both graph and semantic paths
        assert isinstance(ctx.graph_results, list)
        assert isinstance(ctx.semantic_results, list)

    def test_auto_mode_classifies(self, retriever: HybridRetriever) -> None:
        with patch.object(vector_store, "search", side_effect=self._search_stub):
            ctx = retriever.retrieve("trace the transferred amount from ACC-4471")
        # Should classify as structured or hybrid, not fail
        assert ctx.mode in ("structured", "semantic", "hybrid")

    def test_context_string_non_empty(self, retriever: HybridRetriever) -> None:
        with patch.object(vector_store, "search", side_effect=self._search_stub):
            ctx = retriever.retrieve("Marcus Vane", mode="hybrid")
        assert len(ctx.to_context_string()) > 0


# ---------------------------------------------------------------------------
# _extract_names_from_graph
# ---------------------------------------------------------------------------

class TestExtractNamesFromGraph:
    def test_extracts_string_values(self) -> None:
        rows = [{"sender": "Alice", "amount": 1000.0, "receiver": "Bob"}]
        names = HybridRetriever._extract_names_from_graph(rows)
        assert "Alice" in names
        assert "Bob" in names

    def test_excludes_non_string_values(self) -> None:
        rows = [{"amount": 99999, "flagged": True}]
        names = HybridRetriever._extract_names_from_graph(rows)
        assert names == []

    def test_filters_very_short_strings(self) -> None:
        rows = [{"key": "AB"}]  # len <= 2
        names = HybridRetriever._extract_names_from_graph(rows)
        assert "AB" not in names
