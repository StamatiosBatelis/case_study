"""Tests for ChromaDB vector store — chunking, indexing, and search."""

import json
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import chromadb
import pytest

from chromadb import EmbeddingFunction, Documents, Embeddings
from src.storage import vector_store
from src.storage.vector_store import (
    DOCS_COLL,
    COMMS_COLL,
    _chunk_text,
    _entity_names_in_text,
    index_documents,
    index_comms,
    search,
    open_client,
)


# ---------------------------------------------------------------------------
# Stub embedder — extends EmbeddingFunction to get embed_query for free
# ---------------------------------------------------------------------------

class StubEmbedder(EmbeddingFunction):
    DIM = 8

    def __call__(self, input: Documents) -> Embeddings:
        return [[float(abs(hash(t)) % 1000) / 1000.0] * self.DIM for t in input]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

STUB_DOCS = [
    {
        "doc_id": "DOC-T01",
        "title":  "Test Document",
        "type":   "report",
        "date":   "2024-01-01",
        "content": "Marcus Vane transferred funds to Northstar Trading.\n\nThe transaction was flagged.",
    },
    {
        "doc_id": "DOC-T02",
        "title":  "Second Document",
        "type":   "report",
        "date":   "2024-02-01",
        "content": "Elena Ross communicated with external parties.",
    },
]

STUB_COMMS = [
    {
        "event_id":     "evt_T01",
        "sender_node":  "Marcus Vane",
        "channel":      "email",
        "timestamp":    "2024-03-01T10:00:00Z",
        "body":         "Let us proceed with the Rotterdam transfer as discussed.",
        "intent_signal": "coordination",
        "subject":      "Transfer",
    },
    {
        "event_id":     "evt_T02",
        "sender_node":  "Elena Ross",
        "channel":      "chat",
        "timestamp":    "2024-03-02T11:00:00Z",
        "body":         "Understood. I will handle the documentation.",
        "intent_signal": "",
        "subject":      "",
    },
]


@pytest.fixture()
def chroma(tmp_path: Path) -> chromadb.ClientAPI:
    return open_client(tmp_path / "chroma_test")


# ---------------------------------------------------------------------------
# _chunk_text
# ---------------------------------------------------------------------------

class TestChunkText:
    def test_single_short_paragraph_returned_as_one_chunk(self) -> None:
        text = "Hello world."
        assert _chunk_text(text) == ["Hello world."]

    def test_two_paragraphs_within_limit_merged(self) -> None:
        text = "Para one.\n\nPara two."
        result = _chunk_text(text, max_chars=200)
        assert len(result) == 1
        assert "Para one." in result[0]
        assert "Para two." in result[0]

    def test_large_paragraphs_split(self) -> None:
        p1 = "A" * 300
        p2 = "B" * 300
        text = f"{p1}\n\n{p2}"
        result = _chunk_text(text, max_chars=400)
        assert len(result) == 2

    def test_empty_string_returns_one_chunk(self) -> None:
        result = _chunk_text("", max_chars=600)
        # text[:max_chars] on empty string is ""
        assert result == [""] or result == []


# ---------------------------------------------------------------------------
# _entity_names_in_text
# ---------------------------------------------------------------------------

class TestEntityNamesInText:
    def test_matching_entity_returned(self) -> None:
        result = _entity_names_in_text("Marcus Vane sent the funds.", ["Marcus Vane", "Elena Ross"])
        assert "Marcus Vane" in result

    def test_non_matching_entity_excluded(self) -> None:
        result = _entity_names_in_text("Marcus Vane sent the funds.", ["Marcus Vane", "Elena Ross"])
        assert "Elena Ross" not in result

    def test_case_insensitive_match(self) -> None:
        result = _entity_names_in_text("marcus vane was involved.", ["Marcus Vane"])
        assert "Marcus Vane" in result

    def test_empty_entity_list(self) -> None:
        assert _entity_names_in_text("some text", []) == []


# ---------------------------------------------------------------------------
# index_documents
# ---------------------------------------------------------------------------

class TestIndexDocuments:
    def test_returns_chunk_count(self, chroma: chromadb.ClientAPI) -> None:
        n = index_documents(chroma, STUB_DOCS, embedder=StubEmbedder())
        assert n >= len(STUB_DOCS)

    def test_documents_stored_in_collection(self, chroma: chromadb.ClientAPI) -> None:
        index_documents(chroma, STUB_DOCS, embedder=StubEmbedder())
        coll = chroma.get_collection(DOCS_COLL, embedding_function=StubEmbedder())
        assert coll.count() >= len(STUB_DOCS)

    def test_entity_names_stored_as_metadata(self, chroma: chromadb.ClientAPI) -> None:
        index_documents(
            chroma, STUB_DOCS,
            known_entities=["Marcus Vane"],
            embedder=StubEmbedder(),
        )
        coll = chroma.get_collection(DOCS_COLL, embedding_function=StubEmbedder())
        results = coll.get(ids=["DOC-T01_chunk0"], include=["metadatas"])
        meta = results["metadatas"][0]
        entities = json.loads(meta["entity_names"])
        assert "Marcus Vane" in entities

    def test_upsert_is_idempotent(self, chroma: chromadb.ClientAPI) -> None:
        n1 = index_documents(chroma, STUB_DOCS, embedder=StubEmbedder())
        n2 = index_documents(chroma, STUB_DOCS, embedder=StubEmbedder())
        assert n1 == n2
        coll = chroma.get_collection(DOCS_COLL, embedding_function=StubEmbedder())
        assert coll.count() == n1


# ---------------------------------------------------------------------------
# index_comms
# ---------------------------------------------------------------------------

class TestIndexComms:
    def test_returns_comm_count(self, chroma: chromadb.ClientAPI) -> None:
        n = index_comms(chroma, STUB_COMMS, embedder=StubEmbedder())
        assert n == len(STUB_COMMS)

    def test_comms_stored_in_collection(self, chroma: chromadb.ClientAPI) -> None:
        index_comms(chroma, STUB_COMMS, embedder=StubEmbedder())
        coll = chroma.get_collection(COMMS_COLL, embedding_function=StubEmbedder())
        assert coll.count() == len(STUB_COMMS)

    def test_empty_body_skipped(self, chroma: chromadb.ClientAPI) -> None:
        comms_with_empty = STUB_COMMS + [
            {"event_id": "evt_empty", "sender_node": "X", "channel": "sms",
             "timestamp": "2024-01-01T00:00:00Z", "body": ""}
        ]
        n = index_comms(chroma, comms_with_empty, embedder=StubEmbedder())
        assert n == len(STUB_COMMS)


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

class TestSearch:
    def test_search_returns_results(self, chroma: chromadb.ClientAPI) -> None:
        index_documents(chroma, STUB_DOCS, embedder=StubEmbedder())
        results = search(chroma, DOCS_COLL, "funds transfer", n_results=2, embedder=StubEmbedder())
        assert isinstance(results, list)
        assert len(results) > 0

    def test_search_result_has_expected_keys(self, chroma: chromadb.ClientAPI) -> None:
        index_documents(chroma, STUB_DOCS, embedder=StubEmbedder())
        results = search(chroma, DOCS_COLL, "flagged transaction", embedder=StubEmbedder())
        for r in results:
            assert "id" in r
            assert "text" in r
            assert "metadata" in r
            assert "distance" in r

    def test_search_missing_collection_returns_empty(self, chroma: chromadb.ClientAPI) -> None:
        results = search(chroma, "nonexistent_collection", "query", embedder=StubEmbedder())
        assert results == []

    def test_entity_filter_applied(self, chroma: chromadb.ClientAPI) -> None:
        index_documents(
            chroma, STUB_DOCS,
            known_entities=["Marcus Vane"],
            embedder=StubEmbedder(),
        )
        results = search(
            chroma, DOCS_COLL, "transfer",
            entity_filter=["Marcus Vane"],
            embedder=StubEmbedder(),
        )
        # All returned chunks should mention Marcus Vane in entity_names
        for r in results:
            entities = json.loads(r["metadata"].get("entity_names", "[]"))
            assert "Marcus Vane" in entities
