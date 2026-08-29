"""
ChromaDB vector store backed by local Ollama embeddings (nomic-embed-text).

Two collections:
  intelligence_documents — chunked document content (DOC-001 … DOC-010)
  intelligence_comms     — communication bodies (evt_001 … evt_015)

Transactions are NOT embedded — they are structured and better queried
via Cypher in KuzuDB.

Each chunk carries metadata (doc_id / event_id, entity_names, date, channel)
so vector search results can be filtered by entity and anchored back to the
graph — this is the entity-anchoring step in hybrid retrieval.
"""

import json
import re
from pathlib import Path
from typing import Optional

import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings

from src.llm_client import OllamaClient

CHROMA_PATH   = Path(__file__).parents[2] / "data" / "chroma_db"
DOCS_COLL     = "intelligence_documents"
COMMS_COLL    = "intelligence_comms"
CHUNK_SIZE    = 600   # chars — our docs are short; one chunk per paragraph
EMBED_MODEL   = "nomic-embed-text"
EMBED_BATCH   = 16


# ---------------------------------------------------------------------------
# Embedding function (Ollama, OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------

class OllamaEmbedder(EmbeddingFunction):
    """Calls nomic-embed-text via Ollama's local REST API. No cloud calls."""

    def __init__(self, base_url: str = "http://localhost:11434/v1") -> None:
        self._client = OllamaClient(base_url=base_url)

    def __call__(self, input: Documents) -> Embeddings:
        embeddings: Embeddings = []
        for i in range(0, len(input), EMBED_BATCH):
            batch = input[i : i + EMBED_BATCH]
            resp = self._client.embeddings_create(model=EMBED_MODEL, input=batch)
            embeddings.extend([d.embedding for d in resp.data])
        return embeddings


# ---------------------------------------------------------------------------
# ChromaDB client
# ---------------------------------------------------------------------------

def open_client(chroma_path: Path = CHROMA_PATH) -> chromadb.ClientAPI:
    chroma_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(chroma_path))


def _get_or_create(client: chromadb.ClientAPI, name: str, embedder: EmbeddingFunction):
    return client.get_or_create_collection(name=name, embedding_function=embedder)


# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------

def _chunk_text(text: str, max_chars: int = CHUNK_SIZE) -> list[str]:
    """Split on double-newline (paragraph), merge short fragments."""
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) < max_chars:
            current = (current + " " + para).strip()
        else:
            if current:
                chunks.append(current)
            current = para
    if current:
        chunks.append(current)
    return chunks or [text[:max_chars]]


def _entity_names_in_text(text: str, known_entities: list[str]) -> list[str]:
    """Return which known entity names appear in the text (case-insensitive)."""
    text_lower = text.lower()
    return [e for e in known_entities if e.lower() in text_lower]


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def index_documents(
    client: chromadb.ClientAPI,
    docs: list[dict],
    known_entities: Optional[list[str]] = None,
    embedder: Optional[EmbeddingFunction] = None,
) -> int:
    """
    Chunk and embed documents into the intelligence_documents collection.
    Returns the number of chunks stored.
    """
    ef = embedder or OllamaEmbedder()
    coll = _get_or_create(client, DOCS_COLL, ef)
    known_entities = known_entities or []

    ids, texts, metas = [], [], []
    for doc in docs:
        chunks = _chunk_text(doc["content"])
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc['doc_id']}_chunk{i}"
            ids.append(chunk_id)
            texts.append(chunk)
            metas.append({
                "doc_id":       doc["doc_id"],
                "title":        doc["title"],
                "doc_type":     doc.get("type", ""),
                "date":         doc.get("date", ""),
                "entity_names": json.dumps(
                    _entity_names_in_text(chunk, known_entities)
                ),
            })

    if ids:
        coll.upsert(ids=ids, documents=texts, metadatas=metas)
    return len(ids)


def index_comms(
    client: chromadb.ClientAPI,
    comms: list[dict],
    known_entities: Optional[list[str]] = None,
    embedder: Optional[EmbeddingFunction] = None,
) -> int:
    """
    Embed communication bodies into the intelligence_comms collection.
    Returns the number of records stored.
    """
    ef = embedder or OllamaEmbedder()
    coll = _get_or_create(client, COMMS_COLL, ef)
    known_entities = known_entities or []

    ids, texts, metas = [], [], []
    for comm in comms:
        body = comm.get("body") or ""
        if not body.strip():
            continue
        ids.append(comm["event_id"])
        texts.append(body)
        metas.append({
            "event_id":      comm["event_id"],
            "sender_node":   comm["sender_node"],
            "channel":       comm["channel"],
            "timestamp":     comm["timestamp"],
            "intent_signal": comm.get("intent_signal") or "",
            "subject":       comm.get("subject") or "",
            "entity_names":  json.dumps(
                _entity_names_in_text(body, known_entities)
            ),
        })

    if ids:
        coll.upsert(ids=ids, documents=texts, metadatas=metas)
    return len(ids)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search(
    client: chromadb.ClientAPI,
    collection_name: str,
    query: str,
    n_results: int = 5,
    entity_filter: Optional[list[str]] = None,
    embedder: Optional[EmbeddingFunction] = None,
) -> list[dict]:
    """
    Vector similarity search. If entity_filter is provided, only chunks
    whose entity_names metadata overlaps with the filter are returned
    (entity-anchoring step in hybrid retrieval).

    Returns a list of dicts: {id, text, metadata, distance}.
    """
    ef = embedder or OllamaEmbedder()
    try:
        coll = client.get_collection(name=collection_name, embedding_function=ef)
    except Exception:
        return []

    where = None
    if entity_filter:
        # ChromaDB metadata filter: entity_names field contains any of the entities
        # We store entity_names as JSON string, so we use $contains on the serialised value
        if len(entity_filter) == 1:
            where = {"entity_names": {"$contains": entity_filter[0]}}
        else:
            where = {"$or": [{"entity_names": {"$contains": e}} for e in entity_filter]}

    kwargs = {"query_texts": [query], "n_results": min(n_results, coll.count() or 1)}
    if where:
        kwargs["where"] = where

    try:
        results = coll.query(**kwargs)
    except Exception:
        # Fall back without filter if collection is too small
        kwargs.pop("where", None)
        results = coll.query(**kwargs)

    output = []
    for i, doc_id in enumerate(results["ids"][0]):
        output.append({
            "id":       doc_id,
            "text":     results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })
    return output
