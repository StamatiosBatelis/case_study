"""Tests for KuzuDB store — schema, sync, and Cypher queries."""

from pathlib import Path
from unittest.mock import patch

import kuzu
import pytest

from src.storage import entity_store, kuzu_store


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_sqlite(tmp_path: Path) -> Path:
    db = tmp_path / "test.db"
    entity_store.init_db(db)
    # Populate with minimal data
    from src.models import DocumentExtractionResult, ExtractedEntity, ExtractedRelation
    entity_store.upsert_extraction(
        DocumentExtractionResult(
            doc_id="DOC-T01",
            entities=[
                ExtractedEntity(name="Alice", entity_type="Person"),
                ExtractedEntity(name="Acme Corp", entity_type="Company"),
            ],
            relations=[
                ExtractedRelation(
                    source_entity="Alice",
                    relation_type="DIRECTOR_OF",
                    target_entity="Acme Corp",
                    context="Alice is director",
                )
            ],
        ),
        db_path=db,
    )
    entity_store.upsert_transaction(
        {
            "txn_id": "TXN-T01", "timestamp": "2024-03-03T08:00:00Z",
            "txn_type": "wire_transfer", "amount": 50000.0, "currency": "USD",
            "sender_node": "Alice", "sender_name": "Alice", "sender_country": "GB",
            "receiver_node": "Acme Corp", "receiver_name": "Acme Corp", "receiver_country": "GB",
            "reference": "Test", "flagged": False,
        },
        db_path=db,
    )
    return db


@pytest.fixture()
def tmp_kuzu(tmp_path: Path) -> Path:
    return tmp_path / "kuzu_db"


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestSchema:
    def test_init_creates_node_table(self, tmp_kuzu: Path) -> None:
        db = kuzu_store.open_db(tmp_kuzu)
        kuzu_store.init_schema(db)
        conn = kuzu.Connection(db)
        # Should not raise
        conn.execute("MATCH (e:Entity) RETURN count(e)")

    def test_init_creates_relation_tables(self, tmp_kuzu: Path) -> None:
        db = kuzu_store.open_db(tmp_kuzu)
        kuzu_store.init_schema(db)
        conn = kuzu.Connection(db)
        for rel in ("RELATION", "FINANCIAL_FLOW", "COMMUNICATION"):
            result = conn.execute(f"MATCH ()-[r:{rel}]->() RETURN count(r)")
            assert result is not None

    def test_init_is_idempotent(self, tmp_kuzu: Path) -> None:
        db = kuzu_store.open_db(tmp_kuzu)
        kuzu_store.init_schema(db)
        kuzu_store.init_schema(db)  # should not raise


# ---------------------------------------------------------------------------
# Sync tests
# ---------------------------------------------------------------------------

class TestSync:
    def test_sync_returns_counts(self, tmp_kuzu: Path, tmp_sqlite: Path) -> None:
        counts = kuzu_store.sync(tmp_kuzu, tmp_sqlite)
        assert counts["nodes"] >= 2
        assert counts["relations"] >= 1
        assert counts["financial_flows"] >= 1

    def test_entity_nodes_exist_after_sync(self, tmp_kuzu: Path, tmp_sqlite: Path) -> None:
        kuzu_store.sync(tmp_kuzu, tmp_sqlite)
        db = kuzu_store.open_db(tmp_kuzu)
        conn = kuzu.Connection(db)
        rows = kuzu_store.cypher(conn, "MATCH (e:Entity {name: 'Alice'}) RETURN e.name AS name")
        assert any(r.get("name") == "Alice" for r in rows)

    def test_relation_edges_exist_after_sync(self, tmp_kuzu: Path, tmp_sqlite: Path) -> None:
        kuzu_store.sync(tmp_kuzu, tmp_sqlite)
        db = kuzu_store.open_db(tmp_kuzu)
        conn = kuzu.Connection(db)
        rows = kuzu_store.cypher(
            conn,
            "MATCH (a:Entity)-[r:RELATION]->(b:Entity) "
            "RETURN r.relation_type AS rel",
        )
        assert any(r.get("rel") == "DIRECTOR_OF" for r in rows)

    def test_financial_flow_edges_exist_after_sync(self, tmp_kuzu: Path, tmp_sqlite: Path) -> None:
        kuzu_store.sync(tmp_kuzu, tmp_sqlite)
        db = kuzu_store.open_db(tmp_kuzu)
        conn = kuzu.Connection(db)
        rows = kuzu_store.cypher(
            conn,
            "MATCH (a:Entity)-[r:FINANCIAL_FLOW]->(b:Entity) "
            "RETURN r.amount AS amount",
        )
        assert any(r.get("amount") == 50000.0 for r in rows)


# ---------------------------------------------------------------------------
# Query helper tests
# ---------------------------------------------------------------------------

class TestQueryHelpers:
    def test_all_entity_names(self, tmp_kuzu: Path, tmp_sqlite: Path) -> None:
        kuzu_store.sync(tmp_kuzu, tmp_sqlite)
        db = kuzu_store.open_db(tmp_kuzu)
        conn = kuzu.Connection(db)
        names = kuzu_store.all_entity_names(conn)
        assert "Alice" in names
        assert "Acme Corp" in names

    def test_entity_subgraph(self, tmp_kuzu: Path, tmp_sqlite: Path) -> None:
        kuzu_store.sync(tmp_kuzu, tmp_sqlite)
        db = kuzu_store.open_db(tmp_kuzu)
        conn = kuzu.Connection(db)
        results = kuzu_store.entity_subgraph(conn, "Alice", hops=2)
        assert isinstance(results, list)
