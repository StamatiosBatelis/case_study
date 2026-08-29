"""
Tests for the document ETL pipeline stage.

Storage tests run fully offline against a tmp SQLite DB.
ETL integration tests inject a deterministic mock engine so tests pass
without a running inference server or GPU.
"""

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.etl.etl_documents import LocalLLMExtractionEngine, run
from src.models import DocumentExtractionResult, ExtractedEntity, ExtractedRelation
from src.storage import entity_store


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_db(tmp_path: Path) -> Path:
    db = tmp_path / "test.db"
    entity_store.init_db(db)
    return db


@pytest.fixture()
def sample_result() -> DocumentExtractionResult:
    return DocumentExtractionResult(
        doc_id="DOC-001",
        entities=[
            ExtractedEntity(name="Elena Ross", entity_type="Person"),
            ExtractedEntity(name="Northstar Trading Ltd", entity_type="Company"),
            ExtractedEntity(name="ACC-4471", entity_type="Account"),
        ],
        relations=[
            ExtractedRelation(
                source_entity="Elena Ross",
                relation_type="BENEFICIAL_OWNER",
                target_entity="Northstar Trading Ltd",
                context="Beneficial owner: Elena Ross via a Liechtenstein trust",
            ),
            ExtractedRelation(
                source_entity="James Chen",
                relation_type="HOLDS_ACCOUNT",
                target_entity="ACC-4471",
                context=None,
            ),
        ],
    )


@pytest.fixture()
def docs_file(tmp_path: Path) -> Path:
    docs = [
        {
            "doc_id": "DOC-T01",
            "title": "Test Subject Profile",
            "type": "subject_profile",
            "date": "2024-03-01",
            "source": "test",
            "author": "Tester",
            "content": "Alice controls Company X through a nominee director Bob.",
        }
    ]
    p = tmp_path / "documents.json"
    p.write_text(json.dumps(docs))
    return p


def _mock_engine(result: DocumentExtractionResult) -> LocalLLMExtractionEngine:
    engine = MagicMock(spec=LocalLLMExtractionEngine)
    engine.extract.return_value = result
    return engine


# ---------------------------------------------------------------------------
# Storage layer tests
# ---------------------------------------------------------------------------

class TestEntityStore:
    def test_init_creates_tables(self, tmp_db: Path) -> None:
        conn = sqlite3.connect(tmp_db)
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        conn.close()
        assert "entities" in tables
        assert "relations" in tables

    def test_upsert_stores_entities(self, tmp_db: Path, sample_result: DocumentExtractionResult) -> None:
        entity_store.upsert_extraction(sample_result, db_path=tmp_db)
        rows = entity_store.all_entities(tmp_db)
        names = {r["name"] for r in rows}
        assert "Elena Ross" in names
        assert "Northstar Trading Ltd" in names
        assert "ACC-4471" in names

    def test_upsert_stores_relations(self, tmp_db: Path, sample_result: DocumentExtractionResult) -> None:
        entity_store.upsert_extraction(sample_result, db_path=tmp_db)
        rows = entity_store.all_relations(tmp_db)
        rel_types = {r["relation_type"] for r in rows}
        assert "BENEFICIAL_OWNER" in rel_types
        assert "HOLDS_ACCOUNT" in rel_types

    def test_upsert_is_idempotent(self, tmp_db: Path, sample_result: DocumentExtractionResult) -> None:
        entity_store.upsert_extraction(sample_result, db_path=tmp_db)
        entity_store.upsert_extraction(sample_result, db_path=tmp_db)
        rows = entity_store.all_entities(tmp_db)
        names = [r["name"] for r in rows]
        assert len(names) == len(set(names))

    def test_relations_for_entity(self, tmp_db: Path, sample_result: DocumentExtractionResult) -> None:
        entity_store.upsert_extraction(sample_result, db_path=tmp_db)
        rows = entity_store.relations_for_entity("Elena Ross", db_path=tmp_db)
        assert len(rows) == 1
        assert rows[0]["relation_type"] == "BENEFICIAL_OWNER"

    def test_entities_for_doc(self, tmp_db: Path, sample_result: DocumentExtractionResult) -> None:
        entity_store.upsert_extraction(sample_result, db_path=tmp_db)
        rows = entity_store.entities_for_doc("DOC-001", db_path=tmp_db)
        assert len(rows) == 3


# ---------------------------------------------------------------------------
# ETL stage tests (inference engine mocked)
# ---------------------------------------------------------------------------

class TestDocumentETL:
    def _fixed_result(self, doc_id: str) -> DocumentExtractionResult:
        return DocumentExtractionResult(
            doc_id=doc_id,
            entities=[ExtractedEntity(name="Alice", entity_type="Person")],
            relations=[
                ExtractedRelation(
                    source_entity="Alice",
                    relation_type="DIRECTOR_OF",
                    target_entity="Company X",
                    context="Alice controls Company X",
                )
            ],
        )

    def test_run_writes_to_db(self, docs_file: Path, tmp_db: Path) -> None:
        engine = _mock_engine(self._fixed_result("DOC-T01"))
        results = run(docs_path=docs_file, db_path=tmp_db, engine=engine)

        assert len(results) == 1
        assert results[0].doc_id == "DOC-T01"
        assert any(r["name"] == "Alice" for r in entity_store.all_entities(tmp_db))

    def test_run_dry_run_skips_db(self, docs_file: Path, tmp_db: Path) -> None:
        engine = _mock_engine(self._fixed_result("DOC-T01"))
        run(docs_path=docs_file, db_path=tmp_db, engine=engine, dry_run=True)

        assert len(entity_store.all_entities(tmp_db)) == 0

    def test_extraction_called_once_per_doc(self, docs_file: Path, tmp_db: Path) -> None:
        engine = _mock_engine(self._fixed_result("DOC-T01"))
        run(docs_path=docs_file, db_path=tmp_db, engine=engine)

        assert engine.extract.call_count == 1

    def test_missing_docs_file_raises(self, tmp_db: Path) -> None:
        engine = _mock_engine(self._fixed_result("X"))
        with pytest.raises(FileNotFoundError):
            run(docs_path=Path("/nonexistent/docs.json"), db_path=tmp_db, engine=engine)
