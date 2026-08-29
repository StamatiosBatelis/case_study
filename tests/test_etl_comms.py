"""Tests for the communications ETL stage."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.etl.etl_comms import LocalLLMCommsEngine, _sender_node, run
from src.resolution.alias_resolver import _local_to_name as _email_to_name
from src.models import CommsBodyExtraction
from src.storage import entity_store


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_RECORD = {
    "event_id": "evt_001",
    "timestamp": "2024-03-01T08:12:43Z",
    "channel": "email",
    "from": "marcus.vane@shell-corp.io",
    "to": ["elena.ross@privatenet.com"],
    "subject": "Re: shipment timing",
    "body": "Elena, the window is narrow. Move the funds before end of week.",
    "attachments": [],
    "metadata": {"ip": "185.220.101.34", "country": "RO"},
}

SAMPLE_EXTRACTION = CommsBodyExtraction(
    event_id="evt_001",
    entity_mentions=["Elena Ross", "Northstar Trading Ltd"],
    sender_alias="Marcus Vane",
    intent_signal="instruct_wire_transfer",
)


@pytest.fixture()
def tmp_db(tmp_path: Path) -> Path:
    db = tmp_path / "test.db"
    entity_store.init_db(db)
    return db


@pytest.fixture()
def comms_file(tmp_path: Path) -> Path:
    p = tmp_path / "comms_log.json"
    p.write_text(json.dumps([SAMPLE_RECORD]))
    return p


def _mock_engine(extraction: CommsBodyExtraction) -> LocalLLMCommsEngine:
    engine = MagicMock(spec=LocalLLMCommsEngine)
    engine.extract.return_value = extraction
    return engine


# ---------------------------------------------------------------------------
# Unit tests — alias resolution helpers
# ---------------------------------------------------------------------------

class TestAliasResolution:
    def test_email_to_name_dot_separated(self) -> None:
        assert _email_to_name("marcus.vane") == "Marcus Vane"

    def test_email_to_name_single_part(self) -> None:
        assert _email_to_name("elena") == "Elena"

    def test_sender_node_prefers_llm_alias(self) -> None:
        assert _sender_node("marcus.vane@shell-corp.io", "Marcus Vane") == "Marcus Vane"

    def test_sender_node_falls_back_to_heuristic(self) -> None:
        assert _sender_node("marcus.vane@shell-corp.io", None) == "Marcus Vane"

    def test_sender_node_raw_fallback(self) -> None:
        # Heuristic produces something, so raw fallback only applies to empty local parts
        result = _sender_node("x@y.com", None)
        assert result == "X"


# ---------------------------------------------------------------------------
# ETL stage tests (LLM mocked)
# ---------------------------------------------------------------------------

class TestCommsETL:
    def test_run_returns_enriched_records(self, comms_file: Path, tmp_db: Path) -> None:
        engine = _mock_engine(SAMPLE_EXTRACTION)
        results = run(comms_path=comms_file, db_path=tmp_db, engine=engine)
        assert len(results) == 1

    def test_run_writes_to_db(self, comms_file: Path, tmp_db: Path) -> None:
        engine = _mock_engine(SAMPLE_EXTRACTION)
        run(comms_path=comms_file, db_path=tmp_db, engine=engine)
        rows = entity_store.all_comms(tmp_db)
        assert len(rows) == 1

    def test_sender_node_resolved_in_db(self, comms_file: Path, tmp_db: Path) -> None:
        engine = _mock_engine(SAMPLE_EXTRACTION)
        run(comms_path=comms_file, db_path=tmp_db, engine=engine)
        row = entity_store.all_comms(tmp_db)[0]
        assert row["sender_node"] == "Marcus Vane"

    def test_intent_signal_stored(self, comms_file: Path, tmp_db: Path) -> None:
        engine = _mock_engine(SAMPLE_EXTRACTION)
        run(comms_path=comms_file, db_path=tmp_db, engine=engine)
        row = entity_store.all_comms(tmp_db)[0]
        assert row["intent_signal"] == "instruct_wire_transfer"

    def test_entity_mentions_stored_as_relations(self, comms_file: Path, tmp_db: Path) -> None:
        engine = _mock_engine(SAMPLE_EXTRACTION)
        run(comms_path=comms_file, db_path=tmp_db, engine=engine)
        rels = entity_store.all_relations(tmp_db)
        rel_types = {r["relation_type"] for r in rels}
        assert "MENTIONED_IN_COMM" in rel_types

    def test_run_is_idempotent(self, comms_file: Path, tmp_db: Path) -> None:
        engine = _mock_engine(SAMPLE_EXTRACTION)
        run(comms_path=comms_file, db_path=tmp_db, engine=engine)
        run(comms_path=comms_file, db_path=tmp_db, engine=engine)
        assert len(entity_store.all_comms(tmp_db)) == 1

    def test_dry_run_skips_db(self, comms_file: Path, tmp_db: Path) -> None:
        engine = _mock_engine(SAMPLE_EXTRACTION)
        run(comms_path=comms_file, db_path=tmp_db, engine=engine, dry_run=True)
        assert len(entity_store.all_comms(tmp_db)) == 0

    def test_missing_file_raises(self, tmp_db: Path) -> None:
        engine = _mock_engine(SAMPLE_EXTRACTION)
        with pytest.raises(FileNotFoundError):
            run(comms_path=Path("/nonexistent.json"), db_path=tmp_db, engine=engine)
