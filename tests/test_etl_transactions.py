"""Tests for the deterministic transaction ETL stage."""

import json
from pathlib import Path

import pytest

from src.etl.etl_transactions import _canonical_node, _parse, run
from src.storage import entity_store


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_TX = {
    "txn_id": "TXN-T01",
    "timestamp": "2024-03-03T08:30:00Z",
    "type": "wire_transfer",
    "amount": 48500.00,
    "currency": "USD",
    "sender": {"account_id": "ACC-4471", "name": "James Chen", "bank": "ClearBank", "iban": None, "country": "GB"},
    "receiver": {"account_id": None, "name": "Northstar Trading Ltd", "bank": "NatWest", "iban": "GB29NWBK", "country": "GB"},
    "reference": "Consulting Q1",
    "flagged": False,
}

FLAGGED_TX = {**SAMPLE_TX, "txn_id": "TXN-T02", "flagged": True}


@pytest.fixture()
def tmp_db(tmp_path: Path) -> Path:
    db = tmp_path / "test.db"
    entity_store.init_db(db)
    return db


@pytest.fixture()
def txn_file(tmp_path: Path) -> Path:
    p = tmp_path / "transactions.json"
    p.write_text(json.dumps([SAMPLE_TX, FLAGGED_TX]))
    return p


# ---------------------------------------------------------------------------
# Unit tests — parsing logic
# ---------------------------------------------------------------------------

class TestCanonicalNode:
    def test_prefers_account_id(self) -> None:
        assert _canonical_node({"account_id": "ACC-4471", "name": "James Chen"}) == "ACC-4471"

    def test_falls_back_to_name(self) -> None:
        assert _canonical_node({"account_id": None, "name": "Northstar Trading Ltd"}) == "Northstar Trading Ltd"

    def test_missing_account_id_key(self) -> None:
        assert _canonical_node({"name": "Elena Ross"}) == "Elena Ross"


class TestParse:
    def test_sender_node_uses_account_id(self) -> None:
        parsed = _parse(SAMPLE_TX)
        assert parsed["sender_node"] == "ACC-4471"

    def test_receiver_node_falls_back_to_name(self) -> None:
        parsed = _parse(SAMPLE_TX)
        assert parsed["receiver_node"] == "Northstar Trading Ltd"

    def test_amount_and_currency_preserved(self) -> None:
        parsed = _parse(SAMPLE_TX)
        assert parsed["amount"] == 48500.00
        assert parsed["currency"] == "USD"

    def test_flagged_coerced_to_bool(self) -> None:
        assert _parse(SAMPLE_TX)["flagged"] is False
        assert _parse(FLAGGED_TX)["flagged"] is True


# ---------------------------------------------------------------------------
# ETL stage tests
# ---------------------------------------------------------------------------

class TestTransactionETL:
    def test_run_returns_all_records(self, txn_file: Path, tmp_db: Path) -> None:
        results = run(transactions_path=txn_file, db_path=tmp_db)
        assert len(results) == 2

    def test_run_writes_to_db(self, txn_file: Path, tmp_db: Path) -> None:
        run(transactions_path=txn_file, db_path=tmp_db)
        rows = entity_store.all_transactions(tmp_db)
        assert len(rows) == 2

    def test_run_is_idempotent(self, txn_file: Path, tmp_db: Path) -> None:
        run(transactions_path=txn_file, db_path=tmp_db)
        run(transactions_path=txn_file, db_path=tmp_db)
        assert len(entity_store.all_transactions(tmp_db)) == 2

    def test_dry_run_skips_db(self, txn_file: Path, tmp_db: Path) -> None:
        run(transactions_path=txn_file, db_path=tmp_db, dry_run=True)
        assert len(entity_store.all_transactions(tmp_db)) == 0

    def test_missing_file_raises(self, tmp_db: Path) -> None:
        with pytest.raises(FileNotFoundError):
            run(transactions_path=Path("/nonexistent.json"), db_path=tmp_db)

    def test_transactions_for_node(self, txn_file: Path, tmp_db: Path) -> None:
        run(transactions_path=txn_file, db_path=tmp_db)
        rows = entity_store.transactions_for_node("ACC-4471", db_path=tmp_db)
        assert len(rows) == 2


