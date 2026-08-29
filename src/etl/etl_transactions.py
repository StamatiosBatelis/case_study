"""
ETL step: transactions.json → deterministic parsing → SQLite

No LLM involved. All fields are structured and typed — the data is trusted as-is.
The only non-trivial logic is canonical node resolution: we prefer account_id over
entity name so graph nodes align with what document extraction already produced.

Pipeline stage — import and call run() from src/pipeline.py.
"""

import json
from pathlib import Path
from typing import Optional

from rich.console import Console

from src.storage import entity_store

TRANSACTIONS_PATH = Path(__file__).parents[2] / "data" / "transactions.json"

console = Console()


def _canonical_node(party: dict) -> str:
    """
    Resolve the graph node identifier for a transaction party.

    Prefer account_id (e.g. ACC-4471) because document extraction already
    created nodes with those identifiers.  Fall back to entity name so that
    parties without registered accounts still appear as nodes.
    """
    return party.get("account_id") or party["name"]


def _parse(raw: dict) -> dict:
    """Map a raw transaction record to the flat shape expected by the DB."""
    return {
        "txn_id":           raw["txn_id"],
        "timestamp":        raw["timestamp"],
        "txn_type":         raw.get("type"),
        "amount":           raw["amount"],
        "currency":         raw["currency"],
        "sender_node":      _canonical_node(raw["sender"]),
        "sender_name":      raw["sender"]["name"],
        "sender_country":   raw["sender"].get("country"),
        "receiver_node":    _canonical_node(raw["receiver"]),
        "receiver_name":    raw["receiver"]["name"],
        "receiver_country": raw["receiver"].get("country"),
        "reference":        raw.get("reference"),
        "flagged":          raw.get("flagged", False),
    }


def run(
    transactions_path: Path = TRANSACTIONS_PATH,
    db_path: Optional[Path] = None,
    dry_run: bool = False,
) -> list[dict]:
    """
    Parse transactions.json and persist each record to SQLite.

    Returns the list of parsed transaction dicts (used in tests and by the
    graph builder when called in-process).
    """
    if not transactions_path.exists():
        raise FileNotFoundError(f"Transactions file not found: {transactions_path}")

    db = db_path or entity_store.DB_PATH

    if not dry_run:
        entity_store.init_db(db)

    with open(transactions_path) as f:
        raw_txns = json.load(f)

    parsed = [_parse(tx) for tx in raw_txns]

    flagged = sum(1 for t in parsed if t["flagged"])
    console.print(
        f"  [transactions] {len(parsed)} records parsed, {flagged} flagged"
    )

    if not dry_run:
        for tx in parsed:
            entity_store.upsert_transaction(tx, db_path=db)

    return parsed
