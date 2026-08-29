"""
SQLite-backed persistence layer for the intelligence knowledge graph.

Three tables:
  entities     — typed nodes extracted from documents (LLM output)
  relations    — typed edges extracted from documents (LLM output)
  transactions — financial flows parsed deterministically from transactions.json

Separating storage from graph construction means the NetworkX graph can be
rebuilt at any time from the DB without re-calling the LLM.
"""

import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List

from src.models import DocumentExtractionResult


DB_PATH = Path(__file__).parents[2] / "data" / "knowledge_graph.db"


def _schema_sql() -> str:
    return """
    CREATE TABLE IF NOT EXISTS entities (
        id          TEXT PRIMARY KEY,
        name        TEXT NOT NULL,
        entity_type TEXT NOT NULL,
        source_doc  TEXT,
        UNIQUE(name, source_doc)
    );

    CREATE TABLE IF NOT EXISTS relations (
        id            TEXT PRIMARY KEY,
        source_entity TEXT NOT NULL,
        relation_type TEXT NOT NULL,
        target_entity TEXT NOT NULL,
        context       TEXT,
        source_doc    TEXT NOT NULL,
        UNIQUE(source_entity, relation_type, target_entity, source_doc)
    );

    CREATE TABLE IF NOT EXISTS transactions (
        txn_id          TEXT PRIMARY KEY,
        timestamp       TEXT NOT NULL,
        txn_type        TEXT,
        amount          REAL NOT NULL,
        currency        TEXT NOT NULL,
        sender_node     TEXT NOT NULL,
        sender_name     TEXT NOT NULL,
        sender_country  TEXT,
        receiver_node   TEXT NOT NULL,
        receiver_name   TEXT NOT NULL,
        receiver_country TEXT,
        reference       TEXT,
        flagged         INTEGER NOT NULL DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS comms (
        event_id       TEXT PRIMARY KEY,
        timestamp      TEXT NOT NULL,
        channel        TEXT NOT NULL,
        sender_raw     TEXT NOT NULL,
        sender_node    TEXT NOT NULL,
        recipients     TEXT NOT NULL,
        subject        TEXT,
        body           TEXT,
        attachments    TEXT,
        ip_address     TEXT,
        country        TEXT,
        intent_signal  TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_relations_source  ON relations(source_entity);
    CREATE INDEX IF NOT EXISTS idx_relations_target  ON relations(target_entity);
    CREATE INDEX IF NOT EXISTS idx_entities_name     ON entities(name);
    CREATE INDEX IF NOT EXISTS idx_txn_sender        ON transactions(sender_node);
    CREATE INDEX IF NOT EXISTS idx_txn_receiver      ON transactions(receiver_node);
    CREATE INDEX IF NOT EXISTS idx_comms_sender      ON comms(sender_node);
    CREATE INDEX IF NOT EXISTS idx_comms_timestamp   ON comms(timestamp);
    """


@contextmanager
def _connect(db_path: Path = DB_PATH) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Path = DB_PATH) -> None:
    """Create tables if they don't already exist."""
    with _connect(db_path) as conn:
        conn.executescript(_schema_sql())


def upsert_extraction(result: DocumentExtractionResult, db_path: Path = DB_PATH) -> None:
    """
    Persist entities and relations for one document.
    Uses INSERT OR IGNORE so re-running ETL on the same doc is a no-op
    for entities, and appends new relations without duplication.
    """
    with _connect(db_path) as conn:
        for ent in result.entities:
            conn.execute(
                "INSERT OR IGNORE INTO entities (id, name, entity_type, source_doc) VALUES (?, ?, ?, ?)",
                (str(uuid.uuid4()), ent.name.strip(), ent.entity_type, result.doc_id),
            )

        for rel in result.relations:
            conn.execute(
                """
                INSERT OR IGNORE INTO relations (id, source_entity, relation_type, target_entity, context, source_doc)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    rel.source_entity.strip(),
                    rel.relation_type,
                    rel.target_entity.strip(),
                    rel.context,
                    result.doc_id,
                ),
            )


def all_entities(db_path: Path = DB_PATH) -> List[sqlite3.Row]:
    with _connect(db_path) as conn:
        return conn.execute("SELECT DISTINCT name, entity_type FROM entities").fetchall()


def all_relations(db_path: Path = DB_PATH) -> List[sqlite3.Row]:
    with _connect(db_path) as conn:
        return conn.execute(
            "SELECT source_entity, relation_type, target_entity, context, source_doc FROM relations"
        ).fetchall()


def entities_for_doc(doc_id: str, db_path: Path = DB_PATH) -> List[sqlite3.Row]:
    with _connect(db_path) as conn:
        return conn.execute(
            "SELECT name, entity_type FROM entities WHERE source_doc = ?", (doc_id,)
        ).fetchall()


def relations_for_entity(name: str, db_path: Path = DB_PATH) -> List[sqlite3.Row]:
    with _connect(db_path) as conn:
        return conn.execute(
            """
            SELECT source_entity, relation_type, target_entity, context, source_doc
            FROM   relations
            WHERE  source_entity = ? OR target_entity = ?
            """,
            (name, name),
        ).fetchall()


def upsert_transaction(tx: dict, db_path: Path = DB_PATH) -> None:
    """
    Persist a single parsed transaction record.
    sender_node / receiver_node are the canonical graph identifiers:
      account_id when present, entity name otherwise.
    INSERT OR IGNORE makes re-runs idempotent on txn_id.
    """
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO transactions (
                txn_id, timestamp, txn_type, amount, currency,
                sender_node, sender_name, sender_country,
                receiver_node, receiver_name, receiver_country,
                reference, flagged
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                tx["txn_id"], tx["timestamp"], tx["txn_type"],
                tx["amount"], tx["currency"],
                tx["sender_node"], tx["sender_name"], tx.get("sender_country"),
                tx["receiver_node"], tx["receiver_name"], tx.get("receiver_country"),
                tx.get("reference"), int(tx.get("flagged", False)),
            ),
        )


def all_transactions(db_path: Path = DB_PATH) -> List[sqlite3.Row]:
    with _connect(db_path) as conn:
        return conn.execute("SELECT * FROM transactions ORDER BY timestamp").fetchall()


def transactions_for_node(node: str, db_path: Path = DB_PATH) -> List[sqlite3.Row]:
    with _connect(db_path) as conn:
        return conn.execute(
            "SELECT * FROM transactions WHERE sender_node = ? OR receiver_node = ?",
            (node, node),
        ).fetchall()


def upsert_comm(record: dict, db_path: Path = DB_PATH) -> None:
    """
    Persist a single communication record.
    recipients and attachments are stored as JSON strings.
    INSERT OR IGNORE on event_id makes re-runs idempotent.
    """
    import json as _json
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO comms (
                event_id, timestamp, channel,
                sender_raw, sender_node, recipients,
                subject, body, attachments,
                ip_address, country, intent_signal
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["event_id"], record["timestamp"], record["channel"],
                record["sender_raw"], record["sender_node"],
                _json.dumps(record.get("recipients", [])),
                record.get("subject"), record.get("body"),
                _json.dumps(record.get("attachments", [])),
                record.get("ip_address"), record.get("country"),
                record.get("intent_signal"),
            ),
        )


def all_comms(db_path: Path = DB_PATH) -> List[sqlite3.Row]:
    with _connect(db_path) as conn:
        return conn.execute("SELECT * FROM comms ORDER BY timestamp").fetchall()


def comms_for_node(node: str, db_path: Path = DB_PATH) -> List[sqlite3.Row]:
    with _connect(db_path) as conn:
        return conn.execute(
            "SELECT * FROM comms WHERE sender_node = ? OR recipients LIKE ?",
            (node, f'%"{node}"%'),
        ).fetchall()
