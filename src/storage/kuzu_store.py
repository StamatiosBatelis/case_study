"""
KuzuDB graph store.

Replaces NetworkX as the graph query engine. KuzuDB is an embedded,
Cypher-native graph database — no server, pip-installable, air-gapped safe.

Responsibilities:
  - Define the property graph schema (Entity nodes, typed relationship tables)
  - Sync all data from SQLite (entities, relations, transactions, comms)
    applying alias resolution so the graph is deduplicated from the start
  - Expose Cypher query helpers used by the HybridRetriever

Node table   : Entity  (single table, entity_type as property — avoids schema
                        explosion from LLM-generated type strings)
Rel tables   : RELATION        (document-extracted typed edges)
               FINANCIAL_FLOW  (transaction money flows)
               COMMUNICATION   (sender → recipient with intent signal)
"""

import json
import re
from pathlib import Path
from typing import Any, Optional

import kuzu

from src.resolution.alias_resolver import AliasResolver
from src.storage import entity_store

KUZU_PATH = Path(__file__).parents[2] / "data" / "kuzu_db"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = [
    """CREATE NODE TABLE IF NOT EXISTS Entity(
        name        STRING,
        entity_type STRING,
        layer       STRING,
        PRIMARY KEY (name)
    )""",
    """CREATE REL TABLE IF NOT EXISTS RELATION(
        FROM Entity TO Entity,
        relation_type STRING,
        context       STRING,
        source_doc    STRING
    )""",
    """CREATE REL TABLE IF NOT EXISTS FINANCIAL_FLOW(
        FROM Entity TO Entity,
        txn_id    STRING,
        amount    DOUBLE,
        currency  STRING,
        timestamp STRING,
        flagged   BOOLEAN,
        reference STRING
    )""",
    """CREATE REL TABLE IF NOT EXISTS COMMUNICATION(
        FROM Entity TO Entity,
        event_id      STRING,
        channel       STRING,
        timestamp     STRING,
        intent_signal STRING,
        subject       STRING
    )""",
]


_BLOCKED_TYPES: frozenset[str] = frozenset({
    "date", "phone_number", "company_registration_number",
    "legislation", "source", "operation",
})

_NOISE_PATTERNS: list[re.Pattern] = [
    re.compile(r"^\d{4}$"),                            # bare years: 2021
    re.compile(r"^\d{4}-\d{2}-\d{2}"),                 # ISO dates: 2024-02-20
    re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$"),          # IP addresses
    re.compile(r"^TXN-", re.I),                         # transaction IDs
    re.compile(r"^[\w-]+\.[\w]{2,}$"),                  # bare domain names: freight-anon.com
    re.compile(r"^[A-Z]{2,4}-[\d.]+$"),                # registration codes: CHE-123.456
    re.compile(r"^[A-Z]{2,4}-\d{4}"),                  # reference codes: ROT-2024-0312
    re.compile(r".+\s\d+\.\d+$"),                       # software versions: LibreOffice 7.4
    re.compile(r"^\+?[\d\s\-\(\)]{8,}$"),              # phone numbers: +447700900112
    re.compile(r"^\d+\s+\w.+,"),                        # street addresses: 12 Canary Wharf, London
    re.compile(r"@"),                                   # raw email addresses
    re.compile(r"^[A-Z]{3}\s+[\d,]+"),                 # monetary amounts: EUR 22,000
]


def _is_noise_entity(name: str, entity_type: str) -> bool:
    if entity_type.lower() in _BLOCKED_TYPES:
        return True
    if entity_type.lower() == "location" and len(name) <= 3:
        return True
    for pattern in _NOISE_PATTERNS:
        if pattern.search(name):
            return True
    return False


def open_db(db_path: Path = KUZU_PATH) -> kuzu.Database:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return kuzu.Database(str(db_path))


def init_schema(db: kuzu.Database) -> None:
    conn = kuzu.Connection(db)
    for stmt in _SCHEMA:
        conn.execute(stmt)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _merge_entity(conn: kuzu.Connection, name: str, entity_type: str, layer: str) -> None:
    conn.execute(
        "MERGE (e:Entity {name: $name}) "
        "ON MATCH SET e.entity_type = $etype, e.layer = $layer "
        "ON CREATE SET e.entity_type = $etype, e.layer = $layer",
        {"name": name, "etype": entity_type, "layer": layer},
    )


def _create_rel(conn: kuzu.Connection, src: str, tgt: str, rel_table: str, props: dict) -> None:
    prop_keys = ", ".join(f"{k}: ${k}" for k in props)
    conn.execute(
        f"MATCH (a:Entity {{name: $src}}), (b:Entity {{name: $tgt}}) "
        f"CREATE (a)-[:{rel_table} {{{prop_keys}}}]->(b)",
        {"src": src, "tgt": tgt, **props},
    )


def _rows_to_dicts(result: kuzu.QueryResult) -> list[dict]:
    cols = result.get_column_names()
    rows = []
    while result.has_next():
        rows.append(dict(zip(cols, result.get_next())))
    return rows


# ---------------------------------------------------------------------------
# Sync from SQLite
# ---------------------------------------------------------------------------

def sync(db_path: Path = KUZU_PATH, sqlite_path: Path = entity_store.DB_PATH) -> dict:
    """
    Load all data from SQLite into KuzuDB, applying alias resolution for
    communication recipients so the graph is deduplicated from the start.

    Returns a summary dict {nodes, relations, financial_flows, communications}.
    """
    db = open_db(db_path)
    init_schema(db)
    conn = kuzu.Connection(db)
    resolver = AliasResolver(sqlite_path)
    counts = {"nodes": 0, "relations": 0, "financial_flows": 0, "communications": 0}

    # 1. Entity nodes — skip noise: bare dates, years, IPs, phone numbers,
    # registration codes, software versions, and short country codes.
    for row in entity_store.all_entities(sqlite_path):
        if _is_noise_entity(row["name"], row["entity_type"]):
            continue
        _merge_entity(conn, row["name"], row["entity_type"], "document")
        counts["nodes"] += 1

    # 2. Document relation edges — ensure both endpoints exist as nodes.
    # Skip relations where either endpoint is noise (dates, IPs, phone numbers, etc.)
    for row in entity_store.all_relations(sqlite_path):
        src, tgt = row["source_entity"], row["target_entity"]
        if _is_noise_entity(src, "Unknown") or _is_noise_entity(tgt, "Unknown"):
            continue
        for n in (src, tgt):
            _merge_entity(conn, n, "Unknown", "document")
        try:
            _create_rel(conn, src, tgt, "RELATION", {
                "relation_type": row["relation_type"],
                "context":       row["context"] or "",
                "source_doc":    row["source_doc"],
            })
            counts["relations"] += 1
        except Exception:
            pass  # duplicate edge — skip

    # 3. Financial flow edges + account-to-company cross-links
    for tx in entity_store.all_transactions(sqlite_path):
        sender, receiver = tx["sender_node"], tx["receiver_node"]
        _merge_entity(conn, sender,   "Account", "financial")
        _merge_entity(conn, receiver, "Account", "financial")
        try:
            _create_rel(conn, sender, receiver, "FINANCIAL_FLOW", {
                "txn_id":    tx["txn_id"],
                "amount":    float(tx["amount"]),
                "currency":  tx["currency"],
                "timestamp": tx["timestamp"],
                "flagged":   bool(tx["flagged"]),
                "reference": tx["reference"] or "",
            })
            counts["financial_flows"] += 1
        except Exception:
            pass

        # Create ACCOUNT_OF edges where the canonical node differs from the human name.
        # This links ACC-NST-01 → Northstar Trading Ltd so money paths can traverse
        # through company names across sources.
        for node_col, name_col in (
            ("sender_node", "sender_name"),
            ("receiver_node", "receiver_name"),
        ):
            node = tx[node_col]
            name = tx[name_col]
            if node and name and node != name:
                _merge_entity(conn, name, "Company", "financial")
                try:
                    _create_rel(conn, node, name, "RELATION", {
                        "relation_type": "ACCOUNT_OF",
                        "context":       f"Account {node} belongs to {name}",
                        "source_doc":    tx["txn_id"],
                    })
                except Exception:
                    pass

    # 4. Communication edges — resolve both sender and recipients via AliasResolver.
    # Skip nodes that resolve to noise (phone numbers, raw emails, etc.)
    for comm in entity_store.all_comms(sqlite_path):
        sender = resolver.resolve_person(comm["sender_node"])
        if _is_noise_entity(sender, "Person"):
            continue
        _merge_entity(conn, sender, "Person", "comms")
        for raw_recipient in json.loads(comm["recipients"]):
            person_node, company_node = resolver.resolve(raw_recipient)
            if _is_noise_entity(person_node, "Person"):
                continue
            _merge_entity(conn, person_node, "Person", "comms")
            if company_node:
                _merge_entity(conn, company_node, "Company", "comms")
            try:
                _create_rel(conn, sender, person_node, "COMMUNICATION", {
                    "event_id":      comm["event_id"],
                    "channel":       comm["channel"],
                    "timestamp":     comm["timestamp"],
                    "intent_signal": comm["intent_signal"] or "",
                    "subject":       comm["subject"] or "",
                })
                counts["communications"] += 1
            except Exception:
                pass

    return counts


# ---------------------------------------------------------------------------
# Query helpers used by the retriever
# ---------------------------------------------------------------------------

def cypher(conn: kuzu.Connection, query: str, params: Optional[dict] = None) -> list[dict]:
    """Execute a Cypher query and return results as a list of dicts."""
    result = conn.execute(query, params or {})
    return _rows_to_dicts(result)


def entity_subgraph(conn: kuzu.Connection, name: str) -> list[dict]:
    """
    Return all direct edges connected to a named entity as flat, readable dicts.
    Uses separate queries per relationship type — avoids opaque path objects.
    """
    results: list[dict] = []

    # Document-extracted typed relations (outgoing and incoming)
    for src_filter, tgt_filter in [
        (f"{{name: $name}}", ""),
        ("", f"{{name: $name}}"),
    ]:
        rows = cypher(
            conn,
            f"MATCH (a:Entity {src_filter})-[r:RELATION]->(b:Entity {tgt_filter}) "
            "RETURN a.name AS source, r.relation_type AS relation, b.name AS target, "
            "r.context AS context, r.source_doc AS doc",
            {"name": name},
        )
        results.extend(rows)

    # Financial flows (outgoing and incoming)
    for src_filter, tgt_filter in [
        (f"{{name: $name}}", ""),
        ("", f"{{name: $name}}"),
    ]:
        rows = cypher(
            conn,
            f"MATCH (a:Entity {src_filter})-[r:FINANCIAL_FLOW]->(b:Entity {tgt_filter}) "
            "RETURN a.name AS source, 'FINANCIAL_FLOW' AS relation, b.name AS target, "
            "r.amount AS amount, r.currency AS currency, r.flagged AS flagged, r.txn_id AS txn_id",
            {"name": name},
        )
        results.extend(rows)

    # Communications (outgoing)
    rows = cypher(
        conn,
        "MATCH (a:Entity {name: $name})-[r:COMMUNICATION]->(b:Entity) "
        "RETURN a.name AS source, 'COMMUNICATED_WITH' AS relation, b.name AS target, "
        "r.intent_signal AS intent, r.event_id AS event_id, r.channel AS channel",
        {"name": name},
    )
    results.extend(rows)

    return results


def money_path(conn: kuzu.Connection, src: str, dst: str, max_hops: int = 5) -> list[dict]:
    """
    Trace all financial flows reachable from `src` and report the chain.

    Because the graph bridges account IDs (ACC-NST-01) and company names
    (Northstar Trading Ltd) via ACCOUNT_OF edges, a simple variable-length
    FINANCIAL_FLOW path query misses layered structures.  Instead we perform
    a breadth-first expansion: at each hop, follow FINANCIAL_FLOW out, then
    optionally resolve through ACCOUNT_OF to the owning company before
    continuing.  Results are returned as a flat list of step dicts so the
    LLM can read the full chain.
    """
    visited: set[str] = set()
    frontier: list[str] = [src]
    steps: list[dict] = []
    hop = 0
    found_dst = False

    while frontier and hop < max_hops:
        hop += 1
        next_frontier: list[str] = []
        for node in frontier:
            if node in visited:
                continue
            visited.add(node)

            # Follow FINANCIAL_FLOW edges out from this node
            flows = cypher(
                conn,
                "MATCH (a:Entity {name: $name})-[r:FINANCIAL_FLOW]->(b:Entity) "
                "RETURN b.name AS receiver, r.amount AS amount, r.currency AS currency, "
                "r.flagged AS flagged, r.txn_id AS txn_id",
                {"name": node},
            )
            seen_receivers: set[str] = set()
            for f in flows:
                receiver = f["receiver"]
                steps.append({
                    "hop":      hop,
                    "from":     node,
                    "to":       receiver,
                    "amount":   f.get("amount"),
                    "currency": f.get("currency"),
                    "flagged":  f.get("flagged"),
                    "txn_id":   f.get("txn_id"),
                })
                if receiver == dst:
                    found_dst = True

                # Expand each unique receiver only once
                if receiver in seen_receivers:
                    continue
                seen_receivers.add(receiver)

                if receiver not in visited and receiver not in next_frontier:
                    next_frontier.append(receiver)

                # Resolve through ACCOUNT_OF: find accounts that belong to this company.
                account_links = cypher(
                    conn,
                    "MATCH (acct:Entity)-[r:RELATION {relation_type: 'ACCOUNT_OF'}]->(co:Entity {name: $name}) "
                    "RETURN DISTINCT acct.name AS account",
                    {"name": receiver},
                )
                for link in account_links:
                    acct = link["account"]
                    if acct not in visited and acct not in next_frontier:
                        next_frontier.append(acct)

        frontier = next_frontier
        if found_dst:
            break

    # Deduplicate steps by txn_id — multiple ACCOUNT_OF edges cause duplicate steps
    seen_txns: set = set()
    deduped_steps = []
    for s in steps:
        key = s.get("txn_id") or str(s)
        if key not in seen_txns:
            seen_txns.add(key)
            deduped_steps.append(s)

    if not found_dst and dst not in visited:
        deduped_steps.append({"note": f"No path from '{src}' to '{dst}' found within {max_hops} hops."})

    return deduped_steps


def all_entity_names(conn: kuzu.Connection) -> list[str]:
    """Return all entity names — used by the retriever for entity spotting."""
    rows = cypher(conn, "MATCH (e:Entity) RETURN e.name AS name ORDER BY e.name")
    return [r["name"] for r in rows if r.get("name")]
