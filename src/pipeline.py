"""
Intelligence ETL Pipeline
=========================
Orchestrates all stages in sequence:

  1. documents.json    → LLM entity/relation extraction  → SQLite
  2. transactions.json → deterministic field parsing      → SQLite
  3. comms_log.json   → hybrid (rules + LLM body)        → SQLite
  4. SQLite           → KuzuDB graph sync                 (Cypher-queryable)
  5. SQLite + files   → ChromaDB vector indexing          (semantic search)

Usage:
    python -m src.pipeline                        # full run, Ollama defaults
    python -m src.pipeline --dry-run              # extract only, no DB writes
    python -m src.pipeline --skip-docs            # skip document ETL stage
    python -m src.pipeline \\
        --base-url http://localhost:8000/v1 \\
        --model Qwen/Qwen2.5-7B-Instruct          # point at vLLM
"""

import argparse
import json
import time
from pathlib import Path

from rich.console import Console
from rich.rule import Rule

from src.etl import etl_comms, etl_documents, etl_transactions
from src.storage import entity_store, kuzu_store, vector_store

console = Console()

DATA_DIR = Path(__file__).parent.parent / "data"


def run_pipeline(
    base_url: str = "http://localhost:11434/v1",
    model: str = "qwen2.5:7b",
    db_path: Path = entity_store.DB_PATH,
    dry_run: bool = False,
    skip_docs: bool = False,
) -> None:
    start = time.perf_counter()

    console.print(Rule("[bold]Intelligence ETL Pipeline"))
    console.print(f"Model   : {model} @ {base_url}")
    console.print(f"DB      : {db_path}")
    console.print(f"Dry-run : {dry_run}")
    console.print()

    # ------------------------------------------------------------------ #
    # Stage 1 — Documents: LLM extraction
    # ------------------------------------------------------------------ #
    if not skip_docs:
        console.print(Rule("[cyan]Stage 1 / 5 — Documents (LLM extraction)"))
        doc_results = etl_documents.run(
            docs_path=DATA_DIR / "documents.json",
            db_path=db_path,
            base_url=base_url,
            model=model,
            dry_run=dry_run,
        )
        console.print(f"  → {len(doc_results)} documents processed\n")
    else:
        console.print("[dim]Stage 1 skipped (--skip-docs)[/dim]\n")

    # ------------------------------------------------------------------ #
    # Stage 2 — Transactions: deterministic parsing (no LLM)
    # ------------------------------------------------------------------ #
    console.print(Rule("[cyan]Stage 2 / 5 — Transactions (deterministic)"))
    txn_results = etl_transactions.run(
        transactions_path=DATA_DIR / "transactions.json",
        db_path=db_path,
        dry_run=dry_run,
    )
    console.print(f"  → {len(txn_results)} transactions processed\n")

    # ------------------------------------------------------------------ #
    # Stage 3 — Communications log: deterministic metadata + LLM body
    # ------------------------------------------------------------------ #
    console.print(Rule("[cyan]Stage 3 / 5 — Communications log (hybrid)"))
    comm_results = etl_comms.run(
        comms_path=DATA_DIR / "comms_log.json",
        db_path=db_path,
        base_url=base_url,
        model=model,
        dry_run=dry_run,
    )
    console.print(f"  → {len(comm_results)} communications processed\n")

    if dry_run:
        elapsed = time.perf_counter() - start
        console.print(Rule("[yellow]Dry-run complete"))
        console.print(f"  Time: {elapsed:.1f}s")
        return

    # ------------------------------------------------------------------ #
    # Stage 4 — KuzuDB graph sync
    # ------------------------------------------------------------------ #
    console.print(Rule("[cyan]Stage 4 / 5 — KuzuDB graph sync"))
    counts = kuzu_store.sync(
        db_path=kuzu_store.KUZU_PATH,
        sqlite_path=db_path,
    )
    console.print(
        f"  Nodes       : {counts['nodes']}\n"
        f"  Relations   : {counts['relations']}\n"
        f"  Fin. flows  : {counts['financial_flows']}\n"
        f"  Comms edges : {counts['communications']}\n"
    )

    # ------------------------------------------------------------------ #
    # Stage 5 — ChromaDB vector indexing
    # ------------------------------------------------------------------ #
    console.print(Rule("[cyan]Stage 5 / 5 — ChromaDB vector indexing"))

    kuzu_db   = kuzu_store.open_db()
    kuzu_conn = kuzu_store.kuzu.Connection(kuzu_db)
    known_entities = kuzu_store.all_entity_names(kuzu_conn)

    chroma = vector_store.open_client()

    with open(DATA_DIR / "documents.json") as f:
        docs = json.load(f)
    n_doc_chunks = vector_store.index_documents(chroma, docs, known_entities)
    console.print(f"  Documents : {n_doc_chunks} chunks indexed")

    comms_rows = entity_store.all_comms(db_path)
    comms_dicts = [dict(r) for r in comms_rows]
    n_comm_chunks = vector_store.index_comms(chroma, comms_dicts, known_entities)
    console.print(f"  Comms     : {n_comm_chunks} records indexed\n")

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    elapsed = time.perf_counter() - start
    console.print(Rule("[bold green]Pipeline complete"))
    console.print(
        f"  Graph  : {counts['nodes']} nodes, "
        f"{counts['relations'] + counts['financial_flows'] + counts['communications']} edges\n"
        f"  Vectors: {n_doc_chunks + n_comm_chunks} chunks\n"
        f"  Time   : {elapsed:.1f}s\n"
        f"  SQLite : {db_path}\n"
        f"  KuzuDB : {kuzu_store.KUZU_PATH}\n"
        f"  Chroma : {vector_store.CHROMA_PATH}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the intelligence ETL pipeline")
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--db", type=Path, default=entity_store.DB_PATH)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-docs", action="store_true")
    args = parser.parse_args()

    run_pipeline(
        base_url=args.base_url,
        model=args.model,
        db_path=args.db,
        dry_run=args.dry_run,
        skip_docs=args.skip_docs,
    )


if __name__ == "__main__":
    main()
