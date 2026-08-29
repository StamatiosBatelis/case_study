"""
ETL step: documents.json → LLM entity/relation extraction → SQLite

Pipeline stage — import and call run() from src/pipeline.py.
"""

import json
from pathlib import Path
from typing import Optional

import httpx
from rich.console import Console
from rich.table import Table

from src.llm_client import OllamaClient
from src.models import DocumentExtractionResult
from src.storage import entity_store

DOCS_PATH = Path(__file__).parents[2] / "data" / "documents.json"

console = Console()

SYSTEM_PROMPT = (
    "You are a precision intelligence extraction engine. "
    "Extract named entities and typed relationships from the provided document. "
    "Only extract entities that are explicitly named in the text — never infer or generate names. "
    "Valid entity types: Person, Company, Account, Location, Jurisdiction, Bank, Organisation. "
    "Do NOT extract: dates, years, phone numbers, IP addresses, registration numbers, "
    "monetary amounts, generic descriptions, role titles, nationalities, or adjectives. "
    "An entity must be a specific named person, organisation, account, or place — "
    "not a concept, outcome, or descriptive phrase. "
    "Use the canonical name exactly as it appears in the text. "
    "Return only the JSON object — no explanation, no markdown fences."
)

USER_TEMPLATE = """\
Document ID : {doc_id}
Title       : {title}
Type        : {doc_type}
Date        : {date}
Source      : {source}

Content:
{content}
"""

_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "DocumentExtractionResult",
        "strict": True,
        "schema": DocumentExtractionResult.model_json_schema(),
    },
}


class LocalLLMExtractionEngine:
    """
    Calls a local OpenAI-compatible inference server (Ollama or vLLM).

    Structured output is requested via response_format / json_schema so the
    model is constrained to emit a valid DocumentExtractionResult object.
    A retry loop handles the rare case where a smaller model still drifts.
    """

    MAX_RETRIES = 2

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "qwen2.5:7b",
    ) -> None:
        self.client = OllamaClient(base_url=base_url)
        self.model = model

    def extract(self, doc: dict) -> DocumentExtractionResult:
        prompt = USER_TEMPLATE.format(
            doc_id=doc["doc_id"],
            title=doc["title"],
            doc_type=doc.get("type", ""),
            date=doc.get("date", ""),
            source=doc.get("source", ""),
            content=doc["content"],
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        last_exc: Optional[Exception] = None
        raw: str = ""
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                response = self.client.chat_completions_create(
                    model=self.model,
                    messages=messages,
                    response_format=_RESPONSE_FORMAT,
                    temperature=0,
                )
                raw = response.choices[0].message.content
                result = DocumentExtractionResult.model_validate_json(raw)
                result.doc_id = doc["doc_id"]
                return result
            except Exception as exc:
                last_exc = exc
                if attempt < self.MAX_RETRIES:
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({
                        "role": "user",
                        "content": f"Your response was not valid JSON. Error: {exc}. Try again.",
                    })

        raise RuntimeError(
            f"Extraction failed after {self.MAX_RETRIES} attempts: {last_exc}"
        ) from last_exc


def _print_result(result: DocumentExtractionResult) -> None:
    console.print(f"\n[bold cyan]{result.doc_id}[/bold cyan]")

    ent_table = Table("Name", "Type", show_header=True, header_style="bold magenta")
    for e in result.entities:
        ent_table.add_row(e.name, e.entity_type)
    console.print(ent_table)

    rel_table = Table("Source", "Relation", "Target", show_header=True, header_style="bold green")
    for r in result.relations:
        rel_table.add_row(r.source_entity, r.relation_type, r.target_entity)
    console.print(rel_table)


def run(
    docs_path: Path = DOCS_PATH,
    db_path: Optional[Path] = None,
    base_url: str = "http://localhost:11434/v1",
    model: str = "qwen2.5:7b",
    dry_run: bool = False,
    engine: Optional[LocalLLMExtractionEngine] = None,
) -> list[DocumentExtractionResult]:
    """
    Extract entities and relations from documents.json and persist to SQLite.

    Args:
        docs_path: Path to documents.json.
        db_path:   SQLite DB path (defaults to data/knowledge_graph.db).
        base_url:  OpenAI-compatible inference server URL.
        model:     Model name as known to the inference server.
        dry_run:   If True, print results and skip DB writes.
        engine:    Optional pre-built engine (used in tests to inject a mock).

    Returns:
        List of DocumentExtractionResult — one per document processed.
    """
    if not docs_path.exists():
        raise FileNotFoundError(f"Documents file not found: {docs_path}")

    db = db_path or entity_store.DB_PATH

    if not dry_run:
        entity_store.init_db(db)

    with open(docs_path) as f:
        docs = json.load(f)

    _engine = engine or LocalLLMExtractionEngine(base_url=base_url, model=model)
    results: list[DocumentExtractionResult] = []

    for doc in docs:
        doc_id = doc["doc_id"]
        console.print(f"  [documents] {doc_id} — {doc['title']} ... ", end="")
        try:
            result = _engine.extract(doc)
            results.append(result)
            console.print(
                f"[green]✓[/green] {len(result.entities)} entities, "
                f"{len(result.relations)} relations"
            )
            if dry_run:
                _print_result(result)
            else:
                entity_store.upsert_extraction(result, db_path=db)
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"Inference server unreachable at {base_url}: {exc}\n"
                "Start Ollama:  ollama serve\n"
                "Start vLLM:    vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000"
            ) from exc
        except Exception as exc:
            console.print(f"[red]failed[/red]: {exc}")

    return results
