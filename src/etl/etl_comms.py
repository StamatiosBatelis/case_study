"""
ETL step: comms_log.json → hybrid parsing → SQLite

Two-pass approach:
  Pass 1 (deterministic) — structural fields: event_id, timestamp, channel,
    sender, recipients, subject, attachments, IP/country metadata.
  Pass 2 (LLM) — unstructured body text: entity mentions buried in prose,
    sender alias resolution (email address → canonical name), intent signal.

The split matters: never use an LLM to parse a timestamp or count a dollar
amount, but rules cannot read "move the funds before end of week" and label
it instruct_wire_transfer.

Pipeline stage — import and call run() from src/pipeline.py.
"""

import json
from pathlib import Path
from typing import Optional

import httpx
from rich.console import Console

from src.llm_client import OllamaClient
from src.models import CommsBodyExtraction, DocumentExtractionResult, ExtractedRelation
from src.resolution.alias_resolver import _local_to_name
from src.storage import entity_store

COMMS_PATH = Path(__file__).parents[2] / "data" / "comms_log.json"

console = Console()

SYSTEM_PROMPT = (
    "You are an intelligence analyst assistant. Given a communication record, "
    "identify: (1) canonical names of specific people, companies, or accounts mentioned "
    "in the body text — only extract explicitly named entities, never infer or generate names; "
    "(2) the real-world identity of the sender if it can be confidently inferred from the "
    "email address or message content; (3) a short intent label if the message reveals "
    "coordination, financial instruction, or evasion — null if benign or unclear. "
    "Do NOT extract: dates, phone numbers, IP addresses, generic role titles (e.g. 'compliance team'), "
    "nationalities, or descriptive phrases. Only named people, companies, and account identifiers. "
    "Return only the JSON object — no prose, no markdown."
)

USER_TEMPLATE = """\
Event ID  : {event_id}
Channel   : {channel}
From      : {sender}
To        : {recipients}
Subject   : {subject}
Body:
{body}
"""

_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "CommsBodyExtraction",
        "strict": True,
        "schema": CommsBodyExtraction.model_json_schema(),
    },
}


def _sender_node(raw: str, llm_alias: Optional[str]) -> str:
    """
    Prefer LLM-resolved canonical name; fall back to heuristic; fall back to raw.
    Keeps email addresses out of the graph wherever a real name is available.
    """
    return llm_alias or _local_to_name(raw.split("@")[0]) or raw


class LocalLLMCommsEngine:
    """Extracts entity mentions and intent signals from communication bodies."""

    MAX_RETRIES = 2

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "qwen2.5:7b",
    ) -> None:
        self.client = OllamaClient(base_url=base_url)
        self.model = model

    def extract(self, record: dict) -> CommsBodyExtraction:
        prompt = USER_TEMPLATE.format(
            event_id=record["event_id"],
            channel=record["channel"],
            sender=record["from"],
            recipients=", ".join(record.get("to", [])),
            subject=record.get("subject") or "(none)",
            body=record.get("body", ""),
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
                result = CommsBodyExtraction.model_validate_json(raw)
                result.event_id = record["event_id"]
                return result
            except Exception as exc:
                last_exc = exc
                if attempt < self.MAX_RETRIES:
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({
                        "role": "user",
                        "content": f"Invalid JSON. Error: {exc}. Try again.",
                    })

        raise RuntimeError(
            f"Comms extraction failed after {self.MAX_RETRIES} attempts: {last_exc}"
        ) from last_exc


def run(
    comms_path: Path = COMMS_PATH,
    db_path: Optional[Path] = None,
    base_url: str = "http://localhost:11434/v1",
    model: str = "qwen2.5:7b",
    dry_run: bool = False,
    engine: Optional[LocalLLMCommsEngine] = None,
) -> list[dict]:
    """
    Parse comms_log.json and persist to SQLite.

    Returns the list of parsed+enriched comm dicts.
    The `engine` parameter accepts a mock for testing (same pattern as etl_documents).
    """
    if not comms_path.exists():
        raise FileNotFoundError(f"Comms file not found: {comms_path}")

    db = db_path or entity_store.DB_PATH

    if not dry_run:
        entity_store.init_db(db)

    with open(comms_path) as f:
        raw_records = json.load(f)

    _engine = engine or LocalLLMCommsEngine(base_url=base_url, model=model)
    enriched: list[dict] = []

    for record in raw_records:
        event_id = record["event_id"]
        console.print(f"  [comms] {event_id} ({record['channel']}) ... ", end="")

        try:
            # Pass 2: LLM body extraction
            extraction = _engine.extract(record)
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"Inference server unreachable: {exc}\n"
                "Start Ollama: ollama serve"
            ) from exc
        except Exception as exc:
            console.print(f"[yellow]LLM failed, using heuristics[/yellow]: {exc}")
            extraction = CommsBodyExtraction(
                event_id=event_id,
                entity_mentions=[],
                sender_alias=None,
                intent_signal=None,
            )

        # Pass 1 + 2 merged: build the flat record for SQLite
        sender_raw = record["from"]
        sender_node = _sender_node(sender_raw, extraction.sender_alias)
        metadata = record.get("metadata", {})

        comm_record = {
            "event_id":     event_id,
            "timestamp":    record["timestamp"],
            "channel":      record["channel"],
            "sender_raw":   sender_raw,
            "sender_node":  sender_node,
            "recipients":   record.get("to", []),
            "subject":      record.get("subject"),
            "body":         record.get("body"),
            "attachments":  record.get("attachments", []),
            "ip_address":   metadata.get("ip"),
            "country":      metadata.get("country"),
            "intent_signal": extraction.intent_signal,
        }
        enriched.append(comm_record)

        intent_label = f"[yellow]{extraction.intent_signal}[/yellow]" if extraction.intent_signal else "—"
        console.print(
            f"[green]✓[/green] {len(extraction.entity_mentions)} mentions | intent: {intent_label}"
        )

        if not dry_run:
            entity_store.upsert_comm(comm_record, db_path=db)
            # Store entity mentions as MENTIONED_IN_COMM edges so the graph
            # connects free-text references back to canonical entity nodes
            if extraction.entity_mentions:
                entity_store.upsert_extraction(
                    DocumentExtractionResult(
                        doc_id=event_id,
                        entities=[],
                        relations=[
                            ExtractedRelation(
                                source_entity=sender_node,
                                relation_type="MENTIONED_IN_COMM",
                                target_entity=mention,
                                context=f"Mentioned in {event_id}",
                            )
                            for mention in extraction.entity_mentions
                        ],
                    ),
                    db_path=db,
                )

    return enriched
