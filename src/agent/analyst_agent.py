"""
Analyst Agent — reasoning tier for the intelligence investigation system.

Architecture
------------
Single-model ReAct loop (Qwen2.5-7b via Ollama, OpenAI-compatible API).
The LLM is given three tools and asked to call them until it has enough
evidence to answer the analyst's question:

  retrieve(query, mode)          — hybrid Graph+Vector retrieval
  trace_money(from, to, hops)    — Cypher financial path query
  cypher(query)                  — raw Cypher for power users

After at most MAX_STEPS tool calls the LLM is forced to synthesise a final
answer. Every source cited in tool results is tracked and returned so the
analyst can verify evidence (traceability).

Design notes
------------
- No hidden reasoning: all tool inputs/outputs are logged so analysts can
  audit exactly what evidence was retrieved.
- Retrieval is always done via the HybridRetriever — the agent never
  generates entity names or facts from parametric memory alone.
- The system prompt instructs the model to ground claims in tool output
  and never speculate beyond what the evidence shows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

import kuzu

from src.llm_client import OllamaClient
from src.storage import kuzu_store, vector_store
from src.retrieval.retriever import HybridRetriever

# ---------------------------------------------------------------------------
# Tool schemas (OpenAI / Ollama tool-calling format)
# ---------------------------------------------------------------------------

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve",
            "description": (
                "Search for evidence across the knowledge graph and document/comms store. "
                "Use mode='structured' for entity/transaction/relationship queries "
                "(e.g. 'who owns Northstar Trading?', 'find flagged transactions'). "
                "Use mode='semantic' for narrative or intent questions "
                "(e.g. 'communications suggesting fear of detection'). "
                "Use mode='hybrid' to correlate across sources "
                "(e.g. 'what connects Elena Ross to the Rotterdam shipment?'). "
                "Leave mode='auto' to let the system decide."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["auto", "structured", "semantic", "hybrid"],
                        "description": "Retrieval mode. Default: auto.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "trace_money",
            "description": (
                "Trace financial flow paths between two named entities in the graph. "
                "Returns all FINANCIAL_FLOW edge chains up to max_hops steps long. "
                "Use this to follow a money trail from a sender to an ultimate recipient."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "from_entity": {
                        "type": "string",
                        "description": "Exact name of the source entity (account or company).",
                    },
                    "to_entity": {
                        "type": "string",
                        "description": "Exact name of the target entity.",
                    },
                    "max_hops": {
                        "type": "integer",
                        "description": "Maximum number of hops to search. Default: 5.",
                        "default": 5,
                    },
                },
                "required": ["from_entity", "to_entity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cypher",
            "description": (
                "Execute a raw Cypher query against the knowledge graph for precise lookups. "
                "Schema: Entity(name, entity_type, layer), "
                "RELATION(relation_type, context, source_doc), "
                "FINANCIAL_FLOW(txn_id, amount, currency, timestamp, flagged, reference), "
                "COMMUNICATION(event_id, channel, timestamp, intent_signal, subject). "
                "Example: MATCH (e:Entity {name: 'Marcus Vane'})-[r:FINANCIAL_FLOW]->(b) "
                "RETURN b.name, r.amount, r.flagged"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A valid Cypher query string.",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

_SYSTEM_PROMPT = """\
You are an intelligence analyst assistant. Your job is to help investigate \
suspicious activity across communications, financial transactions, and \
intelligence documents.

IMPORTANT: Always respond in English, regardless of the language used in retrieved data.

Rules:
1. Always call at least two tools before writing your final answer — one to retrieve \
   graph/document evidence (retrieve or cypher) and one to follow up on what you find.
2. For questions about ownership, control, or connections: always call retrieve() with \
   mode='structured' to find entity relationships from the knowledge graph.
3. For questions about money flows: call trace_money() first, then call retrieve() with \
   mode='hybrid' to find ownership and document evidence for each entity in the chain.
4. Ground every claim in specific retrieved data — entity names, transaction IDs \
   (TXN-xxx), document IDs (DOC-001), communication event IDs (evt_001).
5. Never speculate beyond what the evidence shows. If evidence is absent, say so explicitly.
6. When you have gathered enough evidence, write a concise, structured final answer with:
   - A one-paragraph summary
   - Key findings as bullet points with source references
   - Any caveats about missing or ambiguous evidence
"""

MAX_STEPS = 6


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class AgentResponse:
    answer: str
    sources: list[str] = field(default_factory=list)
    tool_calls_made: list[dict] = field(default_factory=list)
    steps: int = 0

    def display(self) -> str:
        lines = [self.answer]
        if self.sources:
            lines.append("\n**Sources referenced:**")
            for s in sorted(set(self.sources)):
                lines.append(f"  • {s}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class AnalystAgent:
    """
    Stateless single-turn investigation agent.
    Instantiate once and call .run(question) repeatedly.
    """

    def __init__(
        self,
        kuzu_conn: kuzu.Connection,
        chroma_client,
        base_url: str = "http://localhost:11434/v1",
        model: str = "qwen2.5:7b",
    ) -> None:
        self._conn = kuzu_conn
        self._retriever = HybridRetriever(kuzu_conn, chroma_client)
        self._llm = OllamaClient(base_url=base_url)
        self._model = model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, question: str) -> AgentResponse:
        """Run the agent on a natural-language analyst question."""
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ]
        sources: list[str] = []
        tool_log: list[dict] = []
        steps = 0

        while steps < MAX_STEPS:
            response = self._llm.chat_completions_create(
                model=self._model,
                messages=messages,
                tools=_TOOLS,
                tool_choice="auto",
            )
            msg = response.choices[0].message
            steps += 1

            # No tool call → model is done
            if not msg.tool_calls:
                return AgentResponse(
                    answer=msg.content or "",
                    sources=sources,
                    tool_calls_made=tool_log,
                    steps=steps,
                )

            # Append assistant turn (with tool_calls) as a plain dict
            messages.append(msg.to_dict())

            # Execute each tool call
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                result_text, call_sources = self._dispatch(tool_name, args)
                sources.extend(call_sources)
                tool_log.append({"tool": tool_name, "args": args})

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      result_text,
                })

        # Exceeded MAX_STEPS — ask for a final answer without tools
        messages.append({
            "role":    "user",
            "content": "You have reached the tool call limit. Synthesise your final answer now based on the evidence retrieved.",
        })
        final = self._llm.chat_completions_create(
            model=self._model,
            messages=messages,
        )
        return AgentResponse(
            answer=final.choices[0].message.content or "",
            sources=sources,
            tool_calls_made=tool_log,
            steps=steps,
        )

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, name: str, args: dict) -> tuple[str, list[str]]:
        """Execute a tool call. Returns (result_text, [sources])."""
        try:
            if name == "retrieve":
                return self._tool_retrieve(args)
            elif name == "trace_money":
                return self._tool_trace_money(args)
            elif name == "cypher":
                return self._tool_cypher(args)
            else:
                return f"Unknown tool: {name}", []
        except Exception as exc:
            return f"Tool error ({name}): {exc}", []

    def _tool_retrieve(self, args: dict) -> tuple[str, list[str]]:
        query = args.get("query", "")
        mode  = args.get("mode", "auto")
        ctx   = self._retriever.retrieve(query, mode=mode)
        text  = ctx.to_context_string()
        # Extract source IDs for traceability
        sources = self._extract_sources(ctx)
        if not text.strip():
            text = "No relevant evidence found for this query."
        return text, sources

    def _tool_trace_money(self, args: dict) -> tuple[str, list[str]]:
        src      = args.get("from_entity", "")
        dst      = args.get("to_entity", "")
        max_hops = int(args.get("max_hops", 5))
        rows     = kuzu_store.money_path(self._conn, src, dst, max_hops)
        if not rows:
            return f"No financial path found between '{src}' and '{dst}'.", []
        lines = [f"Financial path: {src} → {dst} ({len(rows)} path(s) found)"]
        # rows are path objects — format what we can
        for i, row in enumerate(rows[:10]):
            lines.append(f"  Path {i+1}: {row}")
        return "\n".join(lines), [src, dst]

    def _tool_cypher(self, args: dict) -> tuple[str, list[str]]:
        query = args.get("query", "")
        rows  = kuzu_store.cypher(self._conn, query)
        if not rows:
            return "Query returned no results.", []
        lines = [f"Cypher results ({len(rows)} rows):"]
        for row in rows[:20]:
            lines.append(f"  {row}")
        sources = []
        for row in rows:
            for v in row.values():
                if isinstance(v, str) and (v.startswith("DOC-") or v.startswith("evt_") or v.startswith("TXN-")):
                    sources.append(v)
        return "\n".join(lines), sources

    # ------------------------------------------------------------------
    # Source extraction helper
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_sources(ctx) -> list[str]:
        sources = []
        for r in ctx.semantic_results:
            meta = r.get("metadata", {})
            for key in ("doc_id", "event_id"):
                if val := meta.get(key):
                    sources.append(val)
        for r in ctx.graph_results:
            for v in r.values():
                if isinstance(v, str) and (v.startswith("DOC-") or v.startswith("evt_") or v.startswith("TXN-")):
                    sources.append(v)
        return sources
