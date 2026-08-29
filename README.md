# Intelligence Investigation Platform

A working prototype of an AI-powered investigation tool that correlates evidence across three disconnected source systems — communications logs, financial transactions, and unstructured documents — using a hybrid graph-vector retrieval architecture and a local LLM reasoning agent.

Built as a take-home technical assessment. All inference runs locally via Ollama; no data leaves the machine.

---

## What it does

Analysts ask open-ended investigation questions in plain English:

> *"Who controls Northstar Trading Ltd and what transactions did they make?"*  
> *"Trace the money from ACC-4471 to Bluewater Ventures Ltd."*  
> *"What communications suggest coordination around the Rotterdam shipment?"*

The agent uses a ReAct tool-calling loop over three tools — graph traversal (KuzuDB/Cypher), semantic search (ChromaDB), and financial path tracing — before synthesising a grounded, source-cited answer.

---

## Architecture

```
data/                         # Mock dataset (comms, transactions, documents)
src/
  etl/                        # Extraction: LLM for unstructured, regex for structured
  storage/
    kuzu_store.py             # KuzuDB graph: entities, relationships, financial flows
    vector_store.py           # ChromaDB: embedded document and comms chunks
  retrieval/
    retriever.py              # Hybrid retriever: entity-anchored graph + vector search
  agent/
    analyst_agent.py          # ReAct agent: tool-calling loop over local LLM
    __main__.py               # Terminal REPL
  app/
    __main__.py               # Browser-based chat UI (pure Python, no extra deps)
  llm_client.py               # Thin httpx client for Ollama — no openai package
  pipeline.py                 # ETL orchestration: raw files → KuzuDB + ChromaDB
```

See `DESIGN.md` for the full architectural reasoning.

---

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) running locally

Pull the two required models before running:

```bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Running

### Step 1 — Build the knowledge stores

Runs ETL across all three source files and populates KuzuDB (graph) and ChromaDB (vector):

```bash
python -m src.pipeline
```

This takes a few minutes — the LLM extracts entities and relationships from each document and communication. Run it once; subsequent app/agent starts reuse the built stores.

To point at a different model or a vLLM endpoint:

```bash
python -m src.pipeline --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-7B-Instruct
```

### Step 2 — Run the app

**Browser UI** (opens automatically at `http://localhost:8080`):

```bash
python -m src.app
```

**Terminal REPL** (useful for debugging — prefix a query with `/debug` to see the tool trace):

```bash
python -m src.agent
```

Both accept `--model` and `--base-url` flags if you're not using the Ollama defaults.

---

## Tests

```bash
pytest tests/ -v
```

The test suite covers ETL parsing, graph store operations, vector store indexing and search, entity spotting, and the hybrid retriever. Tests use stub embedders and an in-memory KuzuDB instance — no Ollama required to run them.

---

## Dataset

The mock dataset is in `data/` — a few dozen records per source, enough to demonstrate cross-source correlation:

| File | Source | Records |
|------|--------|---------|
| `documents.json` | Intelligence reports (unstructured) | 8 docs |
| `comms_log.json` | Communications log (semi-structured) | ~30 events |
| `transactions.json` | Financial transactions (structured) | ~25 transactions |

The pre-built KuzuDB graph (`data/kuzu_db/`) and ChromaDB index (`data/chroma_db/`) are included in the repo so you can skip Step 1 and go straight to Step 2 if you just want to run the agent.

---

## AI tool usage

**Claude Code** (Anthropic's CLI coding agent) was used as an implementation accelerator during this project. In line with how I work day-to-day, I used it to move faster on code, not to make architectural decisions.

All key design choices were mine: the hybrid graph-vector retrieval strategy, the decision to use KuzuDB over NetworkX, choosing a local open-weight LLM for extraction to satisfy the air-gap constraint, the entity-anchoring mechanism, the ReAct agent architecture, and where to draw the deterministic/AI boundary. These were thought through before implementation began and are explained in `DESIGN.md`.

Claude Code was used for:

- **Code generation** — translating design decisions into working Python: the ETL pipeline, KuzuDB store, ChromaDB vector store, hybrid retriever, ReAct agent loop, and web UI.
- **Debugging** — catching low-level library behaviour: KuzuDB 0.11.3's requirement for a non-existent path on init, ChromaDB's `embed_query` interface in newer versions, the account-ID-to-company-name bridging gap in financial graph traversal.
- **Library recommendations** — e.g. flagging that Python's built-in `SequenceMatcher` penalises word order differences in company names ("Northstar Trading" vs "Trading Northstar"), and recommending `thefuzz.token_sort_ratio` which normalises token order before comparing — a meaningful improvement for entity deduplication.
- **Extraction quality** — identified that without `temperature=0` and explicit exclusion lists in the system prompt, a 7B model treats almost any noun phrase as an entity (dates, role titles, nationalities, descriptive phrases all became graph nodes). Adding both controls reduced extraction noise substantially.
- **Tests and boilerplate** — test file scaffolding, stub embedder setup, and the `http.server` web app shell.

Where Claude Code's suggestions didn't fit the design — keeping NetworkX in the graph layer, adding Streamlit as a dependency, using the `openai` SDK against a local endpoint — I redirected it.
