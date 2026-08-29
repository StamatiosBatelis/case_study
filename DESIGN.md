# Intelligence Investigation Platform — Design Document

## Problem Statement

An intelligence team needs to investigate suspicious activity across three disconnected systems: a communications log (high-volume, semi-structured), a financial transaction database (structured), and a document store (unstructured reports and files). Analysts ask open-ended questions that require correlating evidence across all three — "who controls this entity?", "trace the money from this account to that company", "what comms suggest coordination around this shipment?"

The constraint that shaped every decision: **fully air-gapped deployment**. No data leaves the network. All inference runs locally.

---

## Why Not Pure Vector RAG

The first instinct for an LLM-powered search system is vector RAG — embed everything, search by cosine similarity. It breaks down immediately here.

"Who transferred funds to Bluewater Ventures?" cannot be answered by cosine similarity. The answer is a graph structure: James Chen → ACC-4471 → Northstar Trading Ltd → Harrington Capital → Bluewater Ventures Ltd. That path exists in the transaction edges, not in any document passage. A vector search has no concept of hops, no way to traverse edges, no mechanism to follow money through intermediate accounts.

Similarly, "What is the beneficial ownership structure of Shell Corp IO?" requires joining entity-relationship records — directorship edges, beneficiary edges, entity type metadata. Vector search can find passages that *mention* Shell Corp IO, but it cannot assemble the ownership graph from raw text similarity scores.

Pure vector RAG also suffers from entity aliasing. "Northstar Trading" and "Northstar Trading Ltd" are the same company; they may appear as different strings in different source documents. A vector store treats them as similar-but-distinct passages. A graph store, after entity resolution, treats them as the same node.

---

## Why Not Pure Graph RAG

The opposite failure is equally real. "Find communications suggesting fear of detection" cannot be answered by Cypher. The graph captures only what was explicitly extracted: typed edges with typed relationships. Latent meaning — the anxiety in "we need to move this before Friday", the evasion in "use the usual channel" — lives in free text and requires semantic search to surface.

Cypher also can't handle vague intent queries. "What unusual patterns link the Rotterdam shipment to the Liechtenstein trust?" doesn't map to a graph traversal because we don't know which edges to follow until we understand what "unusual" means in context.

---

## The Architecture: Entity-Anchored Hybrid GraphRAG

The solution is to run both retrieval paths and make them reinforce each other.

```
Analyst Query
      │
      ▼
 Query Classifier
      │
   ┌──┴──────────────────┐
   ▼                     ▼
KuzuDB                ChromaDB
Cypher traversal       Vector similarity
Entity subgraphs       Document chunks
Financial flows        Comms body text
   │                     │
   └──────────┬──────────┘
              ▼
       Context Fusion
       (graph evidence + anchored vector chunks)
              ▼
       LLM Reasoning
       (grounded, cited answer)
```

The key mechanism is **entity anchoring**. Graph traversal returns entity names. Those names filter the vector search — instead of searching all of ChromaDB, we search only chunks associated with the entities the graph found. This dramatically reduces noise and keeps the LLM context focused on relevant evidence rather than loosely related passages.

The reverse works too: vector chunks carry entity metadata from ETL — specifically the canonical entity names as extracted and stored in SQLite. When ChromaDB returns a document excerpt, the agent knows which graph nodes to expand next. Canonical names are the stable identifier in this design: KuzuDB uses `name` as the primary key, SQLite uses the same strings, and ChromaDB metadata mirrors them. If entities are renamed or merged, a re-sync and re-index is required — but that's the correct response since both stores would be stale regardless of whether metadata stored names or UUIDs.

**Context bounding in hybrid mode:** Multi-entity queries are capped at 3 entity anchors per retrieval step (the three highest-confidence spotted entities) and at 10 total chunks returned. This prevents ChromaDB `where` filters from expanding unboundedly when a query mentions many entities, and keeps the LLM context window manageable. The cap is configurable via `n_semantic_results` on `HybridRetriever`.

**Concurrency constraint:** KuzuDB is embedded and uses file-level locking — only one process can hold a write lock at a time. `kuzu_store.sync()` is a strictly offline, single-process batch operation. The agent query service opens KuzuDB only after the pipeline has completed and exited. Running both simultaneously on the same machine will raise a lock error. In production this is enforced by running ingestion as a scheduled job (cron, Airflow) with the query service blocked or restarted after each sync.

Three modes are exposed to the agent:

- **Structured** — Cypher only. For ownership queries, transaction lookups, relationship traversals.
- **Semantic** — Vector only. For intent and narrative questions over free text.
- **Hybrid** — Graph expands entities, then anchored vector search over returned entity names. For cross-source correlation questions.

The query classifier uses keyword heuristics (no LLM call) — fast and deterministic. The agent can also override the mode explicitly if the heuristic misfires.

---

## Extraction Strategy: Why a Local LLM Over spaCy or Cloud APIs

Three options were on the table for entity and relationship extraction from raw source documents.

**spaCy / rules** is fast and deterministic, which is appealing. The problem is the domain. Standard NER models know persons, organisations, and locations. They don't know "Beneficial Owner via Liechtenstein trust", "ACCOUNT_OF relationship between an account ID and a legal entity", or "intent_signal from an encrypted comms channel". You can write rules to catch explicit patterns, and we do — structured fields like event IDs, account numbers, and transaction amounts are all extracted with regex. But relationships require understanding context, not pattern matching.

**Cloud LLMs (Gemini, Claude)** offer the best extraction quality. Structured output via function calling essentially eliminates JSON parsing failures, and frontier models handle ambiguous domain language well. The hard stop is the deployment constraint: financial records, SARs, and comms logs cannot leave the network. A cloud API call is a data exfiltration event in this context.

**Local open-weight LLM (Qwen2.5:7b via Ollama)** threads the needle. Extraction quality is below frontier models — JSON formatting occasionally needs repair, and subtle multi-hop relationships are sometimes missed on the first pass — but it's sufficient for the entity types we need, runs entirely on local hardware, and has no throughput ceiling from API rate limits.

Two extraction controls reduce hallucination significantly. First, `temperature=0` forces the model into its most deterministic mode — entity extraction is a precision task, not a creative one, so stochastic sampling only adds noise. Second, the system prompt explicitly lists what not to extract: dates, phone numbers, IP addresses, registration numbers, monetary amounts, role titles, nationalities, and descriptive phrases. Without these exclusions, a 7B model will treat almost any noun phrase as an entity. The combination reduces extraction noise substantially, though some relation endpoints still slip through as untyped Unknown nodes — these are filtered at graph sync time by a pattern-based noise guard before any node is written to KuzuDB.

On licensing: we specifically chose Qwen2.5:7b (not 3B). The 3B variant ships under a research-only license and cannot be used in company products. Qwen2.5:7b is licensed for commercial use up to 100 million monthly active users — well above the scale of an internal intelligence tool. Llama 3.2 (Apache 2.0 for 1B/3B) would also have been a clean choice, but Qwen2.5:7b performs better on structured extraction tasks at the same parameter count.

The pipeline hybrid reflects a deliberate split:

- **Batch ingestion**: local LLM does entity and relationship extraction, regex handles structured fields. Runs offline, cost is fixed infrastructure.
- **Analyst query**: same local LLM reasons over the pre-built graph and vector store. No re-extraction at query time.

This matters because extraction is a one-time (or periodic) cost. Reasoning is latency-sensitive. Keeping both on the same local model avoids managing two different inference environments while staying air-gapped.

---

## Database Choices

### KuzuDB Over NetworkX

NetworkX was the initial graph layer and was removed. The distinction is that NetworkX is an in-memory Python data structure, not a database. It's excellent for running graph algorithms — PageRank, betweenness centrality, community detection — on subgraphs that fit in RAM. It's not designed for persistent storage, Cypher querying, or multi-hop traversal on large datasets.

KuzuDB is an embedded graph database (C++, Python bindings, `pip install kuzu`) with native Cypher support, disk-backed storage, and index-free adjacency for fast multi-hop traversal. For this use case — persisting an entity graph across sessions, querying with `MATCH` patterns, following financial flow edges up to 5 hops — KuzuDB is the right tool. NetworkX would require loading the entire graph into RAM on every agent startup and re-implementing Cypher's `MATCH` semantics in Python code.

The embedded model is important for air-gap compliance. KuzuDB runs inside the Python process; no server, no network socket.

If we needed to run PageRank over a subgraph returned by a Cypher query, the right pattern would be to use both: KuzuDB fetches the subgraph, NetworkX runs the algorithm. We don't currently need that, so NetworkX isn't a dependency.

### ChromaDB for Vector Storage

ChromaDB is embedded, already a project dependency, and supports metadata filtering — which is how entity anchoring is implemented. When the graph returns entity names, we pass them as a `where` filter to ChromaDB so the vector search is restricted to chunks associated with those entities. Without metadata filtering support, entity anchoring would require a post-retrieval filter over all results, which is much less efficient.

The embedding model is `nomic-embed-text` via Ollama (768 dimensions). It performs well on factual and technical text and requires no separate download if Ollama is already running.

Transactions are not embedded. They're structured records best queried by Cypher. Embedding them would add noise to semantic search results and waste storage.

### Data Flow Across Stores

Each store has a distinct role and a clear write path:

| Store | Contents | Source | Written by |
|-------|----------|--------|------------|
| **SQLite** | Raw extracted entities, relations, transactions, comms — exactly as parsed | `documents.json`, `transactions.json`, `comms_log.json` | ETL pipeline |
| **KuzuDB** | Graph of entity nodes and typed edges — deduplicated, alias-resolved, cross-linked | SQLite | `kuzu_store.sync()` |
| **ChromaDB** | Embedded text chunks for semantic search — document passages and comms bodies | `documents.json`, `comms_log.json` | `vector_store.index_*()` |

SQLite is the raw write target — ETL writes there and nothing else does. KuzuDB and ChromaDB are both built from SQLite in a second pass. This separation means ETL can be re-run to pick up new source documents without touching the graph or vector stores, and the graph can be re-synced (e.g. after improving the alias resolver) without re-running the expensive LLM extraction step.

---

## Deterministic vs. AI Boundaries

A deliberate design principle: use deterministic code everywhere the answer is computable without language understanding. Only call the LLM when language understanding is genuinely required.

| Step | Approach | Reason |
|------|----------|--------|
| Structured field extraction (account IDs, timestamps, amounts) | Regex | 100% deterministic, no hallucination risk |
| Entity and relationship extraction from unstructured text | Local LLM | Requires language understanding; no rule set can generalise |
| Query classification (structured / semantic / hybrid) | Keyword heuristics | Fast, auditable, overridable by agent |
| Entity spotting in analyst queries | String matching + normalisation | Exact match is more reliable than asking the LLM to name entities |
| Graph traversal and money path BFS | Cypher + Python BFS | Graph algorithms don't benefit from language models |
| Final answer synthesis | Local LLM | Requires reasoning over heterogeneous evidence |

The agent is explicitly constrained: it cannot generate entity names or facts from parametric memory. Every claim in the final answer must cite a source — a document ID, transaction ID, or event ID — that came from a tool call. This is enforced by the system prompt and by passing all retrieved evidence as tool output rather than injecting it into the prompt directly.

---

## Production Scaling

The current stack — KuzuDB embedded, ChromaDB embedded, a single Ollama process — is deliberately simple. It runs on one machine with no external services, which suits the air-gapped prototype well. At production scale (millions of transactions, thousands of documents, concurrent analysts) several things would need to change.

**Graph database.** KuzuDB's embedded model means one writer at a time and the database lives on a single node. For large-scale persistent graph storage with concurrent analyst sessions you'd move to a server-mode graph DB — Neo4j or Amazon Neptune depending on whether on-prem or cloud deployment is acceptable. The Cypher queries would transfer directly to Neo4j. The entity-anchoring retrieval logic stays the same; only the connection layer changes.

**Vector store.** ChromaDB embedded doesn't support distributed search. At millions of document chunks you'd move to a dedicated vector database — Qdrant or Weaviate — both of which support HNSW approximate nearest neighbour search, metadata filtering (which the entity-anchoring relies on), and horizontal scaling. The retriever interface is thin enough that swapping the backend is a one-file change.

**LLM inference.** The system already supports vLLM via the `--base-url` flag — `llm_client.py` speaks to any OpenAI-compatible endpoint, so switching from Ollama to vLLM requires no code changes. At production scale the step up is running vLLM as a multi-GPU cluster with continuous batching and tensor parallelism, which handles concurrent analyst sessions and makes larger quantised models (70B) viable. That meaningfully improves extraction quality on complex documents without touching the application layer.

**ETL at scale.** Extracting entities from millions of documents with a single LLM process would take days. The natural approach is to parallelise across a job queue (Celery, Ray, or a simple thread pool against a vLLM endpoint) where each worker processes one document and writes results to a shared store. The extraction logic is stateless per document, so parallelisation is straightforward.

**Reranking.** The current retriever returns the top-5 results per collection ranked by vector distance — fast, but cosine similarity is a weak relevance signal when chunks are long or queries are ambiguous. At scale, a cross-encoder reranker (e.g. `ms-marco-MiniLM-L-6-v2`) would take the top-k vector candidates and re-score them by full query-document relevance before the LLM sees them. Cross-encoders are too slow to run over the entire corpus but are fast enough over a small candidate set — the typical pattern is ANN retrieval for recall, cross-encoder for precision. For an air-gapped deployment the model runs locally via sentence-transformers, no new infrastructure needed.

**Caching.** In the current prototype, `all_entity_names()` is cached in memory at retriever init, but entity subgraph lookups and embedding calls repeat on every query. At scale, a Redis layer in front of both would eliminate redundant Cypher traversals and re-embeddings for popular entities. Subgraphs are particularly good candidates — the graph changes infrequently relative to how often analysts query the same entity.

**Entity resolution.** The current normalisation approach (stripping legal suffixes, exact string matching) handles simple aliasing. At scale with noisy source data you'd need a proper entity resolution pipeline: candidate blocking by token overlap, pairwise similarity scoring, and a merge step with a human review queue for borderline cases. This is the component that degrades most visibly as data volume and source diversity grow.

---

## Design Decisions — Clarifications

Three architectural questions worth addressing explicitly.

**Query-time entity spotting** uses three passes: exact word-boundary match, legal-suffix normalised match, and fuzzy `partial_ratio` match via `thefuzz` to handle typos (e.g. "Markus Vane" resolves to "Marcus Vane"). `partial_ratio` is the right function here — it finds the best matching window of the entity name's length within the full query string, which is the correct comparison when a short entity name is being searched in a longer sentence. `token_sort_ratio` (used in the alias resolver) is better suited to same-length string comparisons like two company name variants. Single-initial abbreviations like "David O." are a known limitation — catching those reliably requires a NER tagger with coreference resolution, which a 7B local model cannot provide deterministically. The three-pass approach catches the realistic failure modes (minor typos, dropped legal suffixes, reordered tokens) without introducing false positives from overly loose matching.

**KuzuDB schema drift from LLM-extracted relation types** is not a problem in this design because we deliberately chose a single `RELATION` table with `relation_type` stored as a string property — rather than one typed REL table per relationship (e.g. a separate `BENEFICIAL_OWNER_OF` table). Any string the LLM produces is stored without a schema declaration. The trade-off is that Cypher pattern matching on a specific relation type (`WHERE r.relation_type = 'DIRECTOR_OF'`) is a property filter rather than a schema constraint, which is slightly less efficient but completely tolerant of LLM output variance. No normalisation layer is needed because there is no typed schema to violate.

**Raw Cypher in the agent tool list** is an escape hatch, not the primary interface. The `retrieve()` and `trace_money()` tools are fully parametric — they call pre-validated Python functions that execute Cypher templates internally. `retrieve()` maps to `entity_subgraph()` and `vector_store.search()`; `trace_money()` maps to the BFS traversal in `money_path()`. The LLM never writes Cypher for these. The `cypher()` tool exists for edge cases where an analyst needs a precise lookup that the parametric tools don't cover — it is intentionally last in the tool list and the system prompt steers the model toward the parametric tools first. In practice, Qwen2.5:7b rarely reaches for raw Cypher.

---

## What This System Intentionally Doesn't Do

The agent won't speculate beyond retrieved evidence. If a money path is incomplete — a transaction leads to a terminal account with no outgoing edges — the agent says so rather than inferring what probably happened next.

Entity resolution is conservative: we merge entities only when there's a structural signal (same account number, explicit alias relationship). We don't merge "Shell Corp IO" and "Shell Corp" on name similarity alone, because in an intelligence context, treating two separate entities as one is a worse failure mode than keeping them separate.

The system has no feedback loop. Analyst corrections to wrong answers don't update the graph. That's a deliberate boundary for this prototype — a production system would need a human-in-the-loop correction workflow, but adding it here would be out of scope and would need careful governance design.
