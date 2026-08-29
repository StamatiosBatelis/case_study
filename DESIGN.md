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

The reverse works too: vector chunks carry entity metadata from ETL, so when ChromaDB returns a document excerpt, the agent knows which graph nodes to expand next.

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

The embedding model is `nomic-embed-text` via Ollama (768 dimensions). It performs well on factual and technical text and requires no separate download if Ollama is already running. The fallback is `all-MiniLM-L6-v2` via sentence-transformers.

Transactions are not embedded. They're structured records best queried by Cypher. Embedding them would add noise to semantic search results and waste storage.

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

## What This System Intentionally Doesn't Do

The agent won't speculate beyond retrieved evidence. If a money path is incomplete — a transaction leads to a terminal account with no outgoing edges — the agent says so rather than inferring what probably happened next.

Entity resolution is conservative: we merge entities only when there's a structural signal (same account number, explicit alias relationship). We don't merge "Shell Corp IO" and "Shell Corp" on name similarity alone, because in an intelligence context, treating two separate entities as one is a worse failure mode than keeping them separate.

The system has no feedback loop. Analyst corrections to wrong answers don't update the graph. That's a deliberate boundary for this prototype — a production system would need a human-in-the-loop correction workflow, but adding it here would be out of scope and would need careful governance design.
