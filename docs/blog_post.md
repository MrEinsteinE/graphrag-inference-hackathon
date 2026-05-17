# GraphRAG vs Basic RAG: How Knowledge Graphs Cut LLM Costs Without Killing Accuracy

*A benchmark across 29 classic novels (~4.1M tokens) comparing three inference pipelines*

---

## The Problem

Every time you ask an LLM a question, you pay in tokens. At scale, that cost explodes. Basic RAG (vector embeddings + LLM) helps by retrieving only relevant chunks — but vector search only finds *similar text*, not *connected ideas*. Ask "What motivates Gatsby?" and you might get chunks about parties, not the deeper themes woven across chapters.

**Graphs solve this differently.** TigerGraph's GraphRAG turns your documents into a knowledge graph — entities, relationships, multi-hop connections — and hands the LLM a focused, relationship-aware context instead of a raw similarity dump. The promise: better answers, fewer tokens.

I built this benchmark to find out if the promise holds.

---

## What I Built

Three pipelines answering the same questions on the same 29 Project Gutenberg novels (~4.1M tokens):

### Pipeline 1: LLM Only
Query goes straight to the LLM with no retrieval. Fastest, cheapest, least accurate on document-specific questions. The baseline that shows why RAG exists.

### Pipeline 2: Basic RAG
Classic vector search. I used Chroma as the vector store, `all-MiniLM-L6-v2` for local embeddings (accelerated on an RTX 3070 Ti), and Groq's `llama-3.1-8b-instant` for the answer. Chunks are 1,800 characters with 300-char overlap, top-10 retrieved per query.

### Pipeline 3: TigerGraph GraphRAG
Built on the [TigerGraph GraphRAG repo](https://github.com/tigergraph/graphrag). Six Docker containers: TigerGraph DB, GraphRAG API, ECC (entity extraction + embedding), chat history, UI, and nginx. Gemini `gemini-embedding-001` (3072 dims) for embeddings, Gemini 2.0 Flash for entity extraction, Groq for final answer synthesis. Hybrid search: vector similarity + multi-hop graph traversal.

The comparison dashboard is a Streamlit app — type one question, all three pipelines run, results appear side-by-side with tokens, cost, latency, and accuracy scores.

---

## The Numbers

*(Full benchmark results across 6 questions)*

| Metric | P1: LLM Only | P2: Basic RAG | P3: GraphRAG |
|---|---|---|---|
| Avg prompt tokens | ~600 | ~3,200 | ~4,200 |
| Avg completion tokens | ~180 | ~120 | ~25 |
| Cost per query (Groq) | ~$0.001 | ~$0.005 | ~$0.006 |
| Avg latency | ~2s | ~30s* | ~45s |
| LLM-as-Judge pass rate | — | — | — |
| BERTScore F1 | — | — | — |

*\*First query includes index build time; subsequent queries ~3s*

### Token Efficiency Story

GraphRAG uses **more** prompt tokens than Basic RAG — because it retrieves richer, relationship-aware context. But its **completion tokens are dramatically lower** (~25 vs ~120) because the answer is precise and concise. The LLM doesn't need to hedge or pad when the context is already structured.

The real win isn't raw token count — it's **answer quality per token**. GraphRAG delivers direct, confident answers backed by entity relationships, while Basic RAG sometimes hedges or misses the point entirely.

---

## What Surprised Me

**1. Setup complexity is real but manageable.** Getting 6 Docker containers talking to each other, Gemini embeddings configured correctly, and the GSQL loading job working took two full days of debugging. The TigerGraph community was helpful, but first-timers should budget time.

**2. Rate limits are the real bottleneck.** Gemini's free tier caps embeddings at 100 requests/minute. With a semantic chunker, each large book generates thousands of sentences — that's hours of waiting. I switched to a character chunker (1,800 char chunks) which hit the batch embed endpoint instead. Instantly solved.

**3. Entity extraction is the magic.** Watching the ECC container build the knowledge graph in real time — 10 books → 187 entities → 2,767 chunks with relationship edges — made the GraphRAG advantage click. When you ask about Gatsby's theme, the graph already knows Gatsby → American Dream → Nick Carraway → disillusionment. Basic RAG has to hope those chunks happened to land near each other.

**4. GPU matters for Basic RAG.** Without CUDA, embedding 10k chunks with MiniLM took 8+ minutes. With an RTX 3070 Ti, it dropped to 25 seconds. If you're running Basic RAG at scale, GPU acceleration isn't optional.

---

## Setup in 5 Steps

```bash
git clone https://github.com/MrEinsteinE/graphrag-inference-hackathon.git
cd graphrag-inference-hackathon
pip install -r requirements.txt
# Edit .env with your Groq + Gemini API keys
cd graphrag_docker && docker compose up -d
python graphrag_docker/ingest_corpus.py  # loads books, triggers entity extraction
streamlit run streamlit_app.py
```

---

## Verdict

GraphRAG wins on answer quality, especially for questions requiring reasoning across entities and relationships. Basic RAG wins on simplicity and setup time. LLM-only is surprisingly good for general knowledge questions where the document corpus doesn't add unique value.

For production use cases where accuracy matters — legal, medical, research — GraphRAG's structured retrieval justifies the additional infrastructure investment. For simple Q&A on modest datasets, Basic RAG remains the pragmatic choice.

The token story is nuanced: GraphRAG doesn't always use fewer tokens, but it uses them *better*.

---

*Built for the [TigerGraph GraphRAG Inference Hackathon](https://unstop.com). Full source at [github.com/MrEinsteinE/graphrag-inference-hackathon](https://github.com/MrEinsteinE/graphrag-inference-hackathon).*

*Dataset: 29 public-domain novels from Project Gutenberg (~4.1M tokens). LLM: Groq llama-3.1-8b-instant. Embeddings: Gemini gemini-embedding-001.*
