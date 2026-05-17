# Architecture Diagram

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Streamlit Dashboard (localhost:8501)                  │
│                   One query → 3 pipelines run → side-by-side results        │
└──────────────┬──────────────────────┬───────────────────────┬───────────────┘
               │                      │                        │
               ▼                      ▼                        ▼
┌──────────────────────┐ ┌────────────────────────┐ ┌─────────────────────────┐
│   Pipeline 1         │ │   Pipeline 2            │ │   Pipeline 3            │
│   LLM Only           │ │   Basic RAG             │ │   TigerGraph GraphRAG   │
│                      │ │                         │ │                         │
│  Query ──► Groq LLM  │ │  Query                  │ │  Query                  │
│           (llama-3.1)│ │    │                    │ │    │                    │
│               │      │ │    ▼                    │ │    ▼                    │
│               ▼      │ │  MiniLM Embeddings      │ │  pyTigerGraph           │
│           Answer     │ │  (RTX 3070 Ti GPU)      │ │  answerQuestion()       │
│                      │ │    │                    │ │    │                    │
│  Tokens: ~500-800    │ │    ▼                    │ │    ▼                    │
│  No retrieval        │ │  Chroma Vector Store    │ │  TigerGraph DB          │
│                      │ │  (10k chunks, 1800c)    │ │  ┌─────────────────┐   │
└──────────────────────┘ │    │                    │ │  │ Knowledge Graph  │   │
                         │    ▼                    │ │  │ 187 entities     │   │
                         │  Top-10 Chunks          │ │  │ 2,767 chunks     │   │
                         │    │                    │ │  │ 10 documents     │   │
                         │    ▼                    │ │  └─────────────────┘   │
                         │  Groq LLM (llama-3.1)   │ │    │                    │
                         │    │                    │ │    ▼                    │
                         │    ▼                    │ │  Hybrid Search          │
                         │  Answer                 │ │  (vector + graph hops)  │
                         │                         │ │    │                    │
                         │  Tokens: ~2000-4000     │ │    ▼                    │
                         │  Vector similarity      │ │  Groq LLM (llama-3.1)  │
                         └────────────────────────┘ │    │                    │
                                                     │    ▼                    │
                                                     │  Answer                 │
                                                     │                         │
                                                     │  Tokens: ~3000-6000     │
                                                     │  Multi-hop reasoning    │
                                                     └─────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         Accuracy Evaluation Layer                            │
│                                                                              │
│   LLM-as-Judge (Groq llama-3.1)          BERTScore (local, GPU)             │
│   PASS / FAIL verdict                    F1 semantic similarity score        │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         TigerGraph Docker Stack                              │
│                                                                              │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │  tigergraph  │  │   graphrag   │  │  graphrag-   │  │   chat-history   │ │
│  │  :14240      │  │   :8000      │  │   ecc :8001  │  │   :8002          │ │
│  │  (graph db)  │  │   (API)      │  │  (chunking + │  │  (conversation   │ │
│  └─────────────┘  └──────────────┘  │  embedding + │  │   storage)       │ │
│                                      │  entity ext) │  └──────────────────┘ │
│  ┌─────────────┐  ┌──────────────┐  └──────────────┘                        │
│  │  graphrag-   │  │    nginx     │                                          │
│  │  ui :3000    │  │    :80       │  Embeddings: Gemini gemini-embedding-001 │
│  │  (web UI)    │  │  (reverse    │  Entity extraction: Gemini 2.0 Flash     │
│  └─────────────┘  │   proxy)     │  Completion: Groq llama-3.1-8b-instant   │
│                    └──────────────┘                                          │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              Dataset                                         │
│   29 Project Gutenberg public-domain books · ~4.1M tokens                   │
│   Alice, Dracula, Frankenstein, Great Gatsby, Moby Dick, Pride & Prejudice  │
│   Sherlock Holmes, Ulysses, Jane Eyre, Great Expectations, and more         │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow for a Single Query

1. User types a question in the Streamlit dashboard
2. Dashboard fires all 3 pipelines concurrently
3. **Pipeline 1**: Query → Groq API → answer (no retrieval)
4. **Pipeline 2**: Query → MiniLM embed (GPU) → Chroma ANN search → top-10 chunks → Groq API → answer
5. **Pipeline 3**: Query → pyTigerGraph `answerQuestion(hybrid)` → TigerGraph multi-hop graph traversal + vector search → focused context → Groq API → answer
6. Each answer is optionally scored by LLM-as-judge + BERTScore
7. Dashboard renders tokens, cost, latency, and accuracy side-by-side
