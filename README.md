# GraphRAG Inference Hackathon — Token Efficiency Benchmark

**TigerGraph GraphRAG Inference Hackathon submission** benchmarking three inference pipelines side-by-side on 29 Project Gutenberg classic novels (~4.1M tokens):

| Pipeline | Method | Avg Tokens | Key Tech |
|---|---|---|---|
| **P1 — LLM Only** | No retrieval | ~600 | Groq llama-3.1-8b-instant |
| **P2 — Basic RAG** | Vector similarity | ~3,500 | Chroma + MiniLM (GPU) |
| **P3 — GraphRAG** | Graph + hybrid search | ~4,200 | TigerGraph + Gemini embeddings |

GraphRAG delivers the highest accuracy by reasoning across entity relationships — not just similar text chunks.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Streamlit Dashboard (:8501)                 │
│        One query → 3 pipelines → side-by-side           │
└────────┬──────────────────┬──────────────────┬──────────┘
         │                  │                  │
         ▼                  ▼                  ▼
   LLM Only           Basic RAG          GraphRAG
   Groq API         Chroma+MiniLM    TigerGraph Docker
   No retrieval      GPU embed        Knowledge graph
                     top-10 chunks    multi-hop reasoning
         │                  │                  │
         └──────────────────┴──────────────────┘
                            │
                ┌───────────▼────────────┐
                │   Accuracy Evaluation   │
                │  LLM-as-Judge + BERTScore│
                └─────────────────────────┘
```

See [`docs/architecture.md`](docs/architecture.md) for the full detailed diagram.

## Dataset

**29 public-domain novels** from Project Gutenberg — ~4.1M tokens:

Alice in Wonderland, Dracula, Frankenstein, The Great Gatsby, Moby Dick, Pride and Prejudice, Sherlock Holmes Adventures, Study in Scarlet, Memoirs of Sherlock Holmes, Jane Eyre, Great Expectations, Ulysses, Wuthering Heights, Jekyll & Hyde, Romeo & Juliet, Emma, Treasure Island, Oliver Twist, The Awakening, Picture of Dorian Gray, Room with a View, Hunchback of Notre Dame, Metamorphosis, The Yellow Wallpaper, Enchanted April, A Modest Proposal, Poetics, and more.

## Quick Start

### Prerequisites
- Python 3.10+
- Docker Desktop
- NVIDIA GPU (optional but recommended for Basic RAG embedding speed)
- Groq API key (free at [console.groq.com](https://console.groq.com))
- Google Gemini API key (free at [aistudio.google.com](https://aistudio.google.com))

### 1. Clone & Install

```powershell
git clone https://github.com/MrEinsteinE/graphrag-inference-hackathon.git
cd graphrag-inference-hackathon
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

For GPU acceleration (NVIDIA):
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 2. Configure Environment

```powershell
copy .env.example .env
```

Edit `.env` and set:
- `OPENAI_API_KEY` — your Groq API key
- `GOOGLE_API_KEY` — your Gemini API key (for GraphRAG entity extraction + embeddings)

### 3. Fetch Corpus

```powershell
python scripts\fetch_gutenberg_corpus.py
```

### 4. Start TigerGraph GraphRAG Stack

```powershell
cd graphrag_docker
docker compose up -d
cd ..
```

Wait ~2 minutes for TigerGraph to initialize, then ingest documents:

```powershell
.\.venv\Scripts\python graphrag_docker\ingest_corpus.py
```

This loads 10 books into TigerGraph, triggers entity extraction (runs ~10-15 min in background via Gemini).

### 5. Run the Dashboard

```powershell
streamlit run streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) — type any question, click Run, see all 3 pipelines answer side-by-side with token counts, cost, latency, and accuracy scores.

## Project Structure

```
graphrag-inference-hackathon/
├── streamlit_app.py                    # Main comparison dashboard
├── graphrag_benchmark/
│   ├── pipelines/
│   │   ├── llm_only.py                # Pipeline 1: bare LLM
│   │   ├── basic_rag.py               # Pipeline 2: Chroma + MiniLM
│   │   └── graphrag_tg.py             # Pipeline 3: TigerGraph GraphRAG
│   ├── eval/
│   │   └── accuracy.py                # LLM-as-judge + BERTScore
│   ├── config.py                      # Settings (LRU-cached)
│   ├── llm_client.py                  # Shared OpenAI-compatible client
│   └── models.py                      # PipelineResult, AccuracyResult
├── graphrag_docker/
│   ├── docker-compose.yml             # 6-container TigerGraph stack
│   ├── configs/server_config.json     # GraphRAG LLM + embedding config
│   └── ingest_corpus.py              # Document ingestion script
├── scripts/
│   └── fetch_gutenberg_corpus.py      # Downloads Gutenberg books
├── data/corpus/                        # 29 plaintext books (~4.1M tokens)
├── docs/architecture.md               # Full architecture diagram
└── .env.example                        # Environment variable template
```

## Metrics Tracked

For every query across all 3 pipelines:
- **Prompt tokens** — input context size
- **Completion tokens** — output size
- **Cost per query** — calculated from token counts × provider pricing
- **Latency (ms)** — end-to-end response time
- **LLM-as-Judge** — PASS/FAIL verdict from a judge LLM
- **BERTScore F1** — semantic similarity to reference answer

## Key Design Decisions

- **Groq** as the LLM provider for Pipelines 1 & 2 (free tier, fast inference)
- **Gemini** (`gemini-embedding-001`, 3072 dims) for GraphRAG's internal embeddings — best quality for entity/relationship embedding
- **Character chunker** (1800 chars, 300 overlap) for Basic RAG — avoids per-sentence embedding calls that hit free-tier rate limits
- **GPU acceleration** (`BENCHMARK_EMBED_DEVICE=cuda`) — RTX 3070 Ti reduces Basic RAG index build from ~8 min → ~25 sec
- **Hybrid search** in GraphRAG — combines vector similarity + graph traversal for the best recall

## Built On

- [TigerGraph GraphRAG](https://github.com/tigergraph/graphrag) — Pipeline 3 backend
- [pyTigerGraph](https://github.com/tigergraph/pyTigerGraph) — Python client
- [Chroma](https://github.com/chroma-core/chroma) — Vector store for Basic RAG
- [sentence-transformers](https://github.com/UKPLab/sentence-transformers) — Local MiniLM embeddings
- [Streamlit](https://streamlit.io) — Dashboard framework
- [bert-score](https://github.com/Tiiiger/bert_score) — Accuracy evaluation
