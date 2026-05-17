# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
# Edit .env — at minimum set OPENAI_API_KEY (Groq key works) and OPENAI_BASE_URL
```

## Running the app

```powershell
streamlit run streamlit_app.py
```

## Fetch corpus (required for Basic RAG)

```powershell
python scripts\fetch_gutenberg_corpus.py
```

## Reset the Basic RAG vector store

Delete `data/chroma_basic_rag/` to force a full rebuild on next run.

## Architecture

Three inference pipelines are benchmarked side-by-side, all returning `PipelineResult`:

| Module | Pipeline |
|---|---|
| `graphrag_benchmark/pipelines/llm_only.py` | Pipeline 1 — bare LLM, no retrieval |
| `graphrag_benchmark/pipelines/basic_rag.py` | Pipeline 2 — Chroma + sentence-transformers (local MiniLM by default, or OpenAI embeddings if `EMBEDDING_MODEL` is set) |
| `graphrag_benchmark/pipelines/graphrag_tg.py` | Pipeline 3 — TigerGraph GraphRAG via `pyTigerGraph.answerQuestion` |

**Key modules:**
- `graphrag_benchmark/config.py` — `get_settings()` (LRU-cached, reads `.env`); call `get_settings.cache_clear()` before re-reading env at runtime.
- `graphrag_benchmark/models.py` — `PipelineResult` and `AccuracyResult` dataclasses shared across all pipelines and the UI.
- `graphrag_benchmark/llm_client.py` — shared OpenAI-compatible client (used by pipelines 1 & 2).
- `graphrag_benchmark/eval/accuracy.py` — `evaluate_accuracy()`: runs LLM-as-judge (HF Inference API or same chat model) and BERTScore.
- `graphrag_benchmark/torch_device.py` — resolves `auto | cuda | mps | cpu` for local embeddings and BERTScore.
- `streamlit_app.py` — dashboard; calls all three pipelines sequentially, then optional accuracy eval.

## Environment variables

See `.env.example` for the full list. Key variables:

- `OPENAI_API_KEY` + `OPENAI_BASE_URL` + `CHAT_MODEL` — LLM for pipelines 1 & 2 (defaults to Groq).
- `EMBEDDING_MODEL` — leave blank to use local MiniLM; set to an OpenAI embedding model ID to use the API instead.
- `TIGERGRAPH_*` + `GRAPHRAG_HOST` + `GRAPHRAG_SEARCH_METHOD` — required only for Pipeline 3.
- `HUGGINGFACEHUB_API_TOKEN` + `HF_JUDGE_MODEL` — optional; falls back to `OPENAI_API_KEY` for judging.
- `BENCHMARK_BASIC_RAG_MAX_CHUNKS` — cap index size for faster dev (2500 is a good default).
- `BENCHMARK_EMBED_DEVICE` — `auto | cuda | mps | cpu`.

## Notes

- `get_settings()` is `@lru_cache`'d — always call `get_settings.cache_clear()` before re-reading settings in the same process (the Streamlit app does this on each run).
- Basic RAG chunk sampling is round-robin across `.txt` files so all books are represented even with a low `max_chunks` cap.
- GraphRAG prompt token counts are approximated from the verbose retrieval payload when the service doesn't return usage data.
