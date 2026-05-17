from __future__ import annotations

import hashlib
import time
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from graphrag_benchmark.config import Settings, get_settings
from graphrag_benchmark.llm_client import chat_completion
from graphrag_benchmark.models import PipelineResult
from graphrag_benchmark.torch_device import resolve_torch_device

DEFAULT_CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "corpus"
CHROMA_DIR = Path(__file__).resolve().parents[2] / "data" / "chroma_basic_rag"
# Bump when index strategy changes (invalidates old Chroma collections).
INDEX_VERSION = "2026-05-17-v2"


def _chunk_text(text: str, size: int = 1800, overlap: int = 300) -> list[str]:
    if not text.strip():
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def _iter_corpus_txt_paths(corpus_dir: Path) -> list[Path]:
    if not corpus_dir.is_dir():
        return []
    return [
        p
        for p in sorted(corpus_dir.rglob("*.txt"))
        if not p.name.upper().startswith("GUTENBERG_ATTRIBUTION")
    ]


def _chunk_with_source_header(source_file: str, body: str) -> str:
    """Prefix text so embeddings align with titles named in the question (e.g. Pride and Prejudice)."""
    stem = Path(source_file).stem
    label = stem.replace("_", " ")
    return f"Document: {label}\nSource file: {source_file}\n\n{body}"


def _round_robin_chunks(
    chunks_by_file: list[list[str]],
    sources: list[str],
    max_total: int | None,
) -> tuple[list[str], list[dict]]:
    """Interleave chunk lists so every book is represented when max_total is set."""
    if not chunks_by_file:
        return [], []
    if max_total is None:
        docs: list[str] = []
        meta: list[dict] = []
        for src, chs in zip(sources, chunks_by_file):
            for j, c in enumerate(chs):
                docs.append(c)
                meta.append({"source": src, "chunk": j})
        return docs, meta

    docs = []
    meta = []
    depth = 0
    while len(docs) < max_total:
        progressed = False
        for src, chs in zip(sources, chunks_by_file):
            if len(docs) >= max_total:
                break
            if depth < len(chs):
                docs.append(chs[depth])
                meta.append({"source": src, "chunk": depth})
                progressed = True
        if not progressed:
            break
        depth += 1
    return docs, meta


def _build_index_documents(
    corpus_dir: Path,
    cap: int | None,
) -> tuple[list[str], list[dict]]:
    paths = _iter_corpus_txt_paths(corpus_dir)
    chunks_by_file: list[list[str]] = []
    sources: list[str] = []
    for p in paths:
        text = p.read_text(encoding="utf-8", errors="replace")
        ch = _chunk_text(text)
        if ch:
            chunks_by_file.append(ch)
            sources.append(p.name)
    bodies, metas = _round_robin_chunks(chunks_by_file, sources, cap)
    docs = [_chunk_with_source_header(m["source"], b) for b, m in zip(bodies, metas)]
    return docs, metas


def _load_corpus_text(corpus_dir: Path) -> str:
    if not corpus_dir.is_dir():
        return ""
    parts: list[str] = []
    for p in _iter_corpus_txt_paths(corpus_dir):
        parts.append(p.read_text(encoding="utf-8", errors="replace"))
    return "\n\n".join(parts)


def ensure_vector_store(
    corpus_dir: Path | None = None,
    *,
    settings: Settings | None = None,
    collection_name: str = "chunks",
    max_chunks: int | None = None,
) -> tuple[chromadb.Collection, str]:
    s = settings or get_settings()
    directory = corpus_dir or DEFAULT_CORPUS_DIR
    full_text = _load_corpus_text(directory)
    fingerprint = hashlib.sha256(full_text.encode("utf-8", errors="replace")).hexdigest()[:16]

    cap = max_chunks if max_chunks is not None else s.basic_rag_max_chunks
    coll_suffix = f"{cap if cap else 'all'}_{INDEX_VERSION}"

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if s.embedding_model:
        ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=s.openai_api_key,
            api_base=s.openai_base_url,
            model_name=s.embedding_model,
        )
        add_batch = 128
    else:
        dev = resolve_torch_device(s.embed_device)
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2",
            device=dev,
        )
        # Larger batches amortize GPU kernel launch; Chroma calls encode per add batch.
        if dev == "cuda":
            add_batch = 512
        elif dev == "mps":
            add_batch = 256
        else:
            add_batch = 128

    coll = client.get_or_create_collection(
        name=f"{collection_name}_{fingerprint}_{coll_suffix}",
        embedding_function=ef,
        metadata={"fingerprint": fingerprint, "max_chunks": str(cap), "index_version": INDEX_VERSION},
    )

    if coll.count() == 0 and full_text.strip():
        docs, metas = _build_index_documents(directory, cap)
        if not docs:
            return coll, full_text
        for start in range(0, len(docs), add_batch):
            part_docs = docs[start : start + add_batch]
            part_meta = metas[start : start + add_batch]
            ids = [f"c{i}" for i in range(start, start + len(part_docs))]
            coll.add(ids=ids, documents=part_docs, metadatas=part_meta)

    return coll, full_text


def run_basic_rag(
    query: str,
    *,
    corpus_dir: Path | None = None,
    top_k: int = 10,
    settings: Settings | None = None,
    max_chunks: int | None = None,
) -> PipelineResult:
    s = settings or get_settings()
    t0 = time.perf_counter()
    try:
        coll, _source = ensure_vector_store(corpus_dir, settings=s, max_chunks=max_chunks)
        if coll.count() == 0:
            raise FileNotFoundError(
                f"No chunks in vector store. Add .txt files under {DEFAULT_CORPUS_DIR}."
            )
        res = coll.query(query_texts=[query], n_results=top_k)
        docs = (res.get("documents") or [[]])[0] or []
        context = "\n\n---\n\n".join(docs)
        system = (
            "You are a helpful assistant. Answer the question using the provided context. "
            "Each context block may start with 'Document:' naming the work. "
            "Use the context as your primary source — synthesize themes, characters, and facts from it. "
            "Be concise and direct."
        )
        user = f"Context:\n{context}\n\nQuestion: {query}"
        answer, pt, ct, raw = chat_completion(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            settings=s,
        )
        latency = (time.perf_counter() - t0) * 1000
        return PipelineResult(
            name="Basic RAG",
            answer=answer,
            latency_ms=latency,
            prompt_tokens=pt,
            completion_tokens=ct,
            raw={**raw, "retrieved_chunks": len(docs)},
        )
    except Exception as e:
        latency = (time.perf_counter() - t0) * 1000
        return PipelineResult(
            name="Basic RAG",
            answer="",
            latency_ms=latency,
            prompt_tokens=None,
            completion_tokens=None,
            error=str(e),
        )
