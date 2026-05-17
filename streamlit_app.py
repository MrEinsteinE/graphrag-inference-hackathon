from __future__ import annotations

from pathlib import Path

import streamlit as st

from graphrag_benchmark.config import get_settings
from graphrag_benchmark.eval.accuracy import evaluate_accuracy
from graphrag_benchmark.models import PipelineResult
from graphrag_benchmark.pipelines import run_basic_rag, run_graphrag, run_llm_only

st.set_page_config(page_title="GraphRAG Inference Benchmark", layout="wide")
st.title("GraphRAG inference benchmark")
st.caption(
    "TigerGraph hackathon: one query → LLM-only, Basic RAG, GraphRAG — tokens, latency, cost, accuracy."
)

with st.sidebar:
    st.subheader("Environment")
    st.code("Copy .env.example → .env and fill keys", language="text")
    corpus_dir = st.text_input("Corpus folder (Basic RAG)", value="data/corpus")
    top_k = st.number_input("Basic RAG top_k", min_value=1, max_value=50, value=10)
    max_chunks_ui = st.number_input(
        "Basic RAG: max chunks to index (0 = all — first run can take a long time on CPU)",
        min_value=0,
        max_value=500_000,
        value=10000,
        help="Cap total chunks for faster indexing. Chunks are taken **round-robin across books** so titles late in the alphabet (e.g. Pride and Prejudice) are still represented. Use 0 for the full corpus.",
    )
    try:
        from graphrag_benchmark.config import get_settings
        from graphrag_benchmark.torch_device import resolve_torch_device

        _dev = resolve_torch_device(get_settings().embed_device)
        st.caption(
            f"Local embed device (MiniLM / BERTScore): **{_dev}** — install CUDA PyTorch if this shows `cpu` but you have an NVIDIA GPU."
        )
    except Exception:
        pass
    include_graphrag = st.checkbox(
        "Include GraphRAG (Pipeline 3)",
        value=False,
        help="Uncheck while TigerGraph is not running to avoid long connection waits.",
    )
    ref = st.text_area("Reference answer (for accuracy)", height=120)
    run_eval = st.checkbox("Run accuracy (judge + BERTScore)", value=True)
    st.divider()
    st.markdown(
        "[TigerGraph GraphRAG](https://github.com/tigergraph/graphrag) must be running "
        "for Pipeline 3. Configure `TIGERGRAPH_*` and `GRAPHRAG_HOST` in `.env`."
    )

query = st.text_input("Question", placeholder="Ask something grounded in your corpus…")

run_clicked = st.button("Run all pipelines", type="primary")

if run_clicked and not query.strip():
    st.warning("Enter a question first.")

if run_clicked and query.strip():
    get_settings.cache_clear()
    _ = get_settings()
    cdir = Path(corpus_dir) if corpus_dir else Path("data/corpus")
    chunk_cap = None if int(max_chunks_ui) == 0 else int(max_chunks_ui)

    with st.spinner("LLM-only…"):
        r0 = run_llm_only(query)
    with st.spinner("Basic RAG (first run indexes the corpus — can take several minutes)…"):
        r1 = run_basic_rag(query, corpus_dir=cdir, top_k=int(top_k), max_chunks=chunk_cap)
    if include_graphrag:
        with st.spinner("GraphRAG (TigerGraph)…"):
            r2 = run_graphrag(query)
    else:
        r2 = PipelineResult(
            name="GraphRAG",
            answer="",
            latency_ms=0.0,
            prompt_tokens=None,
            completion_tokens=None,
            error="Skipped — enable “Include GraphRAG” in the sidebar when TigerGraph GraphRAG is running.",
        )

    ss = get_settings()
    cols = st.columns(3)
    results = [r0, r1, r2]

    for col, res in zip(cols, results):
        with col:
            st.subheader(res.name)
            if res.error:
                st.error(res.error)
            else:
                st.write(res.answer)
            st.metric("Latency (ms)", f"{res.latency_ms:.1f}")
            pt = res.prompt_tokens
            ct = res.completion_tokens
            st.metric("Tokens (prompt + completion)", f"{pt or '—'} + {ct or '—'}")
            cost = res.cost_usd(ss.price_prompt_per_mtok, ss.price_completion_per_mtok)
            st.metric("Est. cost (USD)", f"{cost:.6f}" if cost is not None else "—")
            if res.name == "GraphRAG" and res.prompt_tokens is not None:
                st.caption(
                    "GraphRAG prompt tokens are approximated from verbose retrieval payload "
                    "when usage is not returned by the service."
                )

    if run_eval and ref.strip():
        st.divider()
        st.subheader("Accuracy (vs reference)")
        acols = st.columns(3)
        for col, res in zip(acols, results):
            with col:
                st.markdown(f"**{res.name}**")
                if res.error or not res.answer:
                    st.info("Skipped")
                    continue
                acc = evaluate_accuracy(query, ref, res.answer, run_bert_score=True, run_judge=True)
                if acc.error:
                    st.warning(acc.error)
                st.write(
                    "Judge:",
                    "PASS" if acc.judge_pass is True else "FAIL" if acc.judge_pass is False else "—",
                )
                if acc.judge_raw:
                    st.caption(acc.judge_raw[:200])
                if acc.bertscore_f1 is not None:
                    st.write(
                        f"BERTScore F1: {acc.bertscore_f1:.4f} "
                        f"(P {acc.bertscore_precision:.4f}, R {acc.bertscore_recall:.4f})"
                    )
