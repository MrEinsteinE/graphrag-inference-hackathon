from __future__ import annotations

import time
from typing import Any

from graphrag_benchmark.config import Settings, get_settings
from graphrag_benchmark.llm_client import count_tokens
from graphrag_benchmark.models import PipelineResult


def _method_params(method: str) -> dict[str, Any]:
    m = method.lower()
    if m == "hybrid":
        return {
            "indices": ["DocumentChunk", "Entity"],
            "top_k": 5,
            "num_hops": 2,
            "num_seen_min": 3,
            "verbose": True,
        }
    if m == "similarity":
        return {"index": "DocumentChunk", "top_k": 5, "withHyDE": False, "verbose": True}
    if m == "contextual":
        return {
            "index": "DocumentChunk",
            "top_k": 5,
            "lookahead": 3,
            "lookback": 3,
            "withHyDE": False,
            "verbose": True,
        }
    if m == "community":
        return {"community_level": 2, "top_k": 3, "verbose": True}
    raise ValueError(f"Unknown GRAPHRAG_SEARCH_METHOD: {method}")


def _estimate_tokens_from_verbose(
    resp: dict[str, Any], answer: str, chat_model: str
) -> tuple[int | None, int | None]:
    """GraphRAG does not always return LLM usage; approximate from retrieved context + answer."""
    verbose = resp.get("verbose")
    prompt_hint = None
    if verbose is not None:
        try:
            prompt_hint = count_tokens(str(verbose), chat_model)
        except Exception:
            prompt_hint = None
    ct = count_tokens(answer, chat_model)
    return prompt_hint, ct


def run_graphrag(
    query: str,
    *,
    settings: Settings | None = None,
) -> PipelineResult:
    s = settings or get_settings()
    t0 = time.perf_counter()
    try:
        from pyTigerGraph import TigerGraphConnection
    except ImportError:
        return PipelineResult(
            name="GraphRAG",
            answer="",
            latency_ms=(time.perf_counter() - t0) * 1000,
            prompt_tokens=None,
            completion_tokens=None,
            error="Install pyTigerGraph: pip install pyTigerGraph",
        )

    try:
        conn = TigerGraphConnection(
            host=s.tigergraph_host,
            username=s.tigergraph_username,
            password=s.tigergraph_password,
            gsPort=s.tigergraph_gs_port,
            restppPort=s.tigergraph_restpp_port,
            graphname=s.tigergraph_graphname,
        )
        conn.ai.configureGraphRAGHost(s.graphrag_host)
        try:
            conn.getToken()
        except Exception:
            pass

        method = s.graphrag_search_method
        params = _method_params(method)
        resp = conn.ai.answerQuestion(query, method=method, method_parameters=params)
        answer = (resp.get("response") or "").strip()
        latency = (time.perf_counter() - t0) * 1000
        pt, ct = _estimate_tokens_from_verbose(resp, answer, s.chat_model)
        return PipelineResult(
            name="GraphRAG",
            answer=answer,
            latency_ms=latency,
            prompt_tokens=pt,
            completion_tokens=ct,
            raw={"method": method, "verbose_keys": list(resp.keys())},
        )
    except Exception as e:
        latency = (time.perf_counter() - t0) * 1000
        return PipelineResult(
            name="GraphRAG",
            answer="",
            latency_ms=latency,
            prompt_tokens=None,
            completion_tokens=None,
            error=str(e),
        )
