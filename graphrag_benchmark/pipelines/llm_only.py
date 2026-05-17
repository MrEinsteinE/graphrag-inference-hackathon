from __future__ import annotations

import time

from graphrag_benchmark.config import Settings, get_settings
from graphrag_benchmark.llm_client import chat_completion
from graphrag_benchmark.models import PipelineResult


def run_llm_only(
    query: str,
    *,
    settings: Settings | None = None,
) -> PipelineResult:
    s = settings or get_settings()
    t0 = time.perf_counter()
    try:
        answer, pt, ct, raw = chat_completion(
            [{"role": "user", "content": query}],
            settings=s,
        )
        latency = (time.perf_counter() - t0) * 1000
        return PipelineResult(
            name="LLM-only",
            answer=answer,
            latency_ms=latency,
            prompt_tokens=pt,
            completion_tokens=ct,
            raw=raw,
        )
    except Exception as e:
        latency = (time.perf_counter() - t0) * 1000
        return PipelineResult(
            name="LLM-only",
            answer="",
            latency_ms=latency,
            prompt_tokens=None,
            completion_tokens=None,
            error=str(e),
        )
