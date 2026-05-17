from __future__ import annotations

import time

import tiktoken
from openai import OpenAI

from graphrag_benchmark.config import Settings, get_settings


def _encoding_for_model(model: str) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str) -> int:
    enc = _encoding_for_model(model)
    return len(enc.encode(text))


def chat_completion(
    messages: list[dict[str, str]],
    *,
    settings: Settings | None = None,
    temperature: float = 0.2,
) -> tuple[str, int | None, int | None, dict]:
    s = settings or get_settings()
    if not s.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set")
    client = OpenAI(api_key=s.openai_api_key, base_url=s.openai_base_url)
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=s.chat_model,
        messages=messages,
        temperature=temperature,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    choice = resp.choices[0]
    text = (choice.message.content or "").strip()
    usage = resp.usage
    pt = usage.prompt_tokens if usage else None
    ct = usage.completion_tokens if usage else None
    if pt is None:
        joined = "\n".join(m["content"] for m in messages)
        pt = count_tokens(joined, s.chat_model)
    if ct is None:
        ct = count_tokens(text, s.chat_model)
    return text, pt, ct, {"elapsed_ms_api": elapsed_ms, "model": s.chat_model}
