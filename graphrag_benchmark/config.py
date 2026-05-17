import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


def _optional_positive_int(raw: str | None) -> int | None:
    if raw is None or str(raw).strip() == "":
        return None
    n = int(str(raw).strip())
    return n if n > 0 else None


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_base_url: str
    chat_model: str
    embedding_model: str | None
    hf_judge_model: str
    hf_token: str | None
    tigergraph_host: str
    tigergraph_username: str
    tigergraph_password: str
    tigergraph_graphname: str
    tigergraph_gs_port: str
    tigergraph_restpp_port: str
    graphrag_host: str
    graphrag_search_method: str
    price_prompt_per_mtok: float
    price_completion_per_mtok: float
    basic_rag_max_chunks: int | None
    embed_device: str


@lru_cache
def get_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
        chat_model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
        embedding_model=os.getenv("EMBEDDING_MODEL") or None,
        hf_judge_model=os.getenv("HF_JUDGE_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct"),
        hf_token=os.getenv("HUGGINGFACEHUB_API_TOKEN") or None,
        tigergraph_host=os.getenv("TIGERGRAPH_HOST", "http://127.0.0.1"),
        tigergraph_username=os.getenv("TIGERGRAPH_USERNAME", "tigergraph"),
        tigergraph_password=os.getenv("TIGERGRAPH_PASSWORD", "tigergraph"),
        tigergraph_graphname=os.getenv("TIGERGRAPH_GRAPHNAME", "TigerGraphRAG"),
        tigergraph_gs_port=os.getenv("TIGERGRAPH_GS_PORT", "14240"),
        tigergraph_restpp_port=os.getenv("TIGERGRAPH_RESTPP_PORT", "14240"),
        graphrag_host=os.getenv("GRAPHRAG_HOST", "http://127.0.0.1:8000").rstrip("/"),
        graphrag_search_method=os.getenv("GRAPHRAG_SEARCH_METHOD", "hybrid").lower(),
        price_prompt_per_mtok=float(os.getenv("PRICE_PROMPT_PER_MTOK", "0.15")),
        price_completion_per_mtok=float(os.getenv("PRICE_COMPLETION_PER_MTOK", "0.60")),
        basic_rag_max_chunks=_optional_positive_int(os.getenv("BENCHMARK_BASIC_RAG_MAX_CHUNKS")),
        embed_device=(os.getenv("BENCHMARK_EMBED_DEVICE", "auto").strip() or "auto"),
    )
