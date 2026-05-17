from graphrag_benchmark.pipelines.basic_rag import run_basic_rag
from graphrag_benchmark.pipelines.graphrag_tg import run_graphrag
from graphrag_benchmark.pipelines.llm_only import run_llm_only

__all__ = ["run_llm_only", "run_basic_rag", "run_graphrag"]
