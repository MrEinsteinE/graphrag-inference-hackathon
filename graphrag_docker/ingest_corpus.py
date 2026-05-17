"""
Ingest Gutenberg corpus into TigerGraph GraphRAG.
Run from the project root with the .venv active:
  .venv/Scripts/python graphrag_docker/ingest_corpus.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from pyTigerGraph import TigerGraphConnection

TG_HOST = "http://localhost"
TG_PORT = "14240"
TG_USER = "tigergraph"
TG_PASS = "tigergraph"
GRAPHNAME = "TigerGraphRAG"
GRAPHRAG_HOST = "http://localhost:8000"

PRIORITY_BOOKS = {
    "alice_in_wonderland_carroll.txt",
    "dracula_stoker.txt",
    "frankenstein_shelley.txt",
    "great_gatsby_fitzgerald.txt",
    "pride_and_prejudice_austen.txt",
    "sherlock_adventures_doyle.txt",
    "study_in_scarlet_doyle.txt",
    "moby_dick_melville.txt",
    "romeo_and_juliet_shakespeare.txt",
    "jekyll_hyde_stevenson.txt",
}

CORPUS_DIR = Path(__file__).resolve().parents[1] / "data" / "corpus"
JSONL_PATH = Path(__file__).parent / "corpus_for_graphrag.jsonl"
JSONL_CONTAINER_PATH = "/data/corpus_for_graphrag.jsonl"


def build_jsonl():
    """Convert priority .txt files to JSONL matching TigerGraph loading job schema.

    Required fields: doc_id, doc_type, content
    """
    docs = []
    for root_dir in [CORPUS_DIR, CORPUS_DIR / "gutenberg"]:
        if not root_dir.is_dir():
            continue
        for p in sorted(root_dir.glob("*.txt")):
            if p.name.upper().startswith("GUTENBERG_ATTRIBUTION"):
                continue
            if PRIORITY_BOOKS and p.name not in PRIORITY_BOOKS:
                continue
            text = p.read_text(encoding="utf-8", errors="replace").strip()
            if not text:
                continue
            doc_id = p.stem.lower().replace("_", "-")
            title = p.stem.replace("_", " ").title()
            docs.append({"doc_id": doc_id, "doc_type": "character", "content": text})
            print(f"  + {title} ({len(text):,} chars)")

    with JSONL_PATH.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")
    print(f"\nWrote {len(docs)} documents to {JSONL_PATH}")
    return docs


def wait_for_tg(conn: TigerGraphConnection, retries: int = 20):
    for i in range(retries):
        try:
            conn.echo()
            print("TigerGraph connection OK")
            return
        except Exception as e:
            print(f"  waiting for TigerGraph ({i+1}/{retries})... {e}")
            time.sleep(5)
    raise RuntimeError("TigerGraph did not become ready in time")


def create_graph(conn: TigerGraphConnection):
    try:
        result = conn.gsql(f"CREATE GRAPH {GRAPHNAME}()")
        print("Graph created:", result[:80])
    except Exception as e:
        if "already exists" in str(e).lower() or "exist" in str(e).lower():
            print("Graph already exists, continuing.")
        else:
            raise


def initialize_graphrag(conn: TigerGraphConnection):
    print("Initializing GraphRAG schema...")
    conn.ai.configureGraphRAGHost(GRAPHRAG_HOST)
    try:
        result = conn.ai.initializeGraphRAG()
        print("GraphRAG initialized (schema + queries ready)")
    except Exception as e:
        if "already" in str(e).lower() or "exist" in str(e).lower():
            print("GraphRAG schema already initialized, continuing.")
        else:
            raise


def run_loading_job(conn: TigerGraphConnection):
    """Run the TigerGraph loading job directly via GSQL, bypassing pyTigerGraph's
    runDocumentIngest which generates incorrect $datasource: syntax."""
    print("Running GSQL loading job...")
    gsql_cmd = (
        f"USE GRAPH {GRAPHNAME}\n"
        f'RUN LOADING JOB -noprint load_documents_content_json '
        f'USING DocumentContent="{JSONL_CONTAINER_PATH}"'
    )
    try:
        result = conn.gsql(gsql_cmd)
        print("Loading job result:", result[:200] if result else "(no output)")
    except Exception as e:
        err = str(e)
        # pyTigerGraph raises on "Using graph" prefix — check if job actually started
        if "running the following loading job" in err.lower() or "jobid" in err.lower():
            print("Loading job started (background):", err[:200])
        else:
            raise


def wait_for_documents(conn: TigerGraphConnection, expected: int, timeout: int = 120):
    """Poll until documents appear in the graph."""
    print(f"Waiting for {expected} documents to load...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            count = conn.getVertexCount("Document")
            print(f"  Documents in graph: {count}")
            if count >= expected:
                return count
        except Exception:
            pass
        time.sleep(5)
    count = conn.getVertexCount("Document")
    print(f"  Final document count: {count}")
    return count


def force_consistency(conn: TigerGraphConnection):
    print("\nTriggering GraphRAG consistency update (entity extraction + graph build)...")
    print("  This calls Gemini to extract entities — expect several minutes...")
    conn.ai.forceConsistencyUpdate("graphrag")
    print("Consistency update triggered (runs asynchronously in ECC container).")


def test_query(conn: TigerGraphConnection):
    print("\nRunning test query...")
    try:
        resp = conn.ai.answerQuestion(
            "Who wrote Alice in Wonderland?",
            method="hybrid",
            method_parameters={
                "indices": ["DocumentChunk", "Entity"],
                "top_k": 5,
                "num_hops": 2,
                "num_seen_min": 3,
                "verbose": False,
            },
        )
        print("Answer:", resp.get("response", "—"))
    except Exception as e:
        print("Query error (ECC may still be processing):", e)


if __name__ == "__main__":
    print("=== Building JSONL corpus ===")
    build_jsonl()

    conn = TigerGraphConnection(
        host=TG_HOST,
        username=TG_USER,
        password=TG_PASS,
        restppPort=TG_PORT,
        gsPort=TG_PORT,
    )
    conn.graphname = GRAPHNAME

    print("\n=== Connecting to TigerGraph ===")
    wait_for_tg(conn)

    print("\n=== Creating graph ===")
    create_graph(conn)

    print("\n=== Initializing GraphRAG ===")
    initialize_graphrag(conn)

    print("\n=== Loading documents into TigerGraph ===")
    run_loading_job(conn)
    doc_count = wait_for_documents(conn, expected=len(PRIORITY_BOOKS))

    if doc_count == 0:
        print("WARNING: No documents loaded — check JSONL format or file path.")
    else:
        print(f"\n{doc_count} documents loaded successfully.")
        print("\n=== Triggering GraphRAG processing ===")
        force_consistency(conn)
        print("\n=== Test query ===")
        test_query(conn)

    print("\nDone!")
    print("ECC container continues processing in background (chunking + embedding + entity extraction).")
    print("Monitor: docker logs graphrag-ecc -f")
    print(f"GraphRAG UI: http://localhost:80")
    print("Streamlit: .venv/Scripts/streamlit run streamlit_app.py")
