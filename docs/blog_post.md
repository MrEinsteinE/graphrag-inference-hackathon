# When Sherlock Holmes Meets a Knowledge Graph: How We Benchmarked Three RAG Pipelines on 4.1M Tokens

> *"When an LLM confidently tells you that Sherlock Holmes met Dracula in a Victorian pub — and does it in just 600 tokens — does the efficiency even matter?"*

That question became the north star of our TigerGraph GraphRAG Inference Hackathon submission. We didn't just build a RAG system. We built a **benchmarking arena** — three inference pipelines, one question, and the numbers to prove which approach actually wins.

---

## The Problem: Everyone Picks a Pipeline, Nobody Benchmarks

If you've spent any time in the RAG ecosystem, you know the drill. Teams pick a retrieval strategy — usually vector search — ship it, and call it done. Accuracy is eyeballed. Token costs are ignored until the AWS bill arrives.

The deeper problem is the **accuracy vs. efficiency tradeoff** that nobody talks about honestly:

- **LLM-only inference** is blazing fast and dirt cheap — but it hallucinates freely on knowledge it was never trained on.
- **Basic RAG** improves grounding by retrieving similar text chunks — but it treats every document like a bag of words, blind to the relationships between characters, events, and ideas.
- **GraphRAG** reasons across a knowledge graph of entities and relationships — more expensive per query, but fundamentally more capable on complex, relationship-heavy questions.

The real question is: **by how much, and when does it matter?**

We built a system to answer that — rigorously, reproducibly, and with real data.

---

## Our Approach: Three Pipelines, One Dashboard

The core of our project is a **Streamlit dashboard** that fires any question through all three pipelines simultaneously and displays the results side-by-side — tokens used, cost, latency, and accuracy scores.

| Pipeline | Method | Avg Tokens/Query | Key Tech |
|---|---|---|---|
| **P1 — LLM Only** | No retrieval | ~600 | Groq llama-3.1-8b-instant |
| **P2 — Basic RAG** | Vector similarity search | ~3,500 | Chroma + MiniLM (GPU) |
| **P3 — GraphRAG** | Graph traversal + hybrid search | ~4,200 | TigerGraph + Gemini embeddings |

Here's what each pipeline actually does:

### Pipeline 1: LLM Only
No retrieval. No context. Just a raw prompt to `llama-3.1-8b-instant` via Groq's free-tier API. It answers from parametric memory alone. Fast, cheap, and dangerously confident about things it doesn't know.

### Pipeline 2: Basic RAG
Classic vector search. We embed the corpus with `sentence-transformers/all-MiniLM-L6-v2`, store vectors in Chroma, and retrieve the top-10 most semantically similar chunks at query time. Better than nothing — but it retrieves *similar text*, not *relevant relationships*.

### Pipeline 3: GraphRAG (TigerGraph)
This is where it gets interesting. TigerGraph's GraphRAG system extracts entities and relationships from every document using Gemini (`gemini-embedding-001`, 3072 dimensions), builds a knowledge graph, and answers queries via **hybrid search** — combining vector similarity with multi-hop graph traversal. It doesn't just find text about Sherlock Holmes. It understands that Holmes *works with* Watson, *lives at* 221B Baker Street, and *investigates* crimes involving Moriarty — and reasons across all of those connections at once.

---

## The Dataset: 29 Classics, 4.1M Tokens

We deliberately chose literary text for this benchmark. Not Wikipedia. Not product docs. **Classic novels.**

Why? Because literature is the hardest RAG target:

- **Long-range dependencies**: A character introduced in Chapter 2 may be crucial to a question about Chapter 18.
- **Ambiguous pronouns**: "He said" — which of the fourteen men in the scene?
- **Complex entity webs**: Characters have aliases, relationships evolve, allegiances shift.
- **No simple keyword matches**: You can't find the theme of isolation in *Frankenstein* by searching for the word "alone."

Our corpus includes 29 public-domain novels from Project Gutenberg:

*Alice in Wonderland, Dracula, Frankenstein, The Great Gatsby, Moby Dick, Pride and Prejudice, The Adventures of Sherlock Holmes, A Study in Scarlet, The Memoirs of Sherlock Holmes, Jane Eyre, Great Expectations, Ulysses, Wuthering Heights, The Strange Case of Dr Jekyll and Mr Hyde, Romeo and Juliet, Emma, Treasure Island, Oliver Twist, The Awakening, The Picture of Dorian Gray, A Room with a View, The Hunchback of Notre-Dame, The Metamorphosis, The Yellow Wallpaper, The Enchanted April, A Modest Proposal, Poetics, and more.*

Total: **~4.1 million tokens** of rich, relationship-heavy, deeply interconnected text.

---

## What We Measured

For every query across all three pipelines, we tracked six metrics:

1. **Prompt tokens** — How much context did the pipeline inject into the LLM?
2. **Completion tokens** — How long was the generated answer?
3. **Cost per query** — Calculated from token counts × provider pricing rates.
4. **Latency (ms)** — End-to-end wall-clock time from query to response.
5. **LLM-as-Judge** — A separate LLM evaluates the answer: PASS or FAIL against a reference answer.
6. **BERTScore F1** — Semantic similarity between the generated answer and the reference, using contextual embeddings.

We used **both** neural evaluation (BERTScore) and LLM-based evaluation because they catch different failure modes. BERTScore penalizes semantic drift. The LLM judge catches factual hallucinations that sound plausible. Together, they give a much more honest picture than either alone.

---

## Results & Key Insights

The numbers told a clear story — but with some surprises.

**Where LLM-only holds its own:** For general, commonly-known facts about famous books ("What is the opening line of Pride and Prejudice?"), the bare LLM often passes the judge. It saw this text during pre-training. 600 tokens, near-zero cost, acceptable accuracy. Hard to argue with.

**Where Basic RAG struggles:** The moment a question requires connecting information across documents or tracking how a relationship *evolves*, vector search starts failing. It retrieves the right neighborhood but misses the through-line. Questions like "How does the relationship between Victor Frankenstein and his creation change over the course of the novel?" require reasoning across dozens of non-contiguous chunks — and chunk similarity doesn't capture narrative arc.

**Where GraphRAG wins:** Multi-hop relationship questions are where TigerGraph's knowledge graph earns its token cost. When the graph already encodes that Victor *creates* the Creature, the Creature *rejects* Victor, and Victor *pursues* the Creature — the answer to a relationship question becomes a graph traversal, not a semantic guessing game. BERTScore F1 and LLM-as-Judge both showed consistent improvements for GraphRAG on these question types.

The core insight: **token efficiency is meaningless if the answer is wrong**. GraphRAG uses more tokens because it sends *better context* — structured, relationship-aware, precise — rather than a scatter of loosely-related text chunks.

---

## Engineering Challenges (And How We Solved Them)

Building this system wasn't just a matter of plugging APIs together. Here are the real engineering problems we hit:

### Rate Limits Killed Our Embedding Strategy
Our first approach to Basic RAG used sentence-level embedding — splitting text into individual sentences and embedding each one. At 4.1M tokens, this meant hundreds of thousands of API calls to a free-tier embedding endpoint. We hit rate limits within minutes.

**Fix:** We switched to a **character chunker** (1,800 characters per chunk, 300-character overlap). Fewer, larger chunks. Still semantically coherent. Rate limit problems gone. Index build time dropped dramatically.

### GPU Acceleration: 8 Minutes → 25 Seconds
Running MiniLM embeddings on CPU for 4.1M tokens takes approximately forever. With an RTX 3070 Ti and `BENCHMARK_EMBED_DEVICE=cuda`, Basic RAG index build dropped from **~8 minutes to ~25 seconds**. Worth the environment setup time by a factor of 20.

### TigerGraph's 6-Container Docker Stack
The TigerGraph GraphRAG backend runs as a 6-container Docker Compose stack. Getting it initialized, waiting for the database to stabilize, and then running the ingest pipeline (which triggers Gemini-based entity extraction in the background) took careful orchestration. The entity extraction alone runs 10-15 minutes for a full corpus load — that's Gemini parsing every book for characters, places, events, and relationships.

**Lesson:** If you're building on TigerGraph, budget time for the ingest pipeline. The graph is doing real work upfront so queries can be fast later.

### Evaluating Without Ground Truth
We didn't have pre-labeled Q&A pairs for 29 novels. So we used **LLM-as-Judge** — prompting a separate LLM to compare our generated answer against a reference answer and return a structured PASS/FAIL verdict. This is now a standard pattern in LLM evaluation, and it scales in a way that human annotation does not.

---

## Why GraphRAG Is the Right Tool for Relationship-Heavy Domains

Vector search is a retrieval hammer. It works great when your queries are "find me things similar to this." But a knowledge graph is a different kind of tool — it's a **structured representation of meaning**.

When you ask "How does Elizabeth Bennet's opinion of Darcy evolve throughout Pride and Prejudice?", you're not asking for the most similar text chunks to your query. You're asking for a *temporal narrative* about a *relationship between two entities*. That's a graph query disguised as natural language.

GraphRAG's hybrid search — combining vector similarity for initial retrieval with graph traversal for relationship reasoning — is the architecture that closes this gap. It's not a silver bullet for every use case. Simple factual lookup? LLM-only is fine. Semantic document search? Basic RAG works. But for corpora where the *relationships between entities matter* — medical knowledge graphs, legal document networks, literary analysis, enterprise knowledge bases — GraphRAG is the right foundation.

---

## What's Next

This benchmark is a starting point, not a finish line. Here's what we'd tackle with more time:

- **Streaming evaluation**: Run the judge and BERTScore in real-time as answers stream in, not after the fact.
- **Expand the corpus**: 29 books is compelling. 290 books — across genres, languages, and time periods — would be definitive.
- **Fine-tuned judges**: Domain-specific LLM judges (trained on literary criticism, for example) would likely outperform a general-purpose judge on nuanced accuracy questions.
- **Cost optimization for GraphRAG**: The token cost advantage of GraphRAG is clearest when you amortize the index-build cost over many queries. We'd like to model this more rigorously.
- **Multi-language support**: Gutenberg has thousands of non-English texts. Testing GraphRAG's multilingual entity extraction with Gemini embeddings is an obvious next step.

---

## Try It Yourself

Everything is open source and ready to run.

```bash
git clone https://github.com/MrEinsteinE/graphrag-inference-hackathon.git
cd graphrag-inference-hackathon
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# Set your API keys in .env (Groq + Gemini)
copy .env.example .env

# Fetch the corpus
python scripts\fetch_gutenberg_corpus.py

# Start TigerGraph
cd graphrag_docker && docker compose up -d && cd ..

# Run the dashboard
streamlit run streamlit_app.py
```

Open `http://localhost:8501`, type any question about any of the 29 novels, and watch all three pipelines race to answer. The numbers will speak for themselves.

---

## 🔗 Links

- **GitHub**: [github.com/MrEinsteinE/graphrag-inference-hackathon](https://github.com/MrEinsteinE/graphrag-inference-hackathon)
- **TigerGraph GraphRAG**: [github.com/tigergraph/graphrag](https://github.com/tigergraph/graphrag)
- **Hackathon**: TigerGraph GraphRAG Inference Hackathon

---

*Built with TigerGraph, Groq, Google Gemini, Chroma, Streamlit, sentence-transformers, and BERTScore. All models used are available on free tiers — no budget required to reproduce our results.*

*If you found this useful, star the repo and share your own benchmark results. The more datasets and question types we test across the community, the better we'll all understand when and why GraphRAG wins.*
