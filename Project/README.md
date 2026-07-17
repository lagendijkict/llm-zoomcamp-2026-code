# RAG Application Scaffold

An end-to-end Retrieval-Augmented Generation application: ingestion pipeline,
hybrid (vector + keyword) search, retrieval + LLM evaluation, a chat UI, and
a monitoring dashboard — all running on Postgres + pgvector via Docker Compose.

## Architecture


Source data --> loader.py --> chunking.py --> embed.py --> pipeline.py --(upsert)--> Postgres
                                                                                          |
                                                                                    pgvector + tsvector
                                                                                          |
User question --> app.py --> retrieval/hybrid.py (RRF fusion) --> rag/prompts.py --> LLM --> answer
                                       |                                                        |
                                  conversations table  <-----------------------------------------
                                       |
                                  dashboard.py (monitoring)
```

**Why Postgres + pgvector instead of a dedicated vector DB:** one system to
run and operate instead of two, and it gives hybrid search "for free" via
native full-text search (`tsvector`) in the same table — no need to keep two
databases in sync. Trade-off: at very large scale (tens of millions of
vectors) a purpose-built vector DB (Qdrant, Weaviate) or Postgres extensions
tuned further (HNSW index instead of IVFFlat) will outperform this setup.
For a project-scale corpus, that trade-off doesn't matter yet.

## Setup

1. Copy `.env.example` to `.env` and fill in `OPENAI_API_KEY` (or switch
   `LLM_PROVIDER` to `ollama` and point `LLM_BASE_URL` at a local Ollama
   instance — see comments in `.env.example`).
2. Implement `src/ingestion/loader.py::load_raw_documents()` for your dataset.
3. Bring up Postgres and run ingestion:
   ```bash
   docker compose up -d postgres
   docker compose --profile tools run --rm ingest
   ```
4. Start the app and dashboard:
   ```bash
   docker compose up app dashboard
   ```
   App: http://localhost:8501 · Dashboard: http://localhost:8502

### Running locally without Docker (e.g. inside an existing Codespaces Postgres)

```bash
pip install -r requirements.txt
export POSTGRES_HOST=localhost  # or your Codespaces Postgres hostname
python -m src.ingestion.pipeline
streamlit run app.py
streamlit run dashboard.py --server.port 8502
```

## Evaluation

```bash
# Retrieval: compares vector-only, text-only, and hybrid strategies on
# hit rate and MRR. Generates a synthetic ground-truth set on first run
# (cached to data/ground_truth.json) if none exists.
python -m src.evaluation.retrieval_eval

# LLM-as-judge: compares the three prompt variants in src/rag/prompts.py
# on relevance / faithfulness / completeness. Edit the sample_questions
# list in llm_eval.py's __main__ block to use real questions from your domain.
python -m src.evaluation.llm_eval
```

Both scripts log which strategy/variant scored best — that's the "best one
is used" the rubric asks for. In practice, wire the winning strategy/variant
as the default in `src/config.py` and `src/rag/prompts.py` once you've run
these.

## Testing

```bash
pytest tests/
```

`tests/test_rrf.py` unit-tests the Reciprocal Rank Fusion logic in isolation
(no DB or LLM calls needed) — the fusion math is the part of this scaffold
easiest to subtly break during a refactor, so it's the part with a fast,
dependency-free test.

## Mapping to the evaluation rubric

| Criterion | Where |
|---|---|
| Retrieval flow | `src/retrieval/`, `src/rag/pipeline.py` |
| Retrieval evaluation (multiple approaches) | `src/evaluation/retrieval_eval.py` — vector vs. text vs. hybrid |
| LLM evaluation (multiple approaches) | `src/evaluation/llm_eval.py` — 3 prompt variants judged |
| Interface | `app.py` (Streamlit chat UI) |
| Ingestion pipeline (automated) | `src/ingestion/pipeline.py`, idempotent via content-hash upsert |
| Monitoring (feedback + dashboard) | `src/monitoring/feedback.py`, `dashboard.py` (6 charts) |
| Containerization | `docker-compose.yml` — everything runs via compose |
| Reproducibility | pinned `requirements.txt`, `.env.example` |
| Best practice: hybrid search | `src/retrieval/hybrid.py` (RRF) |

Not yet implemented — worth adding for the best-practices bonus points:
**document re-ranking** (e.g. a cross-encoder reranker on top of the RRF
results before truncating to `top_k`) and **query rewriting** (LLM rewrites
the user's question before retrieval, useful for multi-turn conversations
where the latest message alone lacks context).

## Known limitations

- IVFFlat index `lists=100` is a starting point, not tuned for your corpus size.
- The synthetic ground-truth generator (`generate_ground_truth`) produces
  imperfect questions — spot-check a sample before trusting the eval numbers.
- No reranking or query rewriting yet (see above).
- No caching layer — every question re-embeds and re-queries, fine for a
  project but worth flagging as a known scaling limitation.
