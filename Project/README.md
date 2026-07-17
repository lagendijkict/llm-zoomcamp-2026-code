# RAG Application — Quick Start

This app answers questions using arxiv papers, stored in Postgres with pgvector.

## 1. Start Postgres

```bash
docker compose up -d postgres
```

This runs Postgres in the background. Leave it running.

## 2. Ingest the data

This downloads/chunks/embeds the source documents and loads them into Postgres.
Only needs to be run once (safe to re-run — it won't duplicate data).

```bash
docker compose --profile tools run --rm ingest
```

## 3. Start the chat app

```bash
uv run streamlit run app.py
```

Open your browser to: **http://localhost:8501**

This is where you ask questions and get answers.

## 4. Start the dashboard

```bash
uv run streamlit run dashboard.py --server.port 8502
```

Open your browser to: **http://localhost:8502**

This shows monitoring charts (questions asked, feedback, etc.).

---

## Notes

- Run app.py and dashboard.py in **separate terminals** — they both need to keep running.
- Postgres (step 1) needs to already be running before steps 2–4 will work.
- Make sure your `.env` file has a real `OPENAI_API_KEY` (not the placeholder value) — the app won't work without it.