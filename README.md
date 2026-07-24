# FastAPI RAG

A FastAPI RAG system: ingest text documents as chunked embeddings in Postgres+pgvector,
and query them via an agentic endpoint where Claude runs a `search_documents` tool-use loop.

## Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/)
- Docker (for Postgres + pgvector)
- An `ANTHROPIC_API_KEY` and a `VOYAGE_API_KEY`

## Setup

```bash
cp .env.example .env
# edit .env and fill in ANTHROPIC_API_KEY and VOYAGE_API_KEY

docker compose up -d
uv sync --extra dev
```

Postgres is exposed on host port `5433` (not the default `5432`), mapped to the
container's internal `5432` — see `docker-compose.yml` and `.env.example`'s
`DATABASE_URL`. Change this mapping if `5433` is also unavailable on your machine.

If you already had a `db` container running before this change (so its Postgres
data volume predates `rag_test`), create the test database once manually —
Postgres only runs `docker/initdb/` scripts against a fresh, empty volume:

```bash
docker compose exec db psql -U rag -d rag -c "CREATE DATABASE rag_test"
```

New clones get `rag_test` automatically via `docker/initdb/01-create-test-db.sql`.

## Run the app

```bash
uv run uvicorn app.main:app --reload
```

The API is at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

## API

- `GET /health` — liveness + DB connectivity check.
- `POST /documents` — ingest a document. Body: `{"title": str, "source": str, "content": str, "metadata": dict}`.
  Returns `{"id": uuid, "chunk_count": int}`.
- `GET /documents/{id}` — fetch document metadata.
- `POST /query` — agentic retrieval. Body: `{"question": str}`.
  Returns `{"answer": str, "sources": [...]}`.

## Run tests

Tests require the Postgres container from `docker-compose.yml` to be running
(DB-touching tests use the real dockerized pgvector instance; tables are
truncated after each test). Anthropic and Voyage calls are mocked via FastAPI
dependency overrides — no real API keys or network calls are needed for the
default test run (a `.env` with placeholder keys is enough for `Settings()`
to construct).

Tests always target the `rag_test` database (see `TEST_DATABASE_URL` in
`.env.example`), independent of whatever `DATABASE_URL` in `.env` points the
dev server at — `docker compose exec db psql -U rag -d rag_test` connects to
the same test database tests use. This means the dev server
(`uv run uvicorn app.main:app --reload`) can be left running against `rag`
while `uv run pytest` runs concurrently without either one truncating the
other's data.

```bash
docker compose up -d
uv run pytest -v
```

## Run the agent evaluation

`app/eval/` is an LLM-as-judge evaluation harness for the `/query` agent. It ingests its
own fixture documents (`app/eval/fixtures/`), runs each fixture question through the real
agent, and scores the answer with an independent judge model. It costs real Anthropic +
Voyage API calls and is intentionally excluded from the default `pytest` run.

Like `pytest`, this always targets `rag_test` (see `TEST_DATABASE_URL` in
`.env.example`), not the `rag` database the dev server uses — it's safe to run
alongside a live dev server.

```bash
docker compose up -d
uv run python -m app.eval.run_eval
```

Writes `eval_report.html` (override with `--output`) and exits nonzero if any case fails
any of the three judged dimensions (correctness, faithfulness, retrieval_relevance). Use
`--judge-model`, `--documents`, `--cases` to override defaults.

## Design notes

See `docs/adr/` for architecture decision records, e.g. why `search_chunks`
forces an exhaustive `ivfflat` scan today (`docs/adr/0001-ivfflat-exhaustive-probes.md`).
