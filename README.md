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

```bash
docker compose up -d
uv run pytest -v
```

## Design notes

See `docs/adr/` for architecture decision records, e.g. why `search_chunks`
forces an exhaustive `ivfflat` scan today (`docs/adr/0001-ivfflat-exhaustive-probes.md`).
