# Test Database Isolation — Design

## Goal

Let the local dev server (`uvicorn app.main:app --reload`), the pytest suite,
and the `app/eval` harness run at the same time without stepping on each
other's data. Today all three point at the same `rag` database on the `db`
compose service; `tests/conftest.py` truncates `chunks`/`documents` after
every test, and `app/eval/run_eval.py` truncates them before and after its
run — either one wipes out anything a developer has manually ingested via a
running dev server.

## Approach

Add a second logical Postgres database, `rag_test`, inside the *same*
`db` compose service/container (not a second container) — same Postgres
process, same port (5433), different database name. pytest and the eval
harness always connect to `rag_test`; the dev server keeps using `rag` as
it does today via `.env`.

This was chosen over a second Postgres container for lower resource
overhead, since full container-level isolation isn't needed — database-level
isolation inside one Postgres instance is already enough to stop
truncation/ingestion in tests or eval from touching dev data.

## Components

### 1. `rag_test` database creation

- `docker/initdb/01-create-test-db.sql` (new): `CREATE DATABASE rag_test;`
  Mounted into the `db` service at `/docker-entrypoint-initdb.d/` in
  `docker-compose.yml`. Postgres only runs `initdb.d` scripts the *first*
  time a container starts against a fresh, empty volume.
- Existing dev environments already have a populated `pgdata` volume, so the
  initdb script won't run for them. The README documents a one-time manual
  step for that case:
  `docker compose exec db psql -U rag -d rag -c "CREATE DATABASE rag_test"`.
- Both paths are idempotent from the developer's perspective: a fresh clone
  gets `rag_test` automatically; an existing clone needs the one manual
  command once, and running it again on an already-created DB just errors
  harmlessly (documented as a "run once" step, not something scripted to be
  safely re-runnable).

### 2. Shared schema application (`app/db/schema.py`, new)

`app/main.py`'s lifespan currently reads `schema.sql` and executes each
statement inline. Extract that into:

```python
# app/db/schema.py
from pathlib import Path
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

SCHEMA_SQL_PATH = Path(__file__).parent / "schema.sql"

async def apply_schema(engine: AsyncEngine) -> None:
    sql = SCHEMA_SQL_PATH.read_text()
    async with engine.begin() as conn:
        for statement in sql.split(";"):
            statement = statement.strip()
            if statement:
                await conn.execute(text(statement))
```

- `app/main.py`'s lifespan calls `apply_schema(engine)` instead of inlining
  the loop (behavior-preserving refactor for the dev/prod path).
- A new autouse, session-scoped pytest fixture in `tests/conftest.py` calls
  `apply_schema(engine)` once per test session, so a freshly created
  `rag_test` (which has no tables yet) gets them created automatically —
  today this only happens implicitly when something starts the FastAPI app
  first.
- `app/eval/run_eval.py`'s `main_async` calls `apply_schema(engine)` before
  ingesting fixtures, for the same reason when eval is run standalone.

### 3. Routing pytest and eval at `rag_test`

Both `tests/conftest.py` and `app/eval/run_eval.py` must resolve
`DATABASE_URL` to the test database *before* importing anything from `app.*`
— `app/db/session.py` builds its module-level `engine` from
`get_settings().database_url` at import time, so the override has to land
before that import happens anywhere in the process.

Both files gain this as their first statement, before any other import:

```python
import os
os.environ.setdefault(
    "DATABASE_URL",
    os.environ.get("TEST_DATABASE_URL", "postgresql+asyncpg://rag:rag@localhost:5433/rag_test"),
)
```

`os.environ.setdefault` (not a plain assignment) so an operator can still
force a different target by pre-setting `DATABASE_URL` themselves.

`app/config.py`'s `Settings` gains a `test_database_url: str` field
(default `postgresql+asyncpg://rag:rag@localhost:5433/rag_test`) purely so
`TEST_DATABASE_URL` is documented and validated the same way
`DATABASE_URL` is; the override snippet above reads it from `os.environ`
directly (before `Settings` exists) rather than through `get_settings()`,
since the whole point is to run before settings/engine construction.

### 4. Docs

- `.env.example` gains `TEST_DATABASE_URL=postgresql+asyncpg://rag:rag@localhost:5433/rag_test`.
- README's "Setup" section gains the one-time `CREATE DATABASE rag_test`
  step for existing clones, and a note that `Run tests` / `Run the agent
  evaluation` always target `rag_test`, independent of `.env`'s
  `DATABASE_URL` — so the dev server and tests/eval can run concurrently
  against the same `db` container without conflict.

## Non-goals

- No change to how the dev server or `/documents`, `/query` endpoints pick
  their database — they keep reading `DATABASE_URL` from `.env` normally.
- No new test isolation *within* a pytest run (tests still share `rag_test`
  sequentially and truncate between tests, as today) — this only isolates
  *dev* data from *test/eval* data, not tests from each other.
- No CI wiring changes.

## Testing

- Existing pytest suite continues to pass unchanged (it exercises the new
  `apply_schema` path implicitly via the new autouse fixture; no dedicated
  unit test for `apply_schema` itself beyond that integration coverage,
  consistent with how `app/main.py`'s lifespan is untested today).
- Manual verification: start the dev server against `rag` on a workstation
  where `rag_test` is empty, run `uv run pytest -v` concurrently, confirm
  both succeed and `GET /documents/{id}` against dev-ingested data is
  unaffected by the test run.
