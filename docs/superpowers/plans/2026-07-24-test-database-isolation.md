# Test Database Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give pytest and the `app/eval` harness their own Postgres database (`rag_test`) inside the existing `db` compose container, so they never truncate or overwrite data a developer ingested through a locally running `uvicorn` dev server against the `rag` database.

**Architecture:** Add a second logical database `rag_test` to the same Postgres instance via an `initdb.d` SQL script. Extract the inline schema-application loop currently living in `app/main.py`'s lifespan into a reusable `app/db/schema.py:apply_schema()` function. Both `tests/conftest.py` and `app/eval/run_eval.py` force `DATABASE_URL` to point at `rag_test` (via `os.environ.setdefault`, before any `app.*` import) and call `apply_schema()` once so a fresh `rag_test` gets its tables created automatically.

**Tech Stack:** FastAPI, SQLAlchemy (async, asyncpg), pgvector/Postgres 16, pytest + pytest-asyncio, Docker Compose.

## Global Constraints

- `rag_test` lives in the *same* `db` compose service/container as `rag` — no second container.
- `os.environ.setdefault` (never a plain assignment) must be used for the `DATABASE_URL` override in `tests/conftest.py` and `app/eval/run_eval.py`, so an operator can still force a different target by pre-setting `DATABASE_URL`.
- The override must be the first statement in each of those two files, before any other import — `app/db/session.py` builds its module-level `engine` from `get_settings().database_url` at import time.
- No change to how the dev server or `/documents`, `/query` endpoints pick their database (`DATABASE_URL` from `.env`, unchanged).
- No new test-to-test isolation — tests still share `rag_test` sequentially and truncate between tests, as today.
- No CI wiring changes.
- Default test/eval target: `postgresql+asyncpg://rag:rag@localhost:5433/rag_test`.

---

### Task 1: `rag_test` database creation via compose

**Files:**
- Create: `docker/initdb/01-create-test-db.sql`
- Modify: `docker-compose.yml`
- Modify: `README.md` (Setup section)

**Interfaces:**
- Produces: a `rag_test` database reachable at `postgresql+asyncpg://rag:rag@localhost:5433/rag_test` on the existing `db` service, for later tasks to point `DATABASE_URL` at.

- [ ] **Step 1: Create the initdb script**

Create `docker/initdb/01-create-test-db.sql`:

```sql
CREATE DATABASE rag_test;
```

- [ ] **Step 2: Mount the initdb directory in docker-compose.yml**

Modify `docker-compose.yml` — add the new volume mount to the `db` service's `volumes:` list (keep the existing `pgdata` mount):

```yaml
services:
  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: rag
      POSTGRES_PASSWORD: rag
      POSTGRES_DB: rag
    ports:
      - "5433:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./docker/initdb:/docker-entrypoint-initdb.d
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U rag -d rag"]
      interval: 5s
      timeout: 5s
      retries: 5

volumes:
  pgdata:
```

- [ ] **Step 3: Verify against a fresh volume**

Postgres only runs `initdb.d` scripts the first time a container starts against an empty volume, so this must be tested against a fresh volume, not the developer's existing one. Run:

```bash
docker compose down -v
docker compose up -d
docker compose exec db psql -U rag -d rag -c "\l" | grep rag_test
```

Expected: a line containing `rag_test` in the database listing.

- [ ] **Step 4: Document the one-time manual step for existing clones**

Modify `README.md`'s "Setup" section — after the existing `docker compose up -d` / `uv sync --extra dev` block, add:

```markdown
If you already had a `db` container running before this change (so its Postgres
data volume predates `rag_test`), create the test database once manually —
Postgres only runs `docker/initdb/` scripts against a fresh, empty volume:

\`\`\`bash
docker compose exec db psql -U rag -d rag -c "CREATE DATABASE rag_test"
\`\`\`

New clones get `rag_test` automatically via `docker/initdb/01-create-test-db.sql`.
```

(Use literal triple-backtick fences in the actual README edit, not escaped ones.)

- [ ] **Step 5: Commit**

```bash
git add docker/initdb/01-create-test-db.sql docker-compose.yml README.md
git commit -m "feat: add rag_test database via compose initdb script"
```

---

### Task 2: `test_database_url` setting

**Files:**
- Modify: `app/config.py`
- Modify: `.env.example`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `Settings.test_database_url: str`, default `"postgresql+asyncpg://rag:rag@localhost:5433/rag_test"`, documenting and validating `TEST_DATABASE_URL` the same way `DATABASE_URL` is validated. (Not read via `get_settings()` by Task 4/5's override — that reads `os.environ` directly, before `Settings` exists — this field exists purely so the env var is documented/typed like its sibling.)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
def test_settings_default_test_database_url(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-test")
    settings = Settings(_env_file=None)
    assert settings.test_database_url == "postgresql+asyncpg://rag:rag@localhost:5433/rag_test"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py::test_settings_default_test_database_url -v`
Expected: FAIL with `AttributeError: 'Settings' object has no attribute 'test_database_url'`

- [ ] **Step 3: Add the field**

Modify `app/config.py`:

```python
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str
    voyage_api_key: str
    database_url: str = "postgresql+asyncpg://rag:rag@localhost:5433/rag"
    test_database_url: str = "postgresql+asyncpg://rag:rag@localhost:5433/rag_test"
    embedding_model: str = "voyage-3"
    embedding_dim: int = 1024
    claude_model: str = "claude-opus-4-8"
    chunk_size: int = 500
    chunk_overlap: int = 50


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (all tests in the file, including the new one and the two pre-existing ones)

- [ ] **Step 5: Document `TEST_DATABASE_URL` in `.env.example`**

Modify `.env.example` — add after the existing `DATABASE_URL` line:

```
ANTHROPIC_API_KEY=sk-ant-your-key-here
VOYAGE_API_KEY=pa-your-key-here
DATABASE_URL=postgresql+asyncpg://rag:rag@localhost:5433/rag
TEST_DATABASE_URL=postgresql+asyncpg://rag:rag@localhost:5433/rag_test
```

- [ ] **Step 6: Commit**

```bash
git add app/config.py tests/test_config.py .env.example
git commit -m "feat: add test_database_url setting"
```

---

### Task 3: Extract `apply_schema` and refactor `app/main.py`'s lifespan

**Files:**
- Create: `app/db/schema.py`
- Modify: `app/main.py`

**Interfaces:**
- Consumes: nothing new (uses `SQLAlchemy`'s `AsyncEngine`, already a project dependency).
- Produces: `apply_schema(engine: AsyncEngine) -> None` (async), reading `app/db/schema.sql` relative to `app/db/schema.py` and executing each non-empty, semicolon-split statement. Task 4 and Task 5 both call this.

This is a behavior-preserving refactor (the SQL-splitting/execution logic is copied verbatim out of `app/main.py`), so there is no new unit test for `apply_schema` itself — the existing pytest suite exercises it via `app/main.py`'s lifespan and, once wired up, via Task 4's autouse fixture. This mirrors how `app/main.py`'s lifespan has no dedicated test today.

- [ ] **Step 1: Confirm the baseline test suite passes before refactoring**

Run: `uv run pytest -v`
Expected: PASS (establishes the behavior this refactor must preserve)

- [ ] **Step 2: Create `app/db/schema.py`**

```python
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

- [ ] **Step 3: Update `app/main.py` to use it**

Replace the full contents of `app/main.py`:

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import documents, health, query
from app.db.schema import apply_schema
from app.db.session import engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    await apply_schema(engine)
    yield


app = FastAPI(title="FastAPI RAG", lifespan=lifespan)

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(query.router)
```

- [ ] **Step 4: Run the full test suite to confirm the refactor is behavior-preserving**

Run: `uv run pytest -v`
Expected: PASS, same results as Step 1

- [ ] **Step 5: Commit**

```bash
git add app/db/schema.py app/main.py
git commit -m "refactor: extract apply_schema from app/main.py lifespan"
```

---

### Task 4: Route pytest at `rag_test` and apply schema once per session

**Files:**
- Modify: `tests/conftest.py`

**Interfaces:**
- Consumes: `apply_schema(engine)` from `app/db/schema.py` (Task 3); `engine` from `app/db/session.py`.
- Produces: nothing new for other tasks — this is the terminal consumer of the `DATABASE_URL` override pattern (Task 5 duplicates it independently for `app/eval/run_eval.py`, since the two entry points don't share an import).

- [ ] **Step 1: Add the `DATABASE_URL` override as the first statement in the file, and an autouse session-scoped schema fixture**

Replace the full contents of `tests/conftest.py`:

```python
import os

os.environ.setdefault(
    "DATABASE_URL",
    os.environ.get("TEST_DATABASE_URL", "postgresql+asyncpg://rag:rag@localhost:5433/rag_test"),
)

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

from app.db.schema import apply_schema
from app.db.session import engine
from app.main import app


@pytest_asyncio.fixture(autouse=True, scope="session")
async def _apply_schema():
    await apply_schema(engine)


@pytest_asyncio.fixture(autouse=True)
async def _clean_tables():
    yield
    async with engine.begin() as conn:
        await conn.execute(text("TRUNCATE chunks, documents RESTART IDENTITY CASCADE"))


@pytest_asyncio.fixture
async def client():
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac
    app.dependency_overrides.clear()


from app.db.session import session_factory


@pytest_asyncio.fixture
async def db_session():
    async with session_factory() as session:
        yield session
```

- [ ] **Step 2: Verify pytest now targets `rag_test`, not `rag`**

First, confirm `rag_test` starts empty (from Task 1/Step 3's fresh volume, or the manual `CREATE DATABASE` step):

```bash
docker compose exec db psql -U rag -d rag_test -c "\dt"
```

Expected: `Did not find any relations.` (no tables yet — proves nothing has touched this DB before the test run)

- [ ] **Step 3: Run the full test suite**

Run: `uv run pytest -v`
Expected: PASS. All DB-touching tests succeed against `rag_test`.

- [ ] **Step 4: Confirm tables were created in `rag_test`, not `rag`**

```bash
docker compose exec db psql -U rag -d rag_test -c "\dt"
```

Expected: `chunks` and `documents` tables listed.

```bash
docker compose exec db psql -U rag -d rag -c "\dt"
```

Expected: still `Did not find any relations.` (or whatever `rag` had before — untouched by the test run either way).

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py
git commit -m "feat: route pytest at rag_test database"
```

---

### Task 5: Route `app/eval/run_eval.py` at `rag_test` and apply schema before ingesting

**Files:**
- Modify: `app/eval/run_eval.py`

**Interfaces:**
- Consumes: `apply_schema(engine)` from `app/db/schema.py` (Task 3).
- Produces: nothing further downstream.

- [ ] **Step 1: Add the `DATABASE_URL` override as the first statement in the file**

Modify `app/eval/run_eval.py` — insert before the existing `import argparse` line (which becomes line 1 of the import block):

```python
import os

os.environ.setdefault(
    "DATABASE_URL",
    os.environ.get("TEST_DATABASE_URL", "postgresql+asyncpg://rag:rag@localhost:5433/rag_test"),
)

import argparse
import asyncio
import json
import sys
from pathlib import Path

import anthropic
import voyageai
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.tool_loop import run_agent_query
from app.api.documents import create_document
from app.config import get_settings
from app.db.schema import apply_schema
from app.db.session import engine, session_factory
from app.eval.judge import JudgeError, judge_case
from app.eval.report import CaseResult, case_passed, write_report
from app.schemas import DocumentCreateRequest
from app.services.embeddings import EmbeddingService
```

- [ ] **Step 2: Call `apply_schema` before ingesting fixtures in `main_async`**

In `app/eval/run_eval.py`, modify `main_async`:

```python
async def main_async(args: argparse.Namespace) -> int:
    settings = get_settings()
    documents = load_documents(args.documents)
    cases = load_cases(args.cases)

    await apply_schema(engine)
    await truncate_tables()

    anthropic_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    voyage_client = voyageai.Client(api_key=settings.voyage_api_key)
    embeddings = EmbeddingService(voyage_client, model=settings.embedding_model)

    async with session_factory() as session:
        exit_code = await orchestrate(
            documents=documents,
            cases=cases,
            session=session,
            embeddings=embeddings,
            agent_client=anthropic_client,
            judge_client=anthropic_client,
            agent_model=settings.claude_model,
            judge_model=args.judge_model,
            output_path=args.output,
        )

    await truncate_tables()
    return exit_code
```

(Only the new `await apply_schema(engine)` line before `await truncate_tables()` is added — everything else in `main_async` is unchanged.)

- [ ] **Step 3: Run the existing eval-related pytest suite**

`app/eval/run_eval.py` is imported by `tests/test_eval_run_eval.py`, so this checks the module still imports and its unit-tested functions (`evaluate_case`, `orchestrate`, `parse_args`, `ingest_documents`) still behave correctly under the new top-of-file import order.

Run: `uv run pytest tests/test_eval_run_eval.py -v`
Expected: PASS

- [ ] **Step 4: Run the full test suite**

Run: `uv run pytest -v`
Expected: PASS

- [ ] **Step 5: Manually verify `run_eval.py` targets `rag_test`**

This step requires real `ANTHROPIC_API_KEY` / `VOYAGE_API_KEY` values in `.env` — skip if not available in this environment, but note that in the task handoff.

```bash
docker compose exec db psql -U rag -d rag -c "SELECT count(*) FROM documents;"
uv run python -m app.eval.run_eval
docker compose exec db psql -U rag -d rag -c "SELECT count(*) FROM documents;"
docker compose exec db psql -U rag -d rag_test -c "SELECT count(*) FROM documents;"
```

Expected: the `rag` document count is identical before and after (eval never touched it); `rag_test`'s count is `0` (eval truncates after itself per existing `main_async` behavior).

- [ ] **Step 6: Commit**

```bash
git add app/eval/run_eval.py
git commit -m "feat: route app/eval/run_eval.py at rag_test database"
```

---

### Task 6: README — document concurrent dev/test/eval usage

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing (docs-only task).

- [ ] **Step 1: Update the "Run tests" section**

In `README.md`, replace the "Run tests" section:

```markdown
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
```

- [ ] **Step 2: Update the "Run the agent evaluation" section**

In `README.md`, replace the "Run the agent evaluation" section:

```markdown
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
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document concurrent dev/test/eval database usage"
```

---

## Self-Review Notes

- **Spec coverage:** All four spec components (§1 `rag_test` creation, §2 shared schema application, §3 routing pytest/eval at `rag_test`, §4 docs) map to Tasks 1–6. The spec's `Settings.test_database_url` field (§3) is Task 2. Non-goals (no dev-server routing change, no cross-test isolation, no CI changes) are respected — no task touches those.
- **Placeholder scan:** No TBD/TODO markers; every step has literal, complete code or an exact runnable command with expected output.
- **Type consistency:** `apply_schema(engine: AsyncEngine) -> None` (Task 3) is called identically in Task 4 (`tests/conftest.py`) and Task 5 (`app/eval/run_eval.py`) — same name, same single positional argument, no signature drift.
