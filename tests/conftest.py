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
