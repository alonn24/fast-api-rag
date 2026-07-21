from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from sqlalchemy import text

from app.api import documents, health
from app.db.session import engine

SCHEMA_SQL_PATH = Path(__file__).parent / "db" / "schema.sql"


@asynccontextmanager
async def lifespan(app: FastAPI):
    sql = SCHEMA_SQL_PATH.read_text()
    async with engine.begin() as conn:
        for statement in sql.split(";"):
            statement = statement.strip()
            if statement:
                await conn.execute(text(statement))
    yield


app = FastAPI(title="FastAPI RAG", lifespan=lifespan)

app.include_router(health.router)
app.include_router(documents.router)
