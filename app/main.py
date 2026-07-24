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
