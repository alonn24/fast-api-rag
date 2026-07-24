import json
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.documents import create_document
from app.db.session import engine
from app.schemas import DocumentCreateRequest
from app.services.embeddings import EmbeddingService

DEFAULT_DOCUMENTS_PATH = Path(__file__).parent / "fixtures" / "documents.json"
DEFAULT_CASES_PATH = Path(__file__).parent / "fixtures" / "cases.json"
DEFAULT_OUTPUT_PATH = Path("eval_report.html")
DEFAULT_JUDGE_MODEL = "claude-sonnet-5"


def load_documents(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def load_cases(path: Path) -> list[dict]:
    return json.loads(path.read_text())


async def truncate_tables() -> None:
    async with engine.begin() as conn:
        await conn.execute(text("TRUNCATE chunks, documents RESTART IDENTITY CASCADE"))


async def ingest_documents(session: AsyncSession, embeddings: EmbeddingService, documents: list[dict]) -> None:
    for doc in documents:
        body = DocumentCreateRequest(
            title=doc["title"],
            source=doc["source"],
            content=doc["content"],
            metadata=doc.get("metadata", {}),
        )
        await create_document(body, session=session, embeddings=embeddings)
