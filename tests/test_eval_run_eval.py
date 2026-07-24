import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from sqlalchemy import func, select

from app.db.models import Chunk, Document
from app.eval.run_eval import ingest_documents, load_cases, load_documents

FIXTURES_DIR = Path(__file__).parent.parent / "app" / "eval" / "fixtures"


def test_load_documents_reads_json_array():
    documents = load_documents(FIXTURES_DIR / "documents.json")
    assert isinstance(documents, list)
    assert documents[0]["title"] == "Vacation Policy"


def test_load_cases_reads_json_array():
    cases = load_cases(FIXTURES_DIR / "cases.json")
    assert isinstance(cases, list)
    assert cases[0]["id"] == "vacation-accrual"


def test_load_documents_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "missing.json")


def _fake_embeddings():
    fake = Mock()
    fake.embed_documents.side_effect = lambda texts: [[0.01] * 1024 for _ in texts]
    return fake


@pytest.mark.asyncio
async def test_ingest_documents_creates_documents_and_chunks(db_session):
    documents = [
        {"title": "Doc A", "source": "test", "content": " ".join(["word"] * 50), "metadata": {}},
        {"title": "Doc B", "source": "test", "content": " ".join(["word"] * 50), "metadata": {"lang": "en"}},
    ]

    await ingest_documents(db_session, _fake_embeddings(), documents)

    doc_count = (await db_session.execute(select(func.count()).select_from(Document))).scalar_one()
    chunk_count = (await db_session.execute(select(func.count()).select_from(Chunk))).scalar_one()
    assert doc_count == 2
    assert chunk_count == 2
