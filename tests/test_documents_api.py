import uuid
from unittest.mock import Mock

import pytest

from app.dependencies import get_embedding_service_dep
from app.main import app


def _fake_embeddings():
    fake = Mock()
    fake.embed_documents.side_effect = lambda texts: [[0.01] * 1024 for _ in texts]
    return fake


@pytest.mark.asyncio
async def test_create_document_returns_chunk_count(client):
    app.dependency_overrides[get_embedding_service_dep] = _fake_embeddings
    payload = {"title": "Doc A", "source": "unit-test", "content": " ".join(["word"] * 50)}

    resp = await client.post("/documents", json=payload)

    assert resp.status_code == 201
    body = resp.json()
    assert body["chunk_count"] == 1
    assert uuid.UUID(body["id"])


@pytest.mark.asyncio
async def test_create_document_empty_content_returns_422(client):
    app.dependency_overrides[get_embedding_service_dep] = _fake_embeddings
    payload = {"title": "Doc A", "source": "unit-test", "content": "   "}

    resp = await client.post("/documents", json=payload)

    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_get_document_returns_metadata_and_chunk_count(client):
    app.dependency_overrides[get_embedding_service_dep] = _fake_embeddings
    payload = {
        "title": "Doc B",
        "source": "unit-test",
        "content": " ".join(["word"] * 50),
        "metadata": {"lang": "en"},
    }
    create_resp = await client.post("/documents", json=payload)
    doc_id = create_resp.json()["id"]

    resp = await client.get(f"/documents/{doc_id}")

    assert resp.status_code == 200
    body = resp.json()
    assert body["title"] == "Doc B"
    assert body["metadata"] == {"lang": "en"}
    assert body["chunk_count"] == 1


@pytest.mark.asyncio
async def test_get_document_not_found_returns_404(client):
    resp = await client.get(f"/documents/{uuid.uuid4()}")
    assert resp.status_code == 404
