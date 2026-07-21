from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.dependencies import get_anthropic_client_dep, get_embedding_service_dep
from app.main import app


@pytest.mark.asyncio
async def test_query_endpoint_returns_answer_and_sources(client):
    fake_embeddings = Mock()
    fake_embeddings.embed_documents.side_effect = lambda texts: [[0.02] * 1024 for _ in texts]
    fake_embeddings.embed_query.return_value = [0.02] * 1024
    app.dependency_overrides[get_embedding_service_dep] = lambda: fake_embeddings

    create_resp = await client.post(
        "/documents",
        json={"title": "Facts", "source": "test", "content": "The sky is blue. " * 20},
    )
    assert create_resp.status_code == 201

    fake_anthropic = Mock()
    tool_response = SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", id="toolu_1", input={"query": "sky color", "top_k": 3})],
        stop_reason="tool_use",
    )
    final_response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="The sky is blue [Facts].")],
        stop_reason="end_turn",
    )
    fake_anthropic.messages.create = Mock(side_effect=[tool_response, final_response])
    app.dependency_overrides[get_anthropic_client_dep] = lambda: fake_anthropic

    resp = await client.post("/query", json={"question": "What color is the sky?"})

    assert resp.status_code == 200
    body = resp.json()
    assert "blue" in body["answer"]
    assert len(body["sources"]) >= 1
    assert body["sources"][0]["document_title"] == "Facts"


@pytest.mark.asyncio
async def test_query_endpoint_validates_request_body(client):
    resp = await client.post("/query", json={})
    assert resp.status_code == 422
