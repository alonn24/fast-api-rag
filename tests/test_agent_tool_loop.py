import uuid
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.agent.tool_loop import run_agent_query
from app.services.search import SearchResult


def _text_block(text: str):
    return SimpleNamespace(type="text", text=text)


def _tool_use_block(tool_id: str, query: str, top_k: int = 5):
    return SimpleNamespace(type="tool_use", id=tool_id, input={"query": query, "top_k": top_k})


@pytest.mark.asyncio
async def test_run_agent_query_calls_tool_then_returns_final_answer(monkeypatch):
    tool_response = SimpleNamespace(
        content=[_tool_use_block("toolu_1", "revenue")],
        stop_reason="tool_use",
    )
    final_response = SimpleNamespace(
        content=[_text_block("Revenue grew 10% [Q3 Report].")],
        stop_reason="end_turn",
    )
    fake_client = Mock()
    fake_client.messages.create = Mock(side_effect=[tool_response, final_response])

    fake_embeddings = Mock()
    fake_embeddings.embed_query.return_value = [0.1, 0.2]

    fake_result = SearchResult(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        document_title="Q3 Report",
        content="Revenue grew 10%.",
        distance=0.1,
    )

    async def fake_search_chunks(session, query_embedding, top_k=5):
        return [fake_result]

    import app.agent.tool_loop as tool_loop_module

    monkeypatch.setattr(tool_loop_module, "search_chunks", fake_search_chunks)

    result = await run_agent_query(
        "How did revenue change?",
        client=fake_client,
        session=Mock(),
        embeddings=fake_embeddings,
    )

    assert "Revenue grew 10%" in result.answer
    assert len(result.sources) == 1
    assert result.sources[0].document_title == "Q3 Report"
    assert fake_client.messages.create.call_count == 2


@pytest.mark.asyncio
async def test_run_agent_query_no_tool_call_returns_direct_answer():
    direct_response = SimpleNamespace(
        content=[_text_block("I don't know.")],
        stop_reason="end_turn",
    )
    fake_client = Mock()
    fake_client.messages.create = Mock(return_value=direct_response)

    result = await run_agent_query(
        "Unrelated question",
        client=fake_client,
        session=Mock(),
        embeddings=Mock(),
    )

    assert result.answer == "I don't know."
    assert result.sources == []
    assert fake_client.messages.create.call_count == 1


@pytest.mark.asyncio
async def test_run_agent_query_stops_at_max_iterations(monkeypatch):
    looping_response = SimpleNamespace(
        content=[_tool_use_block("toolu_x", "q")],
        stop_reason="tool_use",
    )
    fake_client = Mock()
    fake_client.messages.create = Mock(return_value=looping_response)

    fake_embeddings = Mock()
    fake_embeddings.embed_query.return_value = [0.1]

    async def fake_search_chunks(session, query_embedding, top_k=5):
        return []

    import app.agent.tool_loop as tool_loop_module

    monkeypatch.setattr(tool_loop_module, "search_chunks", fake_search_chunks)

    result = await run_agent_query(
        "loop forever",
        client=fake_client,
        session=Mock(),
        embeddings=fake_embeddings,
        max_iterations=2,
    )

    assert fake_client.messages.create.call_count == 2
    assert "Unable to produce an answer" in result.answer
