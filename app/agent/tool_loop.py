import json
import uuid
from dataclasses import dataclass

import anthropic
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.embeddings import EmbeddingService
from app.services.search import SearchResult, search_chunks

SEARCH_TOOL = {
    "name": "search_documents",
    "description": (
        "Search the document knowledge base for chunks relevant to a query. "
        "Call this whenever answering requires information from the ingested documents."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"},
            "top_k": {"type": "integer", "description": "Number of results to return", "default": 5},
        },
        "required": ["query"],
    },
}

SYSTEM_PROMPT = (
    "You are a retrieval-augmented assistant. Use the search_documents tool to find "
    "relevant information before answering questions about the ingested documents. "
    "Cite sources by referencing the document title inline in your answer, e.g. [Title]. "
    "If the tool returns nothing relevant, say so plainly rather than guessing."
)


@dataclass
class AgentAnswer:
    answer: str
    sources: list[SearchResult]


async def run_agent_query(
    question: str,
    *,
    client: anthropic.Anthropic,
    session: AsyncSession,
    embeddings: EmbeddingService,
    model: str = "claude-opus-4-8",
    max_iterations: int = 5,
) -> AgentAnswer:
    messages: list[dict] = [{"role": "user", "content": question}]
    collected_sources: dict[uuid.UUID, SearchResult] = {}

    for _ in range(max_iterations):
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=[SEARCH_TOOL],
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            final_text = "".join(block.text for block in response.content if block.type == "text")
            return AgentAnswer(answer=final_text, sources=list(collected_sources.values()))

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            query = block.input["query"]
            top_k = block.input.get("top_k", 5)
            query_embedding = embeddings.embed_query(query)
            results = await search_chunks(session, query_embedding, top_k=top_k)
            for result in results:
                collected_sources[result.chunk_id] = result
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(
                        [
                            {
                                "document_title": r.document_title,
                                "content": r.content,
                                "distance": r.distance,
                            }
                            for r in results
                        ]
                    ),
                }
            )
        messages.append({"role": "user", "content": tool_results})

    return AgentAnswer(
        answer="Unable to produce an answer within the iteration limit.",
        sources=list(collected_sources.values()),
    )
