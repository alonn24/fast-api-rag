import anthropic
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.tool_loop import run_agent_query
from app.config import get_settings
from app.db.session import get_db_session
from app.dependencies import get_anthropic_client_dep, get_embedding_service_dep
from app.schemas import QueryRequest, QueryResponse, SourceRef
from app.services.embeddings import EmbeddingService

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query(
    body: QueryRequest,
    session: AsyncSession = Depends(get_db_session),
    embeddings: EmbeddingService = Depends(get_embedding_service_dep),
    client: anthropic.Anthropic = Depends(get_anthropic_client_dep),
) -> QueryResponse:
    settings = get_settings()
    result = await run_agent_query(
        body.question,
        client=client,
        session=session,
        embeddings=embeddings,
        model=settings.claude_model,
    )
    return QueryResponse(
        answer=result.answer,
        sources=[
            SourceRef(
                document_id=s.document_id,
                document_title=s.document_title,
                chunk_id=s.chunk_id,
                content=s.content,
            )
            for s in result.sources
        ],
    )
