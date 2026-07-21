import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import Chunk, Document
from app.db.session import get_db_session
from app.dependencies import get_embedding_service_dep
from app.schemas import DocumentCreateRequest, DocumentCreateResponse, DocumentGetResponse
from app.services.chunking import chunk_text
from app.services.embeddings import EmbeddingService

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentCreateResponse, status_code=201)
async def create_document(
    body: DocumentCreateRequest,
    session: AsyncSession = Depends(get_db_session),
    embeddings: EmbeddingService = Depends(get_embedding_service_dep),
) -> DocumentCreateResponse:
    settings = get_settings()
    chunks = chunk_text(body.content, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
    if not chunks:
        raise HTTPException(status_code=422, detail="content produced no chunks")

    vectors = embeddings.embed_documents(chunks)

    document = Document(title=body.title, source=body.source, doc_metadata=body.metadata)
    session.add(document)
    await session.flush()

    for index, (text_chunk, vector) in enumerate(zip(chunks, vectors)):
        session.add(Chunk(document_id=document.id, chunk_index=index, content=text_chunk, embedding=vector))

    await session.commit()
    return DocumentCreateResponse(id=document.id, chunk_count=len(chunks))


@router.get("/{document_id}", response_model=DocumentGetResponse)
async def get_document(
    document_id: uuid.UUID, session: AsyncSession = Depends(get_db_session)
) -> DocumentGetResponse:
    document = await session.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="document not found")

    count_stmt = select(func.count()).select_from(Chunk).where(Chunk.document_id == document_id)
    chunk_count = (await session.execute(count_stmt)).scalar_one()

    return DocumentGetResponse(
        id=document.id,
        title=document.title,
        source=document.source,
        metadata=document.doc_metadata,
        created_at=document.created_at,
        chunk_count=chunk_count,
    )
