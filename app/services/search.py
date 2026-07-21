import uuid
from dataclasses import dataclass

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, Document


@dataclass
class SearchResult:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    content: str
    distance: float


async def search_chunks(
    session: AsyncSession, query_embedding: list[float], top_k: int = 5
) -> list[SearchResult]:
    # The chunks_embedding_idx ivfflat index is built with lists=100. IVFFlat is an
    # approximate-nearest-neighbor index: with the small row counts typical of this
    # project's tests (and early-stage real usage), the default probe count (1) can
    # scan too few lists and miss rows that are genuinely closer, so top_k results
    # would come back short or out of order. Raising probes to match the index's
    # list count makes the search scan every list, trading some of the index's
    # speed advantage for correct, deterministic recall.
    await session.execute(text("SET LOCAL ivfflat.probes = 100"))

    distance = Chunk.embedding.cosine_distance(query_embedding)
    stmt = (
        select(Chunk, Document.title, distance.label("distance"))
        .join(Document, Chunk.document_id == Document.id)
        .order_by(distance)
        .limit(top_k)
    )
    rows = (await session.execute(stmt)).all()
    return [
        SearchResult(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            document_title=title,
            content=chunk.content,
            distance=float(dist),
        )
        for chunk, title, dist in rows
    ]
