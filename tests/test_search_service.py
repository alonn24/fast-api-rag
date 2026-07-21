import pytest

from app.db.models import Chunk, Document
from app.services.search import search_chunks


@pytest.mark.asyncio
async def test_search_chunks_orders_by_cosine_similarity(db_session):
    document = Document(title="Doc", source="test", doc_metadata={})
    db_session.add(document)
    await db_session.flush()

    close_vector = [1.0] + [0.0] * 1023
    far_vector = [0.0, 1.0] + [0.0] * 1022

    close_chunk = Chunk(document_id=document.id, chunk_index=0, content="close chunk", embedding=close_vector)
    far_chunk = Chunk(document_id=document.id, chunk_index=1, content="far chunk", embedding=far_vector)
    db_session.add_all([close_chunk, far_chunk])
    await db_session.commit()

    results = await search_chunks(db_session, query_embedding=close_vector, top_k=2)

    assert len(results) == 2
    assert results[0].content == "close chunk"
    assert results[0].distance < results[1].distance
    assert results[0].document_title == "Doc"


@pytest.mark.asyncio
async def test_search_chunks_respects_top_k(db_session):
    document = Document(title="Doc", source="test", doc_metadata={})
    db_session.add(document)
    await db_session.flush()

    for i in range(5):
        vector = [0.0] * 1024
        vector[i] = 1.0
        db_session.add(Chunk(document_id=document.id, chunk_index=i, content=f"chunk {i}", embedding=vector))
    await db_session.commit()

    query_vector = [0.0] * 1024
    query_vector[0] = 1.0
    results = await search_chunks(db_session, query_embedding=query_vector, top_k=3)

    assert len(results) == 3
