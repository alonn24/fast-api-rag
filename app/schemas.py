import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class DocumentCreateRequest(BaseModel):
    title: str
    source: str
    content: str
    metadata: dict = Field(default_factory=dict)


class DocumentCreateResponse(BaseModel):
    id: uuid.UUID
    chunk_count: int


class DocumentGetResponse(BaseModel):
    id: uuid.UUID
    title: str
    source: str
    metadata: dict
    created_at: datetime
    chunk_count: int


class QueryRequest(BaseModel):
    question: str


class SourceRef(BaseModel):
    document_id: uuid.UUID
    document_title: str
    chunk_id: uuid.UUID
    content: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceRef]
