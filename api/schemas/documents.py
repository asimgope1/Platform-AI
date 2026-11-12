from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel


class DocumentUploadRequest(BaseModel):
    title: str
    content: str
    source_type: Optional[str] = "text"


class DocumentReindexRequest(BaseModel):
    title: Optional[str] = None
    content: str
    source_type: Optional[str] = None
    version: Optional[str] = "2.0"


class DocumentResponse(BaseModel):
    id: str
    tenant_id: str
    title: str
    source_type: str
    total_chunks: int
    created_at: datetime


class DocumentListResponse(BaseModel):
    object: str = "list"
    data: List[DocumentResponse]


class SearchRequestSchema(BaseModel):
    query: str
    top_k: Optional[int] = 4
    category: Optional[str] = None
    hybrid: Optional[bool] = True


class SearchResultItem(BaseModel):
    document_id: str
    chunk_index: int
    content: str
    score: float
    citation: Optional[str] = None


class SearchResponseSchema(BaseModel):
    object: str = "list"
    data: List[SearchResultItem]
