from typing import List, Union, Optional
from pydantic import BaseModel, Field


class EmbeddingRequestSchema(BaseModel):
    input: Union[str, List[str]]
    model: str
    user: Optional[str] = None


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingUsage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingResponseSchema(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage
