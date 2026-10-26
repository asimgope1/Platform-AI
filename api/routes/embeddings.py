from fastapi import APIRouter, Depends
from api.schemas.embeddings import EmbeddingRequestSchema, EmbeddingResponseSchema, EmbeddingData, EmbeddingUsage
from platform_core.security.authentication import get_current_tenant
from platform_core.security.tenant_context import TenantContext
from retrieval.embedding_service import EmbeddingService

router = APIRouter(prefix="/v1", tags=["Embeddings"])


@router.post("/embeddings", response_model=EmbeddingResponseSchema)
async def create_embeddings(
    payload: EmbeddingRequestSchema,
    context: TenantContext = Depends(get_current_tenant),
):
    """Generate vector embeddings for input strings."""
    texts = [payload.input] if isinstance(payload.input, str) else payload.input
    service = EmbeddingService(model=payload.model)
    embeddings = await service.embed_texts(texts)

    data = [
        EmbeddingData(
            object="embedding",
            embedding=emb,
            index=i,
        )
        for i, emb in enumerate(embeddings)
    ]

    return EmbeddingResponseSchema(
        object="list",
        data=data,
        model=payload.model,
        usage=EmbeddingUsage(
            prompt_tokens=sum(len(t.split()) for t in texts),
            total_tokens=sum(len(t.split()) for t in texts),
        ),
    )
