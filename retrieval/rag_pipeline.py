from typing import List, Optional
from storage.models.document import DocumentChunk
from retrieval.search import SemanticSearchService
from platform_core.contracts.providers import BaseRerankerAdapter
from providers.registry import provider_registry


class RAGPipeline:
    """Orchestrates query embedding, retrieval of tenant chunks, reranking, and context augmentation."""

    def __init__(
        self,
        search_service: SemanticSearchService,
        reranker: Optional[BaseRerankerAdapter] = None,
    ):
        self.search_service = search_service
        self.reranker = reranker or provider_registry.get_reranker("default")

    async def retrieve_context(
        self,
        tenant_id: str,
        query: str,
        top_k: int = 4,
        min_similarity_score: float = 0.30,
    ) -> str:
        import re
        from services.knowledge.retrieval_service import HybridRetrievalService

        # Acoustic phonetic normalization before vector search
        norm_q = query or ""
        norm_q = re.sub(r'(?i)\b(?:madurai|to\s+badraheel|badraheel|subadra|suvadra|suvardha)\s*(?:scheme|team|stream|screen|scene)?\b', 'Subhadra Scheme', norm_q)
        norm_q = re.sub(r'(?i)\bsubhadra\s+(?:team|stream|screen|scene)\b', 'Subhadra Scheme', norm_q)
        norm_q = re.sub(r'(?i)\b(?:cur|car|cor|kas|kos|cas)[\w]*s?atificat\w*\b', 'caste certificate', norm_q)
        norm_q = re.sub(r'(?i)\b(?:cost|cast|coast|cars)\s+certificates?\b', 'caste certificate', norm_q)
        norm_q = re.sub(r'(?i)\bcertificat\w*\b', 'certificate', norm_q)

        augmented_query = HybridRetrievalService._augment_cross_lingual_query(norm_q)
        results = await self.search_service.hybrid_search(
            tenant_id=tenant_id,
            query=augmented_query,
            top_k=top_k,
        )
        if not results:
            return ""

        # Filter out irrelevant chunks that do not meet the minimum similarity threshold
        relevant_results = [(chunk, score) for chunk, score in results if score >= min_similarity_score]
        if not relevant_results:
            return ""

        chunks = [chunk for chunk, score in relevant_results]

        context_blocks = []
        for idx, chunk in enumerate(chunks[:top_k], start=1):
            context_blocks.append(f"[Source {idx}]: {chunk.content}")

        header = "--- BEGIN UNTRUSTED RETRIEVED CONTEXT ---"
        footer = "--- END RETRIEVED CONTEXT ---"
        return f"{header}\n" + "\n\n".join(context_blocks) + f"\n{footer}"
