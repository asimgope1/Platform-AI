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
        top_k: int = 3,
        min_similarity_score: float = 0.30,
    ) -> str:
        # Retrieve candidate chunks (fetch slightly more if reranking)
        fetch_k = top_k * 2 if self.reranker else top_k
        results = await self.search_service.search(tenant_id=tenant_id, query=query, top_k=fetch_k)
        if not results:
            return ""

        # Filter out irrelevant chunks that do not meet the minimum similarity threshold
        relevant_results = [(chunk, score) for chunk, score in results if score >= min_similarity_score]
        if not relevant_results:
            return ""

        chunks = [chunk for chunk, score in relevant_results]

        # Rerank if reranker is available
        if self.reranker and len(chunks) > 1:
            try:
                reranked = await self.reranker.rerank(
                    query=query,
                    documents=[c.content for c in chunks],
                    top_n=top_k,
                )
                selected_chunks = [chunks[item["index"]] for item in reranked]
            except Exception:
                selected_chunks = chunks[:top_k]
        else:
            selected_chunks = chunks[:top_k]

        context_blocks = []
        for idx, chunk in enumerate(selected_chunks, start=1):
            context_blocks.append(f"[Source {idx}]: {chunk.content}")

        header = "--- BEGIN UNTRUSTED RETRIEVED CONTEXT ---"
        footer = "--- END RETRIEVED CONTEXT ---"
        return f"{header}\n" + "\n\n".join(context_blocks) + f"\n{footer}"
