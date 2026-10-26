from typing import List, Dict, Any, Optional
from platform_core.contracts.providers import BaseRerankerAdapter


class MockReranker(BaseRerankerAdapter):
    """Deterministic reranker based on lexical keyword overlap and query presence."""

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        query_words = set(query.lower().split())
        scored = []

        for idx, doc in enumerate(documents):
            doc_words = set(doc.lower().split())
            intersection = query_words.intersection(doc_words)
            score = len(intersection) / max(1, len(query_words))
            scored.append({
                "index": idx,
                "document": doc,
                "relevance_score": round(score, 4),
            })

        scored.sort(key=lambda x: x["relevance_score"], reverse=True)
        limit = top_n if top_n is not None else len(scored)
        return scored[:limit]
