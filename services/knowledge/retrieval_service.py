"""
Hybrid Knowledge Retrieval Service
==================================
Primary factual retrieval engine combining:
  1. Sub-millisecond fast FAQ cache lookup
  2. Native pgvector cosine semantic search
  3. Lexical keyword/BM25-like overlap + verified source & ask_count reranking
  4. Multi-tier retrieval relevance evaluation
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, List, Optional, Tuple

from retrieval.embedding_service import EmbeddingService
from retrieval.search import SemanticSearchService
from services.evaluation.retrieval_evaluator import (
    RetrievalConfidenceTier,
    RetrievalEvaluation,
    evaluate_relevance,
)
from storage.database import async_session
from storage.repositories.document_repo import DocumentRepository
from telephony.voice_learning import cache_rag_chunks, get_fast_rag_chunks

logger = logging.getLogger(__name__)


@dataclass
class RetrievedContext:
    chunks: List[Any]
    chunk_texts: List[str]
    scored_chunks: List[Tuple[Any, float]]
    evaluation: RetrievalEvaluation
    cache_hit: bool = False


_CROSS_LINGUAL_ANCHORS = (
    (("केवाईसी", "वीडियो", "पैन", "आधार", "दस्तावेज", "कागजात", "dastavez", "କେୱାଇସି", "ଆଧାର", "ପ୍ୟାନ"), "video kyc pan card aadhaar documents verification"),
    (("यूपीआई", "रिफंड", "अटक", "फेल", "पेमेंट", "पैसे कट", "पैसा", "paisa", "paise", "kat gaya", "ୟୁପିଆଇ", "ରିଫଣ୍ଡ"), "upi payment failed money debited refund reversal limit"),
    (("कार्ड", "ब्लॉक", "खो", "चोरी", "पिन", "एटीएम", "kho gaya", "chori", "କାର୍ଡ", "ବ୍ଲକ", "ପିନ"), "lost stolen debit credit card block freeze atm green pin"),
    (("खाता", "अकाउंट", "बैलेंस", "बचत", "पासबुक", "चेक", "स्टेटमेंट", "khata", "ସେଭିଂସ", "ଖାତା", "ବାଲାନ୍ସ"), "savings account balance inquiry mini statement cheque book"),
    (("लोन", "ऋण", "ईएमआई", "ब्याज", "किस्त", "फोरक्लोजर", "karz", "byaj", "ଋଣ", "ଇଏମଆଇ"), "loan emi repayment interest foreclosure auto debit"),
    (("शिकायत", "टिकट", "लोकपाल", "कम्प्लेन", "असंतुष्ट", "shikayat", "ଅଭିଯୋଗ"), "customer complaint grievance ticket escalation ombudsman"),
    (("धोखाधड़ी", "फ्रॉड", "ओटीपी", "ठगी", "हैक", "अनधिकृत", "dhokha", "ଠକେଇ", "ଫ୍ରଡ"), "fraud phishing unauthorized transaction otp security block"),
    (("ऑर्डर", "डिलीवरी", "पार्सल", "ट्रैक", "डिस्पैच", "ଅର୍ଡର", "ଡେଲିଭରି"), "order id dispatch card shipment delivery tracking"),
)


class HybridRetrievalService:
    """RAG-first hybrid retriever over tenant-isolated pgvector knowledge documents."""

    def __init__(self, embedding_provider_id: str = "semantic-default") -> None:
        self.embedding_provider_id = embedding_provider_id

    @staticmethod
    def _augment_cross_lingual_query(query: str) -> str:
        """Append English semantic anchors for Indic (Hindi/Odia) & Hinglish queries to bridge English embeddings."""
        lower_q = (query or "").lower()
        extras: List[str] = []
        for triggers, anchor in _CROSS_LINGUAL_ANCHORS:
            if any(t in lower_q for t in triggers):
                extras.append(anchor)
        return f"{query} {' '.join(extras)}".strip() if extras else query

    @staticmethod
    def _lexical_overlap(query: str, content: str) -> float:
        q_tokens = {w for w in re.findall(r"[a-zA-Z0-9\u0900-\u097f\u0b00-\u0b7f]{2,}", query.lower())}
        if not q_tokens:
            return 0.0
        c_tokens = {w for w in re.findall(r"[a-zA-Z0-9\u0900-\u097f\u0b00-\u0b7f]{2,}", content.lower())}
        if not c_tokens:
            return 0.0
        return len(q_tokens.intersection(c_tokens)) / len(q_tokens)

    @classmethod
    def hybrid_rerank(
        cls,
        query: str,
        raw_results: List[Tuple[Any, float]],
        preferred_category: Optional[str] = None,
        top_k: int = 4,
    ) -> List[Tuple[Any, float]]:
        """
        Blend semantic cosine similarity with keyword overlap, verified authority bonus,
        preferred category affinity, and logarithmic ask_count popularity boost.
        """
        reranked: List[Tuple[Any, float]] = []
        for chunk, sem_score in raw_results:
            content = getattr(chunk, "content", "") or ""
            lex_score = cls._lexical_overlap(query, content)

            meta = {}
            raw_meta = getattr(chunk, "meta_json", None)
            if raw_meta:
                try:
                    meta = json.loads(raw_meta) or {}
                except Exception:
                    meta = {}

            # Exclude unverified unanswered candidate drafts from primary factual answers
            if meta.get("type") == "unanswered_candidate" and meta.get("verified") is False:
                continue

            ask_count = max(1, int(meta.get("ask_count", 1) or 1))
            faq_boost = min(0.05, 0.015 * math.log1p(ask_count - 1))
            verified_boost = 0.03 if meta.get("verified") is True else 0.0
            cat_boost = (
                0.03
                if preferred_category and str(meta.get("category", "")).lower() == preferred_category.lower()
                else 0.0
            )

            hybrid_score = (0.78 * float(sem_score)) + (0.22 * lex_score) + faq_boost + verified_boost + cat_boost
            reranked.append((chunk, round(min(1.0, hybrid_score), 4)))

        reranked.sort(key=lambda item: item[1], reverse=True)
        return reranked[:top_k]

    async def search(
        self,
        query: str,
        tenant_id: str,
        top_k: int = 4,
        preferred_category: Optional[str] = None,
        strong_threshold: Optional[float] = None,
        partial_threshold: Optional[float] = None,
        use_cache: bool = True,
        session_factory: Optional[Any] = None,
    ) -> RetrievedContext:
        clean_query = (query or "").strip()
        if not clean_query:
            eval_empty = evaluate_relevance([])
            return RetrievedContext(chunks=[], chunk_texts=[], scored_chunks=[], evaluation=eval_empty)

        strong_th = strong_threshold if strong_threshold is not None else float(os.getenv("RAG_MATCH_THRESHOLD", "0.55"))
        partial_th = partial_threshold if partial_threshold is not None else max(0.38, strong_th - 0.15)

        if use_cache:
            cached_texts = get_fast_rag_chunks(tenant_id, clean_query)
            if cached_texts:
                synthetic_scored = [
                    (SimpleNamespace(content=t, meta_json=json.dumps({"verified": True, "cached": True})), 0.92)
                    for t in cached_texts[:top_k]
                ]
                eval_cached = evaluate_relevance(synthetic_scored, strong_threshold=strong_th, partial_threshold=partial_th)
                return RetrievedContext(
                    chunks=[c for c, _ in synthetic_scored],
                    chunk_texts=list(cached_texts[:top_k]),
                    scored_chunks=synthetic_scored,
                    evaluation=eval_cached,
                    cache_hit=True,
                )

        scored_chunks: List[Tuple[Any, float]] = []
        search_query = self._augment_cross_lingual_query(clean_query)
        db_factory = session_factory or async_session
        try:
            async with db_factory() as session:
                doc_repo = DocumentRepository(session)
                emb_service = EmbeddingService(provider_id=self.embedding_provider_id)
                search_service = SemanticSearchService(doc_repo=doc_repo, embedding_service=emb_service)
                raw_results = await search_service.search(
                    tenant_id=tenant_id,
                    query=search_query,
                    top_k=max(top_k * 2, 6),
                )
                scored_chunks = self.hybrid_rerank(
                    query=search_query,
                    raw_results=raw_results,
                    preferred_category=preferred_category,
                    top_k=top_k,
                )
        except Exception as exc:
            logger.debug(f"[HybridRetrieval] DB/pgvector search skipped cleanly: {exc}")
            scored_chunks = []

        eval_res = evaluate_relevance(scored_chunks, strong_threshold=strong_th, partial_threshold=partial_th)
        chunk_objs = [c for c, s in scored_chunks if s >= partial_th]
        chunk_texts = [getattr(c, "content", "") for c in chunk_objs if getattr(c, "content", "")]

        if use_cache and chunk_texts and eval_res.tier == RetrievalConfidenceTier.STRONG:
            cache_rag_chunks(tenant_id, clean_query, chunk_texts)

        return RetrievedContext(
            chunks=chunk_objs,
            chunk_texts=chunk_texts,
            scored_chunks=scored_chunks,
            evaluation=eval_res,
            cache_hit=False,
        )

