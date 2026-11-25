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
    (("phone", "call", "contact", "helpline", "email", "mail", "address", "location", "kendrapara", "kalabuda", "website", "नंबर", "फ़ोन", "संपर्क", "ईमेल", "पता", "ଠିକଣା", "ଫୋନ", "ଯୋଗାଯୋଗ", "ଇମେଲ"), "jana seva kendra phone number address email website kalabuda kendrapara 1234567890 support@janasevaodisha.com"),
    (("timing", "timings", "hours", "time", "open", "closing", "sunday", "working days", "समय", "खुलने", "खुला", "କେତେବେଳେ", "ଖୋଲେ", "ସମୟ"), "jana seva kendra operating hours timings monday to saturday 9:00 am to 6:00 pm sunday closed"),
    (("government", "private", "official", "disclaimer", "authority", "approve", "independent", "सरकारी", "प्राइवेट", "ଅନୁମୋଦନ", "ସରକାରୀ"), "jana seva odisha independent service provider not government organisation private facilitation"),
    (("tools", "pdf", "jpg", "photo", "resize", "word", "convert", "टूल", "फोटो", "ଫଟୋ"), "jpg to pdf word to pdf resize photos document utilities"),
    (("सुभद्रा", "योजना", "महिला", "सहायता", "yojana", "scheme", "schemes", "50000", "50,000", "2.5", "250000", "18-60", "18 to 60", "income limit", "age limit", "benefit", "money", "installment", "dbt", "subadra", "subhadra team", "how much money", "money can be given", "given to women", "paisa", "tanka", "tell me about subhadra", "ସୁଭଦ୍ରା", "ଯୋଜନା", "ମହିଳା"), "subhadra yojana scheme women financial assistance 50000 beneficiary 18 to 60 years income 2.5 lakh dbt odisha resident"),
    (("two pan", "second pan", "lost pan", "duplicate pan", "reprint pan", "दो पैन", "दूसरा पैन", "पैन खो गया", "ଦୁଇଟି ପ୍ୟାନ", "ହଜିଗଲା"), "second pan card illegal section 272b income tax penalty 10000 duplicate reprint lost pan"),
    (("voter", "epic", "form 6", "form 8", "election card", "voter list", "electoral roll", "फॉर्म 6", "फॉर्म 8", "मतदाता", "ଭୋଟର", "ଫର୍ମ"), "voter id card form 6 new registration form 8 correction electoral roll eci"),
    (("notary", "notarisation", "registration", "affidavit", "agreement", "power of attorney", "poa", "will", "नोटरी", "रजिस्ट्री", "सत्यपाଠ", "ଆଫିଡେଭିଟ୍", "ସତ୍ୟପାଠ"), "affidavit agreement power of attorney poa will notary notarisation registration distinction stamp paper legal"),
    (("bonafide", "student", "college", "school", "institution", "employer", "बोनाफाइड", "छात्र", "କଲେଜ", "ସ୍କୁଲ", "ବୋନାଫାଇଡ୍"), "bonafide certificate school college university employer educational institution enrollment"),
    (("जाति", "caste", "sc", "st", "obc", "sebc", "jati", "ror", "patta", "khaitan", "सामान्य", "general", "ଜାତି", "ଜମି ପଟ୍ଟା"), "caste certificate sc st obc sebc ror land record patta tahasil revenue inspector lineage general category"),
    (("आय", "income", "salary", "slip", "aay", "itr", "revenue", "income certificate", "annual income", "ଆୟ", "ଦରମା"), "income certificate annual family income salary slip itr tahasildar revenue inspector self declaration"),
    (("validity", "valid", "expire", "expiry", "वैधता", "ବୈଧତା", "ମିଆଦ"), "income certificate validity period 3 years three financial years tahasildar odisha"),
    (("bank account", "account", "single holder", "seeded", "खाता", "ବ୍ୟାଙ୍କ ଖାତା"), "subhadra scheme bank account single-holder individual aadhaar seeded direct benefit transfer dbt passbook"),
    (("fee", "fees", "charge", "free", "cost", "शुल्क", "ମାଗଣା"), "subhadra scheme application form free of cost zero fee no charge government"),
    (("निवास", "residence", "domicile", "niwas", "basasthana", "बिजली बिल", "ବାସସ୍ଥାନ", "ବିଦ୍ୟୁତ"), "residence certificate domicile permanent resident odisha ror electricity bill tahasil enquiry"),
    (("आधार", "aadhaar", "biometric", "iris", "fingerprint", "mobile link", "uidai", "बायोमेट्रिक", "ଆଧାର", "ବାୟୋମେଟ୍ରିକ"), "aadhaar card biometric update fingerprints iris in person visit authorized center uidai demographic mobile linking"),
    (("शिकायत", "टिकट", "लोकपाल", "कम्प्लेन", "status", "track", "complaint", "shikayat", "ଅଭିଯୋଗ", "ଟିକେଟ୍", "ସ୍ଥିତି"), "customer complaint grievance ticket status tracking sla 24-48 hours resolution"),
    (("ऑर्डर", "डिलीवरी", "पार्सल", "ट्रैक", "डिस्पैच", "ଅର୍ଡର", "ଡେଲିଭରି"), "order id dispatch card shipment delivery tracking"),
)


class HybridRetrievalService:
    """RAG-first hybrid retriever over tenant-isolated pgvector knowledge documents."""

    def __init__(self, embedding_provider_id: str = "semantic-default") -> None:
        self.embedding_provider_id = embedding_provider_id

    @staticmethod
    def _normalize_acoustic_phonetics(text: str) -> str:
        """Normalizes common telephone acoustic and Whisper mishearings before search."""
        if not text:
            return ""
        t = text
        t = re.sub(r'(?i)\b(?:madurai|to\s+badraheel|badraheel|subadra|suvadra|suvardha)\s*(?:scheme|team|stream|screen|scene)?\b', 'Subhadra Scheme', t)
        t = re.sub(r'(?i)\bsubhadra\s+(?:team|stream|screen|scene)\b', 'Subhadra Scheme', t)
        t = re.sub(r'(?i)\b(?:cur|car|cor|kas|kos|cas)[\w]*s?atificat\w*\b', 'caste certificate', t)
        t = re.sub(r'(?i)\b(?:cost|cast|coast|cars)\s+certificates?\b', 'caste certificate', t)
        t = re.sub(r'(?i)\bcertificat\w*\b', 'certificate', t)
        t = re.sub(r'(?i)\bpan\s+cart\b', 'PAN card', t)
        t = re.sub(r'(?i)\b(?:water|voter)\s+(?:id|cart|card)\b', 'voter ID card', t)
        return t

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
            # High-precision Q&A resolution boost
            qa_boost = 0.04 if ("Verified Resolution:" in content and lex_score >= 0.25) else 0.0

            hybrid_score = (0.74 * float(sem_score)) + (0.26 * lex_score) + faq_boost + verified_boost + cat_boost + qa_boost
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
        clean_query = self._normalize_acoustic_phonetics((query or "").strip())
        if not clean_query:
            eval_empty = evaluate_relevance([])
            return RetrievedContext(chunks=[], chunk_texts=[], scored_chunks=[], evaluation=eval_empty)

        strong_th = strong_threshold if strong_threshold is not None else float(os.getenv("RAG_MATCH_THRESHOLD", "0.55"))
        partial_th = partial_threshold if partial_threshold is not None else max(0.34, strong_th - 0.20)

        # Do not fast-cache live transactional or complaint status queries so dynamic CRM status changes are always fresh
        is_live_crm_query = bool(
            preferred_category == "crm_complaint_status"
            or any(kw in clean_query.lower() for kw in ("complaint", "ticket", "status", "cmp/", "ord-", "order", "status of"))
        )

        if use_cache and not is_live_crm_query:
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

                # Multi-Query Fusion: Search with both augmented query and raw query
                pool_size = max(top_k * 4, 16)
                raw_results_map: Dict[Any, Tuple[Any, float]] = {}

                # 1. Augmented Cross-Lingual Query
                aug_results = await search_service.search(
                    tenant_id=tenant_id,
                    query=search_query,
                    top_k=pool_size,
                )
                for c, s in aug_results:
                    c_id = getattr(c, "id", None) or str(getattr(c, "content", ""))[:64]
                    raw_results_map[c_id] = (c, float(s))

                # 2. Raw User Query (for exact literal terms)
                if search_query != clean_query:
                    raw_direct = await search_service.search(
                        tenant_id=tenant_id,
                        query=clean_query,
                        top_k=pool_size,
                    )
                    for c, s in raw_direct:
                        c_id = getattr(c, "id", None) or str(getattr(c, "content", ""))[:64]
                        if c_id not in raw_results_map or float(s) > raw_results_map[c_id][1]:
                            raw_results_map[c_id] = (c, float(s))

                combined_raw = list(raw_results_map.values())
                scored_chunks = self.hybrid_rerank(
                    query=clean_query,
                    raw_results=combined_raw,
                    preferred_category=preferred_category,
                    top_k=top_k,
                )
        except Exception as exc:
            logger.debug(f"[HybridRetrieval] DB/pgvector search skipped cleanly: {exc}")
            scored_chunks = []

        eval_res = evaluate_relevance(scored_chunks, strong_threshold=strong_th, partial_threshold=partial_th)
        chunk_objs = [c for c, s in scored_chunks if s >= partial_th]
        chunk_texts = [getattr(c, "content", "") for c in chunk_objs if getattr(c, "content", "")]

        if use_cache and not is_live_crm_query and chunk_texts and eval_res.tier == RetrievalConfidenceTier.STRONG:
            cache_rag_chunks(tenant_id, clean_query, chunk_texts)

        return RetrievedContext(
            chunks=chunk_objs,
            chunk_texts=chunk_texts,
            scored_chunks=scored_chunks,
            evaluation=eval_res,
            cache_hit=False,
        )

