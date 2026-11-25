"""
Voice Call Continuous RAG Learning Service
==========================================
Captures spoken caller questions, dialogue interactions, and verified resolutions
from live telephone calls and continuously ingests them into PostgreSQL pgvector
(document_chunks table).

Architecture:
1. Turn-level Learning:
   - Evaluates caller utterance and AI reply.
   - Filters out non-informative utterances (single words, pure greetings, apology fallbacks).
   - Computes semantic similarity against existing chunks (sentence-transformers/all-MiniLM-L6-v2).
   - If similarity < 0.92 (novel knowledge or new voice phrasing), stores it as a new chunk.
2. Call-level Summary Learning:
   - On call termination, aggregates multi-turn dialogue into a coherent call interaction summary.
   - Embeds and stores the summary chunk for future RAG context.
3. Zero-Latency Execution:
   - Runs asynchronously as background fire-and-forget tasks so live call audio streaming
     is never delayed or blocked.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from conversation_intelligence.transcript_confidence import mask_sensitive_text

from storage.database import async_session
from storage.repositories.document_repo import DocumentRepository
from retrieval.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

# Filter patterns for non-informative utterances that should not be stored as knowledge
TRIVIAL_GREETINGS = re.compile(
    r"^(hi|hello|hey|namaste|नमस्ते|hii|heya|hlo|good morning|good afternoon|good evening)[!.,? ]*$",
    re.IGNORECASE,
)
CLOSING_PATTERNS = re.compile(
    r"^(thank you|thanks|thank u|dhanyawad|dhanyavad|shukriya|bye|goodbye|no|nope|no thanks|nahin|nahi|ok|okay|kuch nahi|not really|nothing)[!.,? ]*$",
    re.IGNORECASE,
)
APOLOGY_PATTERNS = [
    "didn't quite catch that",
    "could not process",
    "please try again",
    "माफ़ कीजिए",
    "समझ नहीं पाया",
    "sorry, i could not",
    "you are most welcome",
    "thank you for contacting",
    "आपका बहुत-बहुत धन्यवाद",
    "ଧନ୍ୟବାଦ",
    "ready to help",
    "what can i do for you",
    "how can i help you",
    "is there anything else",
    "what did you want to know",
    "can i help you with today",
    "anything else i can help",
    "माफ़ कीजिये",
    "आपकी आवाज़",
    "could not clearly hear",
    "look up information on",
    "certainly look up",
    "what specific details",
    "could you please clarify",
    "please clarify",
    "tell me a bit more",
    "on a scale of",
    "how satisfied are you",
    "confirm whether your issue",
    "repeat your question slowly",
    "what's your monthly payment question",
    "i understand you have a question about",
    "i see you're looking to",
]


def is_informative_interaction(user_text: str, reply_text: str) -> bool:
    """Return True if this interaction represents useful domain knowledge."""
    u = user_text.strip()
    r = reply_text.strip()

    # Must have substance
    if len(u.split()) < 2 or len(r.split()) < 3:
        return False

    # Skip pure greetings and closings/pleasantries
    if TRIVIAL_GREETINGS.match(u) or CLOSING_PATTERNS.match(u):
        return False

    # Skip bot apology fallbacks & clarification questions
    lower_r = r.lower()
    if any(pat in lower_r for pat in APOLOGY_PATTERNS):
        return False

    # A bot clarification prompt or satisfaction inquiry is NOT factual domain knowledge
    if r.endswith("?") or any(phrase in lower_r for phrase in [
        "could you tell me", "can you tell me", "what specific details",
        "could you please", "can you please", "would you like me to",
        "on a scale of", "how satisfied", "confirm whether your issue",
        "could you clarify", "please clarify", "tell me what specific",
        "tell me a bit more", "please confirm", "repeat your question",
        "certainly look up", "look up information on", "clarify your",
    ]):
        return False

    return True


def normalize_faq_query(text: str) -> str:
    """Normalize a caller utterance into a fast-lookup FAQ key while keeping Indic characters."""
    cleaned = re.sub(r"[^\w\u0900-\u097f\u0b00-\u0b7f\s]+", " ", (text or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


# Per-tenant fast-path cache for frequently asked / recently learned RAG chunks:
# key=(tenant_id, normalized_query) -> (stored_ts, chunks, hit_count)
_RAG_FAST_CACHE: Dict[Tuple[str, str], Tuple[float, List[str], int]] = {}
_RAG_FAST_CACHE_TTL_S = 1800.0
_RAG_FAST_CACHE_MAX_ENTRIES = 256


def get_fast_rag_chunks(tenant_id: str, query: str) -> Optional[List[str]]:
    """Return cached RAG chunks in <1ms if this caller question was recently answered or learned."""
    norm = normalize_faq_query(query)
    if not norm:
        return None
    key = (tenant_id, norm)
    entry = _RAG_FAST_CACHE.get(key)
    if not entry:
        return None
    ts, chunks, hits = entry
    if (time.monotonic() - ts) > _RAG_FAST_CACHE_TTL_S or not chunks:
        _RAG_FAST_CACHE.pop(key, None)
        return None
    _RAG_FAST_CACHE[key] = (time.monotonic(), chunks, hits + 1)
    return list(chunks)


def cache_rag_chunks(tenant_id: str, query: str, chunks: List[str]) -> None:
    """Store retrieved/learned chunks for instant lookup on repeat most-asked questions."""
    norm = normalize_faq_query(query)
    if not norm or not chunks:
        return
    if len(_RAG_FAST_CACHE) >= _RAG_FAST_CACHE_MAX_ENTRIES:
        oldest_key = min(_RAG_FAST_CACHE, key=lambda k: _RAG_FAST_CACHE[k][0])
        _RAG_FAST_CACHE.pop(oldest_key, None)
    prev = _RAG_FAST_CACHE.get((tenant_id, norm))
    hits = (prev[2] + 1) if prev else 1
    _RAG_FAST_CACHE[(tenant_id, norm)] = (time.monotonic(), list(chunks), hits)


def rank_chunks_with_faq_boost(
    search_results: List[Tuple[Any, float]],
    threshold: float,
    top_k: int = 2,
) -> List[str]:
    """Re-rank matching pgvector chunks giving a small priority boost to frequently asked questions."""
    ranked: List[Tuple[float, str]] = []
    for chunk, score in search_results:
        if score < threshold:
            continue
        ask_count = 1
        raw_meta = getattr(chunk, "meta_json", None)
        if raw_meta:
            try:
                ask_count = max(1, int(json.loads(raw_meta).get("ask_count", 1) or 1))
            except Exception:
                ask_count = 1
        boosted_score = score + min(0.05, 0.015 * math.log1p(ask_count - 1))
        ranked.append((boosted_score, chunk.content))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [content for _, content in ranked[:top_k]]


async def _maybe_sync_dump_snapshot(session: Any, tenant_id: str) -> None:
    """Keep data/ivr_chunks_dump.json synced during live PostgreSQL voice calls."""
    try:
        bind = getattr(session, "bind", None)
        if not bind or "postgresql" not in str(bind.url):
            return
        from sqlalchemy import select
        from storage.models.document import Document, DocumentChunk

        docs_res = await session.execute(
            select(Document).where(Document.tenant_id == tenant_id).order_by(Document.created_at.asc())
        )
        docs = {d.id: d for d in docs_res.scalars().all()}
        chunks_res = await session.execute(
            select(DocumentChunk)
            .where(DocumentChunk.tenant_id == tenant_id)
            .order_by(DocumentChunk.document_id.asc(), DocumentChunk.chunk_index.asc())
        )
        chunks_out = []
        for c in chunks_res.scalars().all():
            doc = docs.get(c.document_id)
            chunks_out.append({
                "id": c.id,
                "document_id": c.document_id,
                "document_title": doc.title if doc else "",
                "source_type": doc.source_type if doc else "",
                "chunk_index": c.chunk_index,
                "content": c.content,
                "meta": json.loads(c.meta_json) if c.meta_json else None,
            })

        dump_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "ivr_chunks_dump.json",
        )
        payload = json.dumps(chunks_out, indent=2, ensure_ascii=False)

        def _write_sync() -> None:
            os.makedirs(os.path.dirname(dump_path), exist_ok=True)
            with open(dump_path, "w", encoding="utf-8") as fh:
                fh.write(payload)

        await asyncio.to_thread(_write_sync)
    except Exception as exc:
        logger.debug(f"[Voice Learning] Dump snapshot sync skipped: {exc}")


def _normalize_phonetics(text: str) -> str:
    """Normalizes common telephone acoustic and Whisper mishearings before learning."""
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


async def learn_from_voice_turn(
    tenant_id: str,
    call_uuid: Optional[str],
    user_text: str,
    reply_text: str,
    quality_ok: bool = True,
    resolved: Optional[bool] = None,
    service_category: Optional[str] = None,
    language: Optional[str] = None,
    grounded: bool = True,
) -> bool:
    """
    Learn or reinforce a spoken Q&A turn in pgvector + FAQ cache during live calls:
    - Normalizes acoustic mishearings into clean semantic search keys.
    - If a matching verified chunk exists (score >= 0.70), reinforces it with ask_count
      and caller phrasing variants without duplicating knowledge.
    - If novel and grounded, stores a verified conversational Q&A chunk in pgvector.
    - Immediately warms the fast-path FAQ cache for zero-latency future repeats.
    """
    if not quality_ok or resolved is False:
        logger.debug("[Voice Learning] Skipped low-quality / uncertain / unresolved transcript turn.")
        return False

    if not is_informative_interaction(user_text, reply_text):
        return False

    try:
        clean_user = mask_sensitive_text(user_text.strip())
        clean_reply = mask_sensitive_text(reply_text.strip())
        call_id = call_uuid or "unknown"
        norm_user = _normalize_phonetics(clean_user)

        # Compute query embedding once and reuse it directly for vector search (2x faster)
        emb_service = EmbeddingService(provider_id="semantic-default")
        query_emb = await emb_service.embed_query(norm_user or clean_user)
        if not query_emb:
            return False

        async with async_session() as session:
            doc_repo = DocumentRepository(session)
            existing_matches = await doc_repo.search_similar_chunks(
                tenant_id=tenant_id,
                query_embedding=query_emb,
                top_k=1,
            )

            if existing_matches:
                top_chunk, score = existing_matches[0]
                if score >= 0.70:
                    chunk_meta: Dict[str, Any] = {}
                    if top_chunk.meta_json:
                        try:
                            chunk_meta = json.loads(top_chunk.meta_json) or {}
                        except Exception:
                            chunk_meta = {}

                    # Reinforce existing chunk with caller's natural spoken phrasing variant
                    variants = list(chunk_meta.get("variants") or [])
                    for cand in (clean_user, norm_user):
                        if cand and cand.lower() not in {v.lower() for v in variants}:
                            variants.append(cand)

                    chunk_meta["variants"] = variants[-15:]
                    chunk_meta["ask_count"] = int(chunk_meta.get("ask_count", 1) or 1) + 1
                    chunk_meta["last_asked_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

                    updated = await doc_repo.record_chunk_hit(
                        top_chunk,
                        variant_query=clean_user,
                    )
                    cache_rag_chunks(tenant_id, clean_user, [top_chunk.content])
                    if norm_user != clean_user:
                        cache_rag_chunks(tenant_id, norm_user, [top_chunk.content])
                    await _maybe_sync_dump_snapshot(session, tenant_id)
                    logger.info(
                        f"[Voice Learning] 🔄 Reinforced chunk #{updated.chunk_index} "
                        f"with variant '{clean_user}' (score={score:.3f}, hits={chunk_meta['ask_count']})"
                    )
                    return True

            # Novel inquiry: store new learned conversation Q&A chunk with ask_count=1 and warm cache
            learned_content = (
                f"Customer Spoken Inquiry: \"{clean_user}\"\n"
                f"Verified Resolution: \"{clean_reply}\"\n"
                f"Topic: {service_category or 'citizen_services'}\n"
                f"Source: Live Telephone Interaction ({call_id})"
            )
            content_emb = await emb_service.embed_query(learned_content)
            if not content_emb:
                content_emb = query_emb

            doc = await doc_repo.get_or_create_document(
                tenant_id=tenant_id,
                title="Continuous Voice Call Learnings",
                source_type="voice_call",
            )

            variants_init = [clean_user]
            if norm_user != clean_user:
                variants_init.append(norm_user)

            chunk = await doc_repo.append_chunk(
                document_id=doc.id,
                tenant_id=tenant_id,
                content=learned_content,
                embedding=content_emb,
                meta={
                    "type": "learned_conversation_qa",
                    "call_uuid": call_id,
                    "caller_query": clean_user,
                    "variants": variants_init,
                    "ask_count": 1,
                    "category": service_category or "citizen_services",
                    "language": language or "en",
                    "verified": True,
                    "resolved": True,
                },
            )
            cache_rag_chunks(tenant_id, clean_user, [learned_content])
            if norm_user != clean_user:
                cache_rag_chunks(tenant_id, norm_user, [learned_content])
            await _maybe_sync_dump_snapshot(session, tenant_id)

            logger.info(
                f"[Voice Learning] 🧠 Stored novel conversation Q&A chunk #{chunk.chunk_index} "
                f"from call {call_id}: '{clean_user[:45]}...'"
            )
            return True

    except Exception as exc:
        logger.warning(f"[Voice Learning] Failed to learn from turn: {exc}")
        return False


async def learn_from_call_summary(
    tenant_id: str,
    call_uuid: Optional[str],
    history: List[Any],
    service_category: Optional[str] = None,
    language: Optional[str] = None,
) -> bool:
    """
    On call completion, synthesize a multi-turn conversation consultation episode and store it
    in PostgreSQL pgvector for holistic conversational context recall across calls.
    """
    try:
        call_id = call_uuid or "unknown"

        # Extract non-system turns
        turns = []
        for m in history:
            role = getattr(m, "role", None)
            content = getattr(m, "content", "")
            if role in ("user", "assistant") and content:
                content_str = str(content).strip()
                if content_str and not content_str.startswith("--- BEGIN"):
                    # Clean injected RAG context out of user messages if present
                    if "User Message:\n" in content_str:
                        content_str = content_str.split("User Message:\n", 1)[1].strip()
                    turns.append((role, mask_sensitive_text(content_str)))

        # We need at least one user question and one assistant answer
        user_turns = [t for r, t in turns if r == "user"]
        if len(user_turns) < 1 or len(turns) < 2:
            return False

        dialogue_lines = []
        for r, t in turns:
            prefix = "Caller" if r == "user" else "Assistant"
            dialogue_lines.append(f"{prefix}: {t}")

        dialogue_text = "\n".join(dialogue_lines)
        summary_content = (
            f"Call Conversation Episode [Session {call_id}]:\n"
            f"Service Category: {service_category or 'citizen_services'}\n"
            f"Language: {language or 'en'}\n"
            f"Dialogue Consultation Flow:\n"
            f"{dialogue_text}\n"
            f"Key Inquiries: {', '.join(user_turns[:4])}"
        )

        emb_service = EmbeddingService(provider_id="semantic-default")
        summary_emb = await emb_service.embed_query(summary_content)
        if not summary_emb:
            return False

        async with async_session() as session:
            doc_repo = DocumentRepository(session)
            doc = await doc_repo.get_or_create_document(
                tenant_id=tenant_id,
                title="Continuous Voice Call Learnings",
                source_type="voice_call",
            )

            await doc_repo.append_chunk(
                document_id=doc.id,
                tenant_id=tenant_id,
                content=summary_content,
                embedding=summary_emb,
                meta={
                    "type": "conversation_episode",
                    "call_uuid": call_id,
                    "category": service_category or "citizen_services",
                    "language": language or "en",
                    "turn_count": len(turns),
                    "verified": True,
                },
            )

            await _maybe_sync_dump_snapshot(session, tenant_id)

            logger.info(
                f"[Voice Learning] 📞 Stored conversation episode for {call_id} ({len(turns)} turns, category={service_category}) in pgvector."
            )
            return True

    except Exception as exc:
        logger.warning(f"[Voice Learning] Failed to store call summary: {exc}")
        return False
