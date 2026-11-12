"""
Continuous Knowledge Learning Service (Level 1 + Level 2)
=========================================================
Separates retrieval reinforcement from verified knowledge creation:
  - Level 1 (Retrieval Learning): reinforces matched verified chunks with
    ask_count popularity and caller phrasing variants, without altering
    verified official policy text.
  - Level 2 (Candidate Knowledge Learning): flags unanswered or missing-policy
    questions as pending review drafts (verified=False) rather than letting
    unverified statements become authoritative bank policy.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from conversation_intelligence.transcript_confidence import mask_sensitive_text
from retrieval.embedding_service import EmbeddingService
from storage.database import async_session
from storage.models.document import DocumentChunk
from storage.repositories.document_repo import DocumentRepository
from telephony.voice_learning import _maybe_sync_dump_snapshot, learn_from_voice_turn

logger = logging.getLogger(__name__)


class KnowledgeLearningService:
    """Coordinates Level 1 retrieval feedback and Level 2 candidate knowledge drafting."""

    def __init__(self, embedding_provider_id: str = "semantic-default") -> None:
        self.embedding_provider_id = embedding_provider_id

    async def process_turn_feedback(
        self,
        tenant_id: str,
        call_uuid: Optional[str],
        user_text: str,
        reply_text: str,
        retrieval_tier: str = "strong",
        quality_ok: bool = True,
        resolved: Optional[bool] = None,
    ) -> bool:
        """
        Route turn learning safely:
          - If retrieval found nothing ('none'), log an unanswered candidate draft (Level 2)
            for human/policy review instead of teaching ungrounded LLM guesses as policy.
          - Otherwise run Level 1 turn reinforcement / variant tracking.
        """
        if not quality_ok:
            return False

        if retrieval_tier == "none":
            return await self.record_unanswered_query(
                tenant_id=tenant_id,
                call_uuid=call_uuid,
                user_text=user_text,
                provisional_reply=reply_text,
            )

        return await learn_from_voice_turn(
            tenant_id=tenant_id,
            call_uuid=call_uuid,
            user_text=user_text,
            reply_text=reply_text,
            quality_ok=quality_ok,
            resolved=resolved,
        )

    async def record_unanswered_query(
        self,
        tenant_id: str,
        call_uuid: Optional[str],
        user_text: str,
        provisional_reply: str = "",
    ) -> bool:
        """
        Level 2: Record an unanswered caller inquiry as a candidate knowledge draft
        with verified=False, status='pending_review' so official policy stays pure.
        """
        clean_q = mask_sensitive_text((user_text or "").strip())
        if len(clean_q.split()) < 2:
            return False

        try:
            emb_service = EmbeddingService(provider_id=self.embedding_provider_id)
            q_emb = await emb_service.embed_query(clean_q)
            if not q_emb:
                return False

            draft_content = (
                f"[CANDIDATE - PENDING REVIEW]\n"
                f"Unanswered Caller Question: \"{clean_q}\"\n"
                f"Provisional Assistant Reply: \"{mask_sensitive_text((provisional_reply or '').strip())}\"\n"
                f"Call UUID: {call_uuid or 'unknown'}"
            )

            async with async_session() as session:
                doc_repo = DocumentRepository(session)
                existing = await doc_repo.search_similar_chunks(
                    tenant_id=tenant_id,
                    query_embedding=q_emb,
                    top_k=1,
                )
                if existing and existing[0][1] >= 0.88:
                    chunk = existing[0][0]
                    await doc_repo.record_chunk_hit(chunk, variant_query=clean_q)
                    return True

                doc = await doc_repo.get_or_create_document(
                    tenant_id=tenant_id,
                    title="Unanswered Knowledge Candidates (Pending Review)",
                    source_type="candidate_draft",
                )
                await doc_repo.append_chunk(
                    document_id=doc.id,
                    tenant_id=tenant_id,
                    content=draft_content,
                    embedding=q_emb,
                    meta={
                        "type": "unanswered_candidate",
                        "verified": False,
                        "status": "pending_review",
                        "caller_query": clean_q,
                        "variants": [clean_q],
                        "ask_count": 1,
                        "call_uuid": call_uuid or "unknown",
                    },
                )
                await _maybe_sync_dump_snapshot(session, tenant_id)
                logger.info(f"[KnowledgeLearning] 📝 Flagged unanswered query for review: '{clean_q[:50]}'")
                return True
        except Exception as exc:
            logger.debug(f"[KnowledgeLearning] Candidate logging skipped: {exc}")
            return False

    async def approve_candidate_chunk(
        self,
        chunk_id: str,
        verified_content: str,
        category: str = "policies",
        topic: str = "reviewed_policy",
    ) -> Optional[Dict[str, Any]]:
        """Promote a reviewed draft chunk into verified authoritative knowledge and re-embed."""
        emb_service = EmbeddingService(provider_id=self.embedding_provider_id)
        emb = await emb_service.embed_query(verified_content)

        async with async_session() as session:
            chunk = await session.get(DocumentChunk, chunk_id)
            if not chunk:
                return None

            meta: Dict[str, Any] = {}
            if chunk.meta_json:
                try:
                    meta = json.loads(chunk.meta_json) or {}
                except Exception:
                    meta = {}

            meta.update({
                "type": "verified_policy",
                "category": category,
                "topic": topic,
                "verified": True,
                "status": "approved",
            })
            chunk.content = verified_content
            chunk.meta_json = json.dumps(meta, ensure_ascii=False)
            if emb:
                chunk.embedding = emb if len(emb) == 384 else None
                chunk.embedding_768 = emb if len(emb) == 768 else None
                chunk.embedding_json = json.dumps(emb)

            await session.commit()
            await session.refresh(chunk)
            await _maybe_sync_dump_snapshot(session, chunk.tenant_id)
            return {"id": chunk.id, "verified": True, "meta": meta}

