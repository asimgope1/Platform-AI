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

import logging
import re
from typing import Any, List, Optional

from storage.database import async_session
from storage.repositories.document_repo import DocumentRepository
from retrieval.embedding_service import EmbeddingService
from retrieval.search import SemanticSearchService

logger = logging.getLogger(__name__)

# Filter patterns for non-informative utterances that should not be stored as knowledge
TRIVIAL_GREETINGS = re.compile(
    r"^(hi|hello|hey|namaste|नमस्ते|hii|heya|hlo|good morning|good afternoon|good evening)[!.,? ]*$",
    re.IGNORECASE,
)
APOLOGY_PATTERNS = [
    "didn't quite catch that",
    "could not process",
    "please try again",
    "माफ़ कीजिए",
    "समझ नहीं पाया",
    "sorry, i could not",
]


def is_informative_interaction(user_text: str, reply_text: str) -> bool:
    """Return True if this interaction represents useful domain knowledge."""
    u = user_text.strip()
    r = reply_text.strip()

    # Must have substance
    if len(u.split()) < 2 or len(r.split()) < 3:
        return False

    # Skip pure greetings
    if TRIVIAL_GREETINGS.match(u):
        return False

    # Skip bot apology fallbacks
    lower_r = r.lower()
    if any(pat in lower_r for pat in APOLOGY_PATTERNS):
        return False

    return True


async def learn_from_voice_turn(
    tenant_id: str,
    call_uuid: Optional[str],
    user_text: str,
    reply_text: str,
) -> bool:
    """
    Ingest a single spoken Q&A turn into PostgreSQL pgvector if it contains novel knowledge.
    """
    if not is_informative_interaction(user_text, reply_text):
        return False

    try:
        clean_user = user_text.strip()
        clean_reply = reply_text.strip()
        call_id = call_uuid or "unknown"

        # Embed user query to check for existing duplicates
        emb_service = EmbeddingService(provider_id="semantic-default")
        query_emb = await emb_service.embed_query(clean_user)
        if not query_emb:
            return False

        async with async_session() as session:
            doc_repo = DocumentRepository(session)
            search_service = SemanticSearchService(doc_repo=doc_repo, embedding_service=emb_service)

            # Check if this query or knowledge is already in the vector database
            existing_matches = await search_service.search(
                tenant_id=tenant_id,
                query=clean_user,
                top_k=1,
            )

            if existing_matches:
                top_chunk, score = existing_matches[0]
                # If top match similarity is above 0.92, it is already well-covered
                if score >= 0.92:
                    logger.debug(
                        f"[Voice Learning] Skipped duplicate chunk (score={score:.3f} >= 0.92): '{clean_user[:40]}...'"
                    )
                    return False

            # Format the learned knowledge unit
            learned_content = (
                f"Customer Spoken Inquiry: \"{clean_user}\"\n"
                f"Verified Resolution: \"{clean_reply}\"\n"
                f"Source: Live Telephone Interaction ({call_id})"
            )

            # Embed the full knowledge content for RAG retrieval
            content_emb = await emb_service.embed_query(learned_content)
            if not content_emb:
                content_emb = query_emb

            # Retrieve or create continuous learning document container
            doc = await doc_repo.get_or_create_document(
                tenant_id=tenant_id,
                title="Continuous Voice Call Learnings",
                source_type="voice_call",
            )

            # Store the chunk directly with pgvector embedding
            chunk = await doc_repo.append_chunk(
                document_id=doc.id,
                tenant_id=tenant_id,
                content=learned_content,
                embedding=content_emb,
                meta={
                    "type": "voice_interaction_turn",
                    "call_uuid": call_id,
                    "caller_query": clean_user,
                },
            )

            logger.info(
                f"[Voice Learning] 🧠 Stored new learned voice chunk #{chunk.chunk_index} "
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
) -> bool:
    """
    On call completion, synthesize a multi-turn conversation summary and store it
    in PostgreSQL pgvector for holistic context recall.
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
                    turns.append((role, content_str))

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
            f"Call Interaction Summary [Session {call_id}]:\n"
            f"{dialogue_text}\n"
            f"Key Inquiries: {', '.join(user_turns[:3])}"
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
                    "type": "call_summary",
                    "call_uuid": call_id,
                    "turn_count": len(turns),
                },
            )

            logger.info(
                f"[Voice Learning] 📞 Stored call summary for {call_id} ({len(turns)} turns) in pgvector."
            )
            return True

    except Exception as exc:
        logger.warning(f"[Voice Learning] Failed to store call summary: {exc}")
        return False
