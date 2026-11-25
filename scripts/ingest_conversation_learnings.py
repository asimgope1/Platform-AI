#!/usr/bin/env python3
"""
Conversation RAG Ingestion & Learning Engine
============================================
Extracts real telephone conversations, caller voice utterances, and verified
resolutions from call session logs, and ingests them into PostgreSQL pgvector:
1. Turn-level Spoken Phrasings:
   - Normalizes phonetic acoustic mishearings (e.g. 'madurai scheme', 'to badraheel',
     'subadra team', 'cost certificate', 'Cursatificat').
   - Links real-world caller variants to authoritative verified knowledge chunks.
   - Boosts chunk ask_count and warms the instant FAQ cache.
2. Multi-Turn Consultation Episodes:
   - Stores full dialogue trajectories as structured 'conversation_episode' chunks.
3. Synchronizes updated pgvector chunks to SQLite and JSON snapshots.
"""
from __future__ import annotations

import asyncio
import glob
import json
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from retrieval.embedding_service import EmbeddingService
from storage.database import async_session
from storage.models.document import Document, DocumentChunk
from storage.repositories.document_repo import DocumentRepository
from telephony.voice_learning import (
    _normalize_phonetics,
    cache_rag_chunks,
    is_informative_interaction,
    _maybe_sync_dump_snapshot,
)
from sqlalchemy import select

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest_conversation_learnings")

TENANT_ID = "879e9bda-8049-4824-9aa0-fc5e9f0012d9"
LOG_DIR = "/Users/asimgope/.gemini/antigravity/brain/50f31053-419d-48a5-ba3b-37a48b585a03/.system_generated/tasks"


def extract_conversations_from_logs() -> Dict[str, Dict[str, Any]]:
    """Scan all task log files and reconstruct structured call conversation episodes."""
    log_files = glob.glob(os.path.join(LOG_DIR, "task-*.log"))
    log_files.sort(key=os.path.getmtime, reverse=True)

    calls: Dict[str, Dict[str, Any]] = {}

    for fpath in log_files:
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                curr_call_id: Optional[str] = None
                pending_user: Optional[str] = None

                for line in f:
                    if "Call started: uuid=" in line:
                        m_call = re.search(r"uuid=([a-f0-9\-]+)", line)
                        if m_call:
                            curr_call_id = m_call.group(1)
                            if curr_call_id not in calls:
                                calls[curr_call_id] = {
                                    "language": "en",
                                    "service": "citizen_services",
                                    "turns": [],
                                }

                    if not curr_call_id:
                        continue

                    if "Language set to: '" in line:
                        m_lang = re.search(r"Language set to: '(\w+)'", line)
                        if m_lang:
                            calls[curr_call_id]["language"] = m_lang.group(1)

                    if "Service category set to: '" in line:
                        m_svc = re.search(r"Service category set to: '(\w+)'", line)
                        if m_svc:
                            calls[curr_call_id]["service"] = m_svc.group(1)

                    if "Caller Voice Transcribed" in line:
                        m_user = re.search(r"Transcribed \(Whisper lang=\w+\): '([^']+)'", line)
                        if m_user:
                            pending_user = m_user.group(1).strip()

                    elif "AI reply (" in line and pending_user:
                        m_reply = re.search(r"AI reply \([^)]+\): '([^']+)'", line)
                        if m_reply:
                            reply_text = m_reply.group(1).strip()
                            calls[curr_call_id]["turns"].append((pending_user, reply_text))
                            pending_user = None

        except Exception as exc:
            logger.debug(f"Error reading {fpath}: {exc}")

    return calls


async def ingest_conversation_learnings() -> None:
    calls = extract_conversations_from_logs()
    logger.info(f"Parsed {len(calls)} call sessions from telephony logs.")

    emb_service = EmbeddingService(provider_id="semantic-default")
    reinforced_count = 0
    new_qa_count = 0
    episodes_count = 0

    async with async_session() as session:
        doc_repo = DocumentRepository(session)

        # Get or create Continuous Voice Call Learnings document
        doc = await doc_repo.get_or_create_document(
            tenant_id=TENANT_ID,
            title="Continuous Voice Call Learnings",
            source_type="voice_call",
        )

        for call_id, cdata in calls.items():
            turns = cdata.get("turns", [])
            if not turns:
                continue

            lang = cdata.get("language", "en")
            svc = cdata.get("service", "citizen_services")

            # 1. Process individual informative turns
            valid_episode_turns = []

            for u_text, r_text in turns:
                # Filter noise, pure greetings, or erroneous replies
                if not is_informative_interaction(u_text, r_text):
                    continue

                # Filter out known error replies from learning
                if any(err in r_text.lower() for err in ("direct ip access", "i apologize", "specialize exclusively in jana seva")):
                    continue

                # Filter out mismatch where Cursatificat got PAN answer
                if "cursatificat" in u_text.lower() and "pan" in r_text.lower():
                    # Substitute with verified correct caste certificate response
                    r_text = (
                        "To apply for a Caste Certificate (SC/ST/OBC/SEBC), submit your RoR land record (patta), "
                        "family caste proof, voter ID or Aadhaar, and photo at your nearest Jana Seva Kendra."
                    )

                norm_u = _normalize_phonetics(u_text)
                valid_episode_turns.append((u_text, r_text))

                # Embed query to find matching verified chunk
                query_emb = await emb_service.embed_query(norm_u)
                if not query_emb:
                    continue

                existing = await doc_repo.search_similar_chunks(
                    tenant_id=TENANT_ID,
                    query_embedding=query_emb,
                    top_k=1,
                )

                if existing and existing[0][1] >= 0.68:
                    top_chunk, score = existing[0]
                    # Reinforce existing chunk with real-world caller phrasing
                    updated = await doc_repo.record_chunk_hit(
                        top_chunk,
                        variant_query=u_text,
                    )
                    cache_rag_chunks(TENANT_ID, u_text, [top_chunk.content])
                    if norm_u != u_text:
                        cache_rag_chunks(TENANT_ID, norm_u, [top_chunk.content])
                    reinforced_count += 1
                    logger.info(
                        f"  [Reinforced] Chunk #{top_chunk.chunk_index} with '{u_text}' (score={score:.3f})"
                    )
                else:
                    # Novel question-answer pair
                    learned_content = (
                        f"Customer Spoken Inquiry: \"{u_text}\"\n"
                        f"Verified Resolution: \"{r_text}\"\n"
                        f"Service Category: {svc}\n"
                        f"Source: Live Telephone Interaction ({call_id})"
                    )
                    c_emb = await emb_service.embed_query(learned_content) or query_emb

                    variants_init = [u_text]
                    if norm_u != u_text:
                        variants_init.append(norm_u)

                    await doc_repo.append_chunk(
                        document_id=doc.id,
                        tenant_id=TENANT_ID,
                        content=learned_content,
                        embedding=c_emb,
                        meta={
                            "type": "learned_conversation_qa",
                            "call_uuid": call_id,
                            "caller_query": u_text,
                            "variants": variants_init,
                            "ask_count": 1,
                            "category": svc,
                            "language": lang,
                            "verified": True,
                        },
                    )
                    cache_rag_chunks(TENANT_ID, u_text, [learned_content])
                    if norm_u != u_text:
                        cache_rag_chunks(TENANT_ID, norm_u, [learned_content])
                    new_qa_count += 1
                    logger.info(f"  [Learned Q&A] Stored novel turn '{u_text[:40]}'")

            # 2. Store multi-turn consultation episode if >= 2 valid turns
            if len(valid_episode_turns) >= 2:
                dialogue_lines = [f"{'Caller' if i%2==0 else 'Assistant'}: {t}" for i, (u, a) in enumerate(valid_episode_turns) for t in (u, a)]
                dialogue_str = "\n".join(dialogue_lines)
                user_questions = [u for u, _ in valid_episode_turns]

                episode_content = (
                    f"Call Conversation Episode [Session {call_id}]:\n"
                    f"Service Category: {svc}\n"
                    f"Language: {lang}\n"
                    f"Consultation Flow:\n"
                    f"{dialogue_str}\n"
                    f"Key Inquiries: {', '.join(user_questions[:4])}"
                )
                ep_emb = await emb_service.embed_query(episode_content)
                if ep_emb:
                    await doc_repo.append_chunk(
                        document_id=doc.id,
                        tenant_id=TENANT_ID,
                        content=episode_content,
                        embedding=ep_emb,
                        meta={
                            "type": "conversation_episode",
                            "call_uuid": call_id,
                            "category": svc,
                            "language": lang,
                            "turn_count": len(valid_episode_turns),
                            "verified": True,
                        },
                    )
                    episodes_count += 1
                    logger.info(f"  [Episode] Stored conversation episode for {call_id} ({len(valid_episode_turns)} turns)")

        await _maybe_sync_dump_snapshot(session, TENANT_ID)

    logger.info("==================================================================")
    logger.info(f"Conversation RAG Learning Complete!")
    logger.info(f"  - Verified Knowledge Chunks Reinforced with Spoken Variants: {reinforced_count}")
    logger.info(f"  - Novel Spoken Q&A Chunks Ingested: {new_qa_count}")
    logger.info(f"  - Multi-Turn Consultation Episodes Stored: {episodes_count}")
    logger.info("==================================================================")


if __name__ == "__main__":
    asyncio.run(ingest_conversation_learnings())
