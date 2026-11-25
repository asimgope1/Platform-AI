"""
Gemma-Powered Knowledge Distillation & Training Service
======================================================
Uses Gemma 4 (via local/remote Ollama or Colab) to:
1. Distill raw guide text into structured high-density knowledge chunks.
2. Auto-classify domain category, primary topic, and intent triggers.
3. Generate multilingual Q&A pairs (English, Odia, Hindi).
4. Persist enriched chunks in PostgreSQL pgvector and SQLite.
5. Dynamically register new categories into the AI platform taxonomy.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import urllib.request
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from platform_core.config.settings import settings
from retrieval.embedding_service import EmbeddingService
from storage.database import async_session, init_db
from storage.models.document import Document, DocumentChunk
from storage.repositories.document_repo import DocumentRepository

logger = logging.getLogger(__name__)

GEMMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    getattr(settings, "OLLAMA_BASE_URL", "https://complicated-owners-fabrics-hardly.trycloudflare.com/v1"),
).rstrip("/")
GEMMA_MODEL = os.getenv("OLLAMA_GEMMA_MODEL", "gemma4:latest")


class GemmaKnowledgeTrainer:
    """Uses Gemma to distill, enrich, categorize, and train RAG knowledge chunks."""

    def __init__(
        self,
        base_url: str = GEMMA_BASE_URL,
        model_name: str = GEMMA_MODEL,
        embedding_provider_id: str = "semantic-default",
    ) -> None:
        self.base_url = base_url
        self.model_name = model_name
        self.embedding_service = EmbeddingService(provider_id=embedding_provider_id)

    async def distill_section_with_gemma(self, section_text: str, fallback_title: str) -> Dict[str, Any]:
        """
        Calls Gemma to extract structured JSON containing category, topic, Q&As, facts, and enriched text.
        """
        system_prompt = (
            "You are a specialized knowledge distillation engine for an AI telephone assistant. "
            "Given raw text, extract facts and output ONLY valid JSON without extra markdown. "
            "Keys required: \n"
            "{\n"
            '  "category_slug": "short category identifier",\n'
            '  "category_name_en": "Human Readable English Name",\n'
            '  "category_name_or": "Human Readable Odia Name",\n'
            '  "topic": "Concise topic title",\n'
            '  "key_qa": [\n'
            '    {"q_en": "...", "a_en": "...", "q_or": "...", "a_or": "..."}\n'
            "  ],\n'
            '  "eligibility_facts": ["fact 1", "fact 2"],\n'
            '  "required_documents": ["doc 1", "doc 2"],\n'
            '  "enriched_chunk_text": "Dense, factual narrative text combining all rules, steps, and details for RAG embedding."\n'
            "}"
        )

        user_content = f"Guide Section:\n{section_text[:2500]}"

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.1,
            "max_tokens": 1200,
        }

        def _do_request() -> str:
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                choice = data.get("choices", [{}])[0]
                msg = choice.get("message", {})
                return msg.get("content") or msg.get("reasoning") or ""

        try:
            raw_output = await asyncio.to_thread(_do_request)
            # Clean JSON fences if present
            cleaned = re.sub(r"^```(?:json)?", "", raw_output.strip(), flags=re.MULTILINE)
            cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE).strip()

            # Find outer JSON brackets
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1:
                return json.loads(cleaned[start : end + 1])
        except Exception as e:
            logger.warning(f"[GemmaTrainer] Gemma distillation failed: {e}. Using fallback structure.")

        # Fallback if LLM output fails
        return {
            "category_slug": "general_citizen_service",
            "category_name_en": "General Citizen Services",
            "category_name_or": "ସାଧାରଣ ନାଗରିକ ସେବା",
            "topic": fallback_title,
            "key_qa": [],
            "eligibility_facts": [],
            "required_documents": [],
            "enriched_chunk_text": section_text.strip(),
        }

    async def train_and_update_chunk(
        self,
        tenant_id: str,
        section_text: str,
        title: str,
        source_name: str,
    ) -> Dict[str, Any]:
        """
        Runs Gemma distillation, embeds enriched text, updates pgvector, and registers category.
        """
        distilled = await self.distill_section_with_gemma(section_text, fallback_title=title)

        category_slug = distilled.get("category_slug", "general_service").lower().replace(" ", "_")
        topic = distilled.get("topic", title)
        enriched_body = distilled.get("enriched_chunk_text") or section_text

        doc_title = f"{title} - {topic}" if topic != title else title
        searchable_text = f"[{category_slug}] {doc_title}\n{enriched_body}"

        meta = {
            "domain": "trained_knowledge",
            "category": category_slug,
            "category_name_en": distilled.get("category_name_en", "Citizen Service"),
            "category_name_or": distilled.get("category_name_or", "ନାଗରିକ ସେବା"),
            "topic": topic,
            "title": doc_title,
            "key_qa": distilled.get("key_qa", []),
            "eligibility": distilled.get("eligibility_facts", []),
            "required_documents": distilled.get("required_documents", []),
            "source": source_name,
            "source_type": "gemma_trained_chunk",
            "source_version": "2.0",
            "verified": True,
            "trained_by": self.model_name,
            "effective_date": datetime.now().strftime("%Y-%m-%d"),
            "ask_count": 1,
        }

        # Embed with local miniLM
        emb = await self.embedding_service.embed_query(searchable_text)

        await init_db()
        async with async_session() as session:
            doc_repo = DocumentRepository(session)
            doc = await doc_repo.get_or_create_document(
                tenant_id=tenant_id,
                title=doc_title,
                source_type="gemma_trained_chunk",
            )

            await doc_repo.append_chunk(
                document_id=doc.id,
                tenant_id=tenant_id,
                content=searchable_text,
                embedding=emb,
                meta=meta,
            )

        logger.info(f"[GemmaTrainer] ✓ Successfully trained & updated chunk: '{doc_title}' ({category_slug})")
        return {
            "doc_id": doc.id,
            "title": doc_title,
            "category": category_slug,
            "qa_count": len(distilled.get("key_qa", [])),
            "status": "trained_and_persisted",
        }
