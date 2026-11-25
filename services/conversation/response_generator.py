"""
RAG-First Response Generator
============================
Orchestrates the 7-step RAG + Collaborative Gemma + Grounded Validation voice pipeline:
  1. Hybrid retrieval over pgvector + FAQ cache
  2. Retrieval relevance & confidence evaluation
  3. Instant mini-brain RAG synthesis (zero latency)
  4. Collaborative Gemma enrichment  (Colab GPU, fills gaps when mini-brain is weak)
  5. Dynamic behavioral context construction
  6. Gemma multi-turn generation via ChatOrchestrator
  7. Groundedness & script validation via AnswerEvaluator
  8. Graceful technical fallback (ivr_mini_brain) + metrics recording
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from orchestration.chat_orchestrator import ChatOrchestrator
from platform_core.contracts.requests import ChatMessage, NormalizedChatRequest
from platform_core.security.tenant_context import TenantContext
from providers.chat.ivr_mini_model import ivr_mini_brain
from services.conversation.context_builder import ConversationContextBuilder
from services.evaluation.answer_evaluator import AnswerEvaluator
from services.evaluation.retrieval_evaluator import (
    RetrievalConfidenceTier,
    RetrievalEvaluation,
    RetrievalEvaluator,
)
from services.knowledge.retrieval_service import HybridRetrievalService, RetrievedContext

logger = logging.getLogger(__name__)


@dataclass
class GeneratedResponse:
    text: str
    source: str  # "tool" | "gemma_rag" | "fallback_mini_brain"
    retrieval_eval: RetrievalEvaluation
    grounded: bool
    used_fallback: bool
    latency_ms: float
    rag_context_text: str = ""
    cache_hit: bool = False
    resolved_customer_id: Optional[str] = None


def _extract_tool_names(tools: Optional[List[Dict[str, Any]]]) -> List[str]:
    names: List[str] = []
    for item in tools or []:
        fn = item.get("function") if isinstance(item, dict) else None
        if isinstance(fn, dict) and fn.get("name"):
            names.append(str(fn["name"]))
    return names


async def _collaborative_enrich(
    mini_reply: str,
    user_query: str,
    rag_context: str,
    language: str,
    orchestrator: ChatOrchestrator,
    timeout_seconds: float = 1.5,
) -> str:
    """
    Collaborative Gemma enrichment pass.
    Sends the mini-brain draft + RAG context to Colab Gemma GPU to fix gaps,
    expand brevity, correct language tone, and ensure completeness.
    Falls back to the original mini_reply on any error.
    """
    lang_instruction = {
        "hi": "Reply in Hindi (Devanagari). Telephone-style: short, clear, friendly.",
        "or": "Reply in Odia (ଓଡ଼ିଆ). Telephone-style: short, clear, friendly.",
        "en": "Reply in English. Telephone-style: short, clear, friendly.",
    }.get(language, "Reply in English. Telephone-style: short, clear, friendly.")

    system_msg = (
        "You are a collaborative citizen services AI helping a mini-RAG model improve its answers. "
        "Given the user question, the RAG knowledge context, and the mini-model's draft answer, "
        "produce one improved, complete, accurate answer. "
        f"{lang_instruction} "
        "Max 40 words. No prefixes like 'Answer:'. Do not make up information not in the RAG context."
    )
    user_content = (
        f"User question: {user_query}\n\n"
        f"RAG knowledge:\n{rag_context[:500] if rag_context else 'No context available.'}\n\n"
        f"Mini-model draft: {mini_reply}\n\n"
        "Improved answer:"
    )

    from platform_core.contracts.requests import ChatMessage as CM, NormalizedChatRequest as NCR
    collab_req = NCR(
        model_alias="gemma-gpu",   # Colab GPU path
        messages=[
            CM(role="system", content=system_msg),
            CM(role="user",   content=user_content),
        ],
        stream=False,
        max_tokens=60,
        temperature=0.2,
    )
    try:
        res = await asyncio.wait_for(
            orchestrator.execute(collab_req, context=None, intelligence_context=None),
            timeout=timeout_seconds,
        )
        enriched = (res.choices[0].message.content or "").strip()
        if enriched and len(enriched) > 10:
            logger.info(f"[IVR] 🤝 Collaborative Gemma enrichment applied ({len(enriched)} chars).")
            return enriched
    except Exception as collab_err:
        logger.debug(f"[IVR] Collaborative enrichment skipped ({collab_err}); using mini-brain reply.")
    return mini_reply


def _needs_live_action_fallback(user_message: str, service_category: Optional[str]) -> bool:
    lower = (user_message or "").lower()
    # Digits, complaint ticket numbers, CMP codes, phone lookups can be synthesized instantly without slow LLM tool loops
    from providers.chat.ivr_mini_model import extract_complaint_identifier, extract_order_last4_digits, ivr_mini_brain
    if extract_complaint_identifier(user_message, allow_bare_digits=True) or extract_order_last4_digits(user_message, allow_bare_digits=True):
        return False
    # Status checks for complaints, orders, or tickets can be synthesized instantly via RAG
    if any(k in lower for k in ("status", "check", "what is", "where is", "track", "tracking", "cmp", "ticket", "case", "phone", "mobile", "number", "स्थिति", "zero", "one", "two", "three")):
        return False
    # Complaint registration flows (intent or details) are handled instantly by ivr_mini_brain without slow LLM tool stalls
    if ivr_mini_brain._is_complaint_register_intent(lower) or any(k in lower for k in ("register a complaint", "file a complaint", "book a complaint", "दर्ज करें", "दर्ज करनी", "नया शिकायत", "new complaint", "register complaint", "register new complain", "शिकायत दर्ज", "ଅଭିଯୋଗ")):
        return False
    return any(
        k in lower
        for k in (
            "book", "invoice",
        )
    )


class RAGFirstResponseGenerator:
    """Centralized RAG-first + Gemma response generator with groundedness validation."""

    def __init__(
        self,
        retrieval_service: Optional[HybridRetrievalService] = None,
        context_builder: Optional[ConversationContextBuilder] = None,
        answer_evaluator: Optional[AnswerEvaluator] = None,
        retrieval_evaluator: Optional[RetrievalEvaluator] = None,
    ) -> None:
        self.retrieval_service = retrieval_service or HybridRetrievalService()
        self.context_builder = context_builder or ConversationContextBuilder()
        self.answer_evaluator = answer_evaluator or AnswerEvaluator()
        self.retrieval_evaluator = retrieval_evaluator or RetrievalEvaluator()

    async def generate_response(
        self,
        user_message: str,
        conversation_history: List[ChatMessage],
        tenant_id: str,
        language: str,
        orchestrator: ChatOrchestrator,
        model_alias: str = "gemma",
        service_category: Optional[str] = None,
        service_name: str = "Customer Support",
        intelligence_context: Optional[Dict[str, Any]] = None,
        tool_output: str = "",
        required_token: Optional[str] = None,
        active_emotion: str = "neutral",
        previous_attempts: int = 0,
        timeout_seconds: float = 3.5,
        session_factory: Optional[Any] = None,
        context: Optional[TenantContext] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        authorized_customer_id: Optional[str] = None,
        call_uuid: Optional[str] = None,
        caller_number: Optional[str] = None,
        tools_require_remote: bool = False,
    ) -> GeneratedResponse:
        t_start = time.perf_counter()

        # Context-expand follow-up questions before RAG retrieval using dialogue history
        search_query = user_message
        if conversation_history and len(conversation_history) >= 2:
            prev_turns = " ".join(
                str(getattr(m, "content", "") or "") for m in conversation_history[-4:]
            ).lower()
            lower_msg = (user_message or "").lower()
            if any(w in lower_msg for w in ["apply", "kaise apply", "kahan", "process", "procedure", "how to", "eligibility", "document", "documents", "money", "kitna"]):
                if ("caste" in prev_turns or "jati" in prev_turns) and "caste" not in lower_msg:
                    search_query = f"{user_message} caste certificate"
                elif ("subhadra" in prev_turns or "scheme" in prev_turns or "yojana" in prev_turns) and "subhadra" not in lower_msg:
                    search_query = f"{user_message} subhadra scheme"
                elif ("income" in prev_turns or "aay" in prev_turns) and "income" not in lower_msg:
                    search_query = f"{user_message} income certificate"
                elif ("residence" in prev_turns or "niwas" in prev_turns) and "residence" not in lower_msg:
                    search_query = f"{user_message} residence certificate"
                elif ("pan" in prev_turns) and "pan" not in lower_msg:
                    search_query = f"{user_message} pan card"
                elif ("voter" in prev_turns) and "voter" not in lower_msg:
                    search_query = f"{user_message} voter id card"

        # 1 & 2. Retrieve relevant knowledge and evaluate retrieval confidence
        retrieved: RetrievedContext = await self.retrieval_service.search(
            query=search_query,
            tenant_id=tenant_id,
            top_k=4,
            preferred_category=service_category,
            session_factory=session_factory,
        )
        rag_context_str = "\n\n".join(retrieved.chunk_texts)
        if tool_output:
            rag_context_str = f"{rag_context_str}\n\nVerified Backend Tool Result:\n{tool_output}".strip()

        # If knowledge exists in RAG or can be answered autonomously by the mini-model (complaint IDs, citizen services FAQs),
        # synthesize instantly with zero latency, without stalling on slow CPU LLMs.
        rag_first_enabled = os.getenv("IVR_RAG_FIRST_INSTANT", "true").lower() in ("1", "true", "yes")
        action_intent = _needs_live_action_fallback(user_message, service_category)
        if rag_first_enabled and not action_intent and not tool_output:
            rag_reply = ivr_mini_brain.synthesize(
                user_query=user_message,
                messages=conversation_history,
                rag_context=rag_context_str,
                language=language,
                service_category=service_category,
                intelligence_context=intelligence_context,
                caller_phone=caller_number,
                call_uuid=call_uuid,
            )
            from providers.chat.ivr_mini_model import AUTONOMOUS_KNOWLEDGE
            generic_fallback = AUTONOMOUS_KNOWLEDGE["fallback"].get(language, AUTONOMOUS_KNOWLEDGE["fallback"]["en"])
            # Synthesize instantly if mini-brain answered with knowledge/catalog/chunk or if no tools to call
            if rag_reply and (rag_reply != generic_fallback or retrieved.chunk_texts or not tools):
                # ── Collaborative Gemma Confirmation & Enrichment ────────────────
                # Always consult Colab Gemma GPU when RAG chunks are retrieved
                # (unless IVR_COLLAB_ENRICH_ENABLED=false) so Gemma confirms that
                # the mini-brain draft accurately answers the caller from RAG, or
                # refines it if needed. Falls back to mini-brain on timeout/error.
                _COLLAB_THRESHOLD = int(os.getenv("IVR_COLLAB_ENRICH_THRESHOLD", "35"))
                _COLLAB_ENABLED   = os.getenv("IVR_COLLAB_ENRICH_ENABLED", "true").lower() in ("1", "true", "yes")
                _COLLAB_ALWAYS    = os.getenv("IVR_COLLAB_CONFIRM_RAG", "true").lower() in ("1", "true", "yes")
                reply_is_weak = len(rag_reply.strip()) < _COLLAB_THRESHOLD
                rich_rag_available = bool(len(retrieved.chunk_texts) >= 1 and rag_context_str.strip())
                is_strong_instant = (
                    (retrieved.cache_hit or retrieved.evaluation.tier == RetrievalConfidenceTier.STRONG)
                    and len(rag_reply.strip()) >= 40
                    and rag_reply != generic_fallback
                )
                if _COLLAB_ENABLED and rich_rag_available and not is_strong_instant and (_COLLAB_ALWAYS or reply_is_weak or rag_reply == generic_fallback):
                    logger.info(
                        f"[IVR] 🤝 Consulting Gemma to confirm/refine RAG reply "
                        f"(reply_len={len(rag_reply)}, weak={reply_is_weak}, chunks={len(retrieved.chunk_texts)})."
                    )
                    rag_reply = await _collaborative_enrich(
                        mini_reply=rag_reply,
                        user_query=user_message,
                        rag_context=rag_context_str,
                        language=language,
                        orchestrator=orchestrator,
                        timeout_seconds=2.2,
                    )
                # ─────────────────────────────────────────────────────────────────
                logger.info(
                    f"[IVR] ⚡ Instant RAG Synthesis ({len(retrieved.chunk_texts)} chunks, tier={retrieved.evaluation.tier.value}) "
                    f"for query: '{user_message[:60]}'"
                )
                formatted_reply = ivr_mini_brain.apply_sentiment_tone(
                    rag_reply,
                    active_emotion,
                    lang=language,
                    previous_attempts=previous_attempts,
                )
                latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
                return GeneratedResponse(
                    text=formatted_reply,
                    source="instant_rag_collab" if _COLLAB_ENABLED else "instant_rag",
                    retrieval_eval=retrieved.evaluation,
                    grounded=True,
                    used_fallback=False,
                    latency_ms=latency_ms,
                    rag_context_text=rag_context_str,
                    cache_hit=retrieved.cache_hit,
                    resolved_customer_id=None,
                )

        if retrieved.cache_hit:
            logger.info(
                f"[IVR] ⚡ Instant RAG hit ({len(retrieved.chunk_texts)} chunks, tier={retrieved.evaluation.tier.value})."
            )
        elif retrieved.chunk_texts:
            logger.info(
                f"[IVR] ✓ Hybrid RAG retrieved {len(retrieved.chunk_texts)} chunks "
                f"(tier={retrieved.evaluation.tier.value}, score={retrieved.evaluation.top_score:.2f})."
            )

        tool_names = _extract_tool_names(tools)
        # 3. Build dynamic behavioral Gemma prompt (last 8 turns + retrieved policy docs + call tools)
        messages = self.context_builder.build_messages(
            history=conversation_history,
            retrieved=retrieved,
            language=language,
            service_name=service_name,
            tool_output=tool_output,
            service_category=service_category,
            available_tools=tool_names,
        )

        req_metadata: Dict[str, Any] = {"source_channel": "voice_ivr"}
        if authorized_customer_id:
            req_metadata["authorized_customer_id"] = authorized_customer_id
        if call_uuid and call_uuid != "unknown":
            req_metadata["call_uuid"] = call_uuid
        if caller_number and caller_number != "unknown":
            req_metadata["caller_number"] = caller_number
        if service_category:
            req_metadata["service_category"] = service_category

        gemma_req = NormalizedChatRequest(
            model_alias=model_alias,
            messages=messages,
            stream=False,
            max_tokens=160 if (tool_output or tools) else 75,
            temperature=0.35,
            tools=tools,
            metadata=req_metadata,
        )

        # 4 & 5. Generate with Gemma and validate groundedness + script compliance
        reply_text = ""
        grounded = True
        used_fallback = False
        source = "gemma_rag"

        try:
            logger.info(
                f"[IVR] 🗣️ Conversational turn with Gemma (model={model_alias}, "
                f"rag_tier={retrieved.evaluation.tier.value}, tools={len(tools or [])}, history={len(messages) - 1}): "
                f"'{user_message[:60]}'..."
            )
            gemma_res = await asyncio.wait_for(
                orchestrator.execute(
                    gemma_req,
                    context=context,
                    intelligence_context=intelligence_context,
                ),
                timeout=timeout_seconds,
            )
            candidate = (gemma_res.choices[0].message.content or "").strip()
            validation = self.answer_evaluator.validate_response(
                response=candidate,
                retrieved_chunks=retrieved.chunks,
                retrieval_quality=retrieved.evaluation,
                language=language,
                required_token=required_token,
                dynamic_tool_used=bool(tool_output or tools),
            )
            if validation.valid:
                reply_text = ivr_mini_brain.apply_sentiment_tone(
                    candidate,
                    active_emotion,
                    lang=language,
                    previous_attempts=previous_attempts,
                )
                grounded = validation.grounded
                logger.info(
                    f"[IVR] 🌟 Gemma conversational reply (grounded={grounded}, "
                    f"score={validation.groundedness_score}): '{reply_text}'"
                )
            else:
                logger.warning(
                    f"[IVR] Validation rejected Gemma candidate (issues={validation.issues}): '{candidate[:80]}'"
                )
        except Exception as gemma_err:
            logger.warning(f"[IVR] Gemma conversational turn error ({gemma_err}); falling back to mini-brain.")

        # 6. Technical fallback ONLY when Gemma fails, times out, or fails safety validation
        if not reply_text:
            used_fallback = True
            if tool_output:
                source = "tool"
                reply_text = ivr_mini_brain.apply_sentiment_tone(
                    tool_output,
                    active_emotion,
                    lang=language,
                    previous_attempts=previous_attempts,
                )
            elif tools and (
                _needs_live_action_fallback(user_message, service_category)
                or (tools_require_remote and retrieved.evaluation.tier == RetrievalConfidenceTier.NONE)
            ):
                source = "tool_unavailable_fallback"
                unavailable_msgs = {
                    "en": (
                        "I'm sorry, our citizen services verification system is temporarily unavailable right now. "
                        "Please try again in a moment, or press 9 to speak with a representative."
                    ),
                    "hi": (
                        "माफ़ कीजिए, हमारा नागरिक सेवा सत्यापन सिस्टम अभी अस्थायी रूप से अनुपलब्ध है। "
                        "कृपया थोड़ी देर बाद प्रयास करें, या प्रतिनिधि से बात करने के लिए 9 दबाएं।"
                    ),
                    "or": (
                        "କ୍ଷମା କରିବେ, ଆମର ନାଗରିକ ସେବା ସତ୍ୟାପନ ସିଷ୍ଟମ୍ ଏବେ ଅସ୍ଥାୟୀ ଭାବରେ ଉପଲବ୍ଧ ନାହିଁ। "
                        "ଦୟାକରି କିଛି ସମୟ ପରେ ଚେଷ୍ଟା କରନ୍ତୁ, କିମ୍ବା ପ୍ରତିନିଧିଙ୍କ ସହ କଥା ହେବାକୁ 9 ଦବାନ୍ତୁ।"
                    ),
                }
                reply_text = ivr_mini_brain.apply_sentiment_tone(
                    unavailable_msgs.get(language, unavailable_msgs["en"]),
                    active_emotion,
                    lang=language,
                    previous_attempts=previous_attempts,
                )
            else:
                source = "fallback_mini_brain"
                reply_text = ivr_mini_brain.synthesize(
                    user_query=user_message,
                    messages=conversation_history,
                    rag_context=rag_context_str,
                    language=language,
                    service_category=service_category,
                    intelligence_context=intelligence_context,
                    caller_phone=caller_number,
                    call_uuid=call_uuid,
                )
            grounded = bool(tool_output or (not tools and retrieved.evaluation.tier != RetrievalConfidenceTier.NONE))
            logger.info(f"[IVR] 🧠 Technical fallback reply (source={source}): '{reply_text}'")

        latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        self.retrieval_evaluator.record_turn(
            evaluation=retrieved.evaluation,
            latency_ms=latency_ms,
            used_fallback=used_fallback,
        )

        return GeneratedResponse(
            text=reply_text,
            source=source,
            retrieval_eval=retrieved.evaluation,
            grounded=grounded,
            used_fallback=used_fallback,
            latency_ms=latency_ms,
            rag_context_text=rag_context_str,
            cache_hit=retrieved.cache_hit,
            resolved_customer_id=req_metadata.get("resolved_customer_id"),
        )
