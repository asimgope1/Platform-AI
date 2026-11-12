"""
RAG-First Response Generator
============================
Orchestrates the 6-step RAG + Gemma + Grounded Validation voice pipeline:
  1. Hybrid retrieval over pgvector + FAQ cache
  2. Retrieval relevance & confidence evaluation
  3. Dynamic behavioral context construction
  4. Gemma multi-turn generation via ChatOrchestrator
  5. Groundedness & script validation via AnswerEvaluator
  6. Graceful technical fallback (ivr_mini_brain) + metrics recording
"""
from __future__ import annotations

import asyncio
import logging
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


def _needs_live_action_fallback(user_message: str, service_category: Optional[str]) -> bool:
    if service_category == "complaint":
        return True
    lower = (user_message or "").lower()
    return any(
        k in lower
        for k in (
            "complaint", "complain", "register a complaint", "file a complaint",
            "ticket", "order", "ord-", "tracking", "track", "payment status",
            "invoice", "शिकायत", "ଅଭିଯୋଗ",
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

        # 1 & 2. Retrieve relevant knowledge and evaluate retrieval confidence
        retrieved: RetrievedContext = await self.retrieval_service.search(
            query=user_message,
            tenant_id=tenant_id,
            top_k=4,
            preferred_category=service_category,
            session_factory=session_factory,
        )
        rag_context_str = "\n\n".join(retrieved.chunk_texts)
        if tool_output:
            rag_context_str = f"{rag_context_str}\n\nVerified Backend Tool Result:\n{tool_output}".strip()

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
            elif tools and (tools_require_remote or _needs_live_action_fallback(user_message, service_category)):
                source = "tool_unavailable_fallback"
                unavailable_msgs = {
                    "en": (
                        "I'm sorry, our live order and account lookup system is temporarily unavailable right now. "
                        "Please try again in a moment, or press 0 to speak with a support agent."
                    ),
                    "hi": (
                        "माफ़ कीजिए, हमारा लाइव ऑर्डर और खाता विवरण सिस्टम अभी अस्थायी रूप से उपलब्ध नहीं है। "
                        "कृपया थोड़ी देर बाद प्रयास करें, या सहायता एजेंट से बात करने के लिए 0 दबाएं।"
                    ),
                    "or": (
                        "କ୍ଷମା କରିବେ, ଆମର ଲାଇଭ୍ ଅର୍ଡର ଏବଂ ଆକାଉଣ୍ଟ ତଥ୍ୟ ସିଷ୍ଟମ୍ ଏବେ ଅସ୍ଥାୟୀ ଭାବରେ ଉପଲବ୍ଧ ନାହିଁ। "
                        "ଦୟାକରି କିଛି ସମୟ ପରେ ଚେଷ୍ଟା କରନ୍ତୁ, କିମ୍ବା ସହାୟତା ଏଜେଣ୍ଟଙ୍କ ସହିତ କଥା ହେବାକୁ 0 ଦବାନ୍ତୁ।"
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
