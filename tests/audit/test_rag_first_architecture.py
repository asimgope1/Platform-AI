"""
Tests for RAG-First Conversational Voice AI Architecture
========================================================
Covers:
  1. Structured banking policy dataset schema & completeness (data/banking_knowledge/*.json)
  2. HybridRetrievalService.hybrid_rerank filtering unverified candidates + boosting verified & frequent chunks
  3. evaluate_relevance tiers (STRONG / PARTIAL / NONE) & RetrievalEvaluator.snapshot telemetry
  4. AnswerEvaluator groundedness, rupee-amount anti-hallucination, required token, and script checks
  5. RAGFirstResponseGenerator 6-step flow (RAG+Gemma happy path, Gemma failure fallback, multi-turn history, tier=NONE clarify)
  6. KnowledgeLearningService Level 1 reinforcement vs Level 2 unanswered candidate drafting & approval promotion
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from platform_core.contracts.requests import ChatMessage
from services.conversation.response_generator import RAGFirstResponseGenerator
from services.evaluation.answer_evaluator import AnswerEvaluator
from services.evaluation.retrieval_evaluator import (
    RetrievalConfidenceTier,
    RetrievalEvaluation,
    RetrievalEvaluator,
    evaluate_relevance,
)
from services.knowledge.ingestion_service import KnowledgeIngestionService
from services.knowledge.knowledge_learning_service import KnowledgeLearningService
from services.knowledge.retrieval_service import HybridRetrievalService, RetrievedContext
from storage.models.document import DocumentChunk

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KNOWLEDGE_DIR = os.path.join(ROOT_DIR, "data", "banking_knowledge")


def test_structured_banking_knowledge_dataset_is_valid():
    records = KnowledgeIngestionService.load_dataset_directory(KNOWLEDGE_DIR)
    assert len(records) >= 7

    categories = {r["category"] for r in records}
    expected_categories = {
        "accounts",
        "upi_payments",
        "cards",
        "loans",
        "complaints",
        "security",
        "policies",
    }
    assert expected_categories.issubset(categories)

    for rec in records:
        assert rec.get("tenant_id")
        assert rec.get("topic")
        assert rec.get("title")
        assert rec.get("verified") is True
        assert rec.get("source_version")
        assert rec.get("effective_date")
        content = rec.get("content", "").strip()
        assert len(content) >= 60
        # Dataset should contain structured domain policy prose rather than raw Q/A lists
        assert not content.startswith("Q:")


def test_hybrid_rerank_excludes_unverified_candidates_and_boosts_verified_faq():
    unverified_draft = SimpleNamespace(
        id="c-draft",
        content="UPI payment pending reversal draft policy",
        meta_json=json.dumps({
            "type": "unanswered_candidate",
            "verified": False,
            "category": "upi_payments",
            "ask_count": 20,
        }),
    )
    low_ask_chunk = SimpleNamespace(
        id="c-low",
        content="UPI payment pending reversal usually settles in 24 to 48 hours",
        meta_json=json.dumps({
            "verified": True,
            "category": "upi_payments",
            "ask_count": 1,
        }),
    )
    high_ask_preferred_chunk = SimpleNamespace(
        id="c-high",
        content="UPI payment pending or debited without credit reverses automatically within 24 to 48 hours",
        meta_json=json.dumps({
            "verified": True,
            "category": "upi_payments",
            "ask_count": 25,
        }),
    )

    ranked = HybridRetrievalService.hybrid_rerank(
        query="my upi payment is pending and debited",
        raw_results=[
            (unverified_draft, 0.99),
            (low_ask_chunk, 0.72),
            (high_ask_preferred_chunk, 0.72),
        ],
        preferred_category="upi_payments",
        top_k=3,
    )

    ids = [chunk.id for chunk, _ in ranked]
    assert "c-draft" not in ids
    assert ids[0] == "c-high"
    assert ranked[0][1] > ranked[1][1]


def test_evaluate_relevance_tiers_and_telemetry_snapshot():
    verified_chunk = SimpleNamespace(
        content="Savings account minimum balance policy",
        meta_json=json.dumps({"verified": True, "category": "accounts"}),
    )
    unverified_chunk = SimpleNamespace(
        content="Draft answer about fee waiver",
        meta_json=json.dumps({"verified": False, "category": "policies"}),
    )

    strong = evaluate_relevance([(verified_chunk, 0.78)], strong_threshold=0.65, partial_threshold=0.42)
    assert strong.tier == RetrievalConfidenceTier.STRONG
    assert strong.verified_hits == 1
    assert strong.categories == ["accounts"]

    partial = evaluate_relevance([(unverified_chunk, 0.72)], strong_threshold=0.65, partial_threshold=0.42)
    assert partial.tier == RetrievalConfidenceTier.PARTIAL
    assert partial.verified_hits == 0

    none_eval = evaluate_relevance([(verified_chunk, 0.25)], strong_threshold=0.65, partial_threshold=0.42)
    assert none_eval.tier == RetrievalConfidenceTier.NONE

    empty_eval = evaluate_relevance([])
    assert empty_eval.tier == RetrievalConfidenceTier.NONE
    assert empty_eval.reason == "no_matching_chunks"

    telemetry = RetrievalEvaluator()
    telemetry.record_turn(strong, latency_ms=100.0, used_fallback=False, transferred=False)
    telemetry.record_turn(partial, latency_ms=200.0, used_fallback=False, transferred=False)
    telemetry.record_turn(none_eval, latency_ms=300.0, used_fallback=True, transferred=True)
    snap = telemetry.snapshot()
    assert snap["total_queries"] == 3
    assert snap["strong_hit_rate"] == pytest.approx(1 / 3, abs=1e-3)
    assert snap["partial_hit_rate"] == pytest.approx(1 / 3, abs=1e-3)
    assert snap["unanswered_query_rate"] == pytest.approx(1 / 3, abs=1e-3)
    assert snap["fallback_rate"] == pytest.approx(1 / 3, abs=1e-3)
    assert snap["transfer_rate"] == pytest.approx(1 / 3, abs=1e-3)
    assert snap["avg_response_latency_ms"] == 200.0


def test_answer_evaluator_grounding_amounts_tokens_and_scripts():
    evaluator = AnswerEvaluator()

    none_eval = RetrievalEvaluation(
        tier=RetrievalConfidenceTier.NONE,
        top_score=0.1,
        verified_hits=0,
        total_hits=0,
        reason="no_matching_chunks",
    )
    ungrounded_money = evaluator.validate_response(
        response="Your refund of ₹5,000 has been approved.",
        retrieved_chunks=[],
        retrieval_quality=none_eval,
        language="en",
        dynamic_tool_used=False,
    )
    assert not ungrounded_money.valid
    assert "ungrounded_financial_amount" in ungrounded_money.issues

    # Same financial figure is allowed when verified via structured backend tool
    tool_grounded = evaluator.validate_response(
        response="Your order 4821 refund of ₹5,000 is being processed.",
        retrieved_chunks=[],
        retrieval_quality=none_eval,
        language="en",
        required_token="4821",
        dynamic_tool_used=True,
    )
    assert tool_grounded.valid
    assert tool_grounded.grounded

    strong_eval = RetrievalEvaluation(
        tier=RetrievalConfidenceTier.STRONG,
        top_score=0.82,
        verified_hits=1,
        total_hits=1,
        reason="verified_strong_match",
    )
    hi_mismatch = evaluator.validate_response(
        response="Your debit card can be blocked immediately.",
        retrieved_chunks=["Instant debit card blocking is available 24x7."],
        retrieval_quality=strong_eval,
        language="hi",
    )
    assert not hi_mismatch.valid
    assert "language_mismatch_expected_hi" in hi_mismatch.issues

    missing_token = evaluator.validate_response(
        response="Your order is currently in transit.",
        retrieved_chunks=["Order tracking information."],
        retrieval_quality=strong_eval,
        language="en",
        required_token="4821",
    )
    assert not missing_token.valid
    assert "missing_required_token:4821" in missing_token.issues

    grounded_en = evaluator.validate_response(
        response="Instant debit card blocking is available right now to protect your account.",
        retrieved_chunks=["Instant debit card blocking is available 24x7 to protect your account."],
        retrieval_quality=strong_eval,
        language="en",
    )
    assert grounded_en.valid
    assert grounded_en.grounded


@pytest.mark.asyncio
async def test_rag_first_response_generator_paths():
    strong_chunk = SimpleNamespace(
        content="For video KYC, keep your original PAN card and Aadhaar details ready in a well-lit room.",
        meta_json=json.dumps({"verified": True, "category": "accounts"}),
    )
    strong_ctx = RetrievedContext(
        chunks=[strong_chunk],
        chunk_texts=[strong_chunk.content],
        scored_chunks=[(strong_chunk, 0.86)],
        evaluation=RetrievalEvaluation(
            tier=RetrievalConfidenceTier.STRONG,
            top_score=0.86,
            verified_hits=1,
            total_hits=1,
            reason="verified_strong_match",
            categories=["accounts"],
        ),
        cache_hit=False,
    )

    mock_retrieval = MagicMock()
    mock_retrieval.search = AsyncMock(return_value=strong_ctx)
    generator = RAGFirstResponseGenerator(retrieval_service=mock_retrieval)

    # (a) Strong RAG + Gemma happy path & multi-turn history forwarding
    mock_orch = MagicMock()
    mock_orch.execute = AsyncMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Please keep your original PAN card and Aadhaar details ready in good lighting for video KYC."
                    )
                )
            ]
        )
    )
    history = [
        ChatMessage(role="user", content="Hi, I want to open an account."),
        ChatMessage(role="assistant", content="Sure, I can guide you with account opening."),
        ChatMessage(role="user", content="What do I need for video KYC?"),
    ]
    res = await generator.generate_response(
        user_message="What do I need for video KYC?",
        conversation_history=history,
        tenant_id="tenant-1",
        language="en",
        orchestrator=mock_orch,
        service_category="accounts",
        service_name="Account Opening",
    )
    assert res.source == "gemma_rag"
    assert res.used_fallback is False
    assert res.grounded is True
    assert "PAN card" in res.text
    sent_req = mock_orch.execute.await_args.args[0]
    assert sent_req.messages[0].role == "system"
    assert "original PAN card" in sent_req.messages[0].content
    assert len(sent_req.messages) == 4  # system + 3 history turns

    # (b) Technical Gemma failure falls back cleanly to ivr_mini_brain
    failing_orch = MagicMock()
    failing_orch.execute = AsyncMock(side_effect=RuntimeError("Ollama unreachable"))
    fallback_res = await generator.generate_response(
        user_message="What do I need for video KYC?",
        conversation_history=history,
        tenant_id="tenant-1",
        language="en",
        orchestrator=failing_orch,
    )
    assert fallback_res.source == "fallback_mini_brain"
    assert fallback_res.used_fallback is True
    assert len(fallback_res.text.strip()) > 0

    # (c) tier=NONE valid clarification response from Gemma does NOT falsely trigger fallback
    none_ctx = RetrievedContext(
        chunks=[],
        chunk_texts=[],
        scored_chunks=[],
        evaluation=RetrievalEvaluation(
            tier=RetrievalConfidenceTier.NONE,
            top_score=0.0,
            verified_hits=0,
            total_hits=0,
            reason="no_matching_chunks",
        ),
    )
    mock_retrieval.search = AsyncMock(return_value=none_ctx)
    clarify_orch = MagicMock()
    clarify_orch.execute = AsyncMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Could you clarify which forex card fee you are asking about, or should I connect an agent?"
                    )
                )
            ]
        )
    )
    none_res = await generator.generate_response(
        user_message="What about special forex waiver?",
        conversation_history=[ChatMessage(role="user", content="What about special forex waiver?")],
        tenant_id="tenant-1",
        language="en",
        orchestrator=clarify_orch,
    )
    assert none_res.source == "gemma_rag"
    assert none_res.used_fallback is False
    assert none_res.retrieval_eval.tier == RetrievalConfidenceTier.NONE


@pytest.mark.asyncio
async def test_knowledge_learning_levels_and_approval_promotion():
    learner = KnowledgeLearningService()

    # Level 1 reinforcement delegates to learn_from_voice_turn
    with patch(
        "services.knowledge.knowledge_learning_service.learn_from_voice_turn",
        new=AsyncMock(return_value=True),
    ) as mock_l1:
        ok = await learner.process_turn_feedback(
            tenant_id="t1",
            call_uuid="call-1",
            user_text="How do I check UPI limit?",
            reply_text="Standard UPI limit is ₹1,00,000 per day.",
            retrieval_tier="strong",
            quality_ok=True,
            resolved=True,
        )
        assert ok is True
        mock_l1.assert_awaited_once()

    # Level 2 unanswered candidate drafting stores verified=False, status='pending_review'
    fake_session = AsyncMock()
    fake_repo = MagicMock()
    fake_repo.search_similar_chunks = AsyncMock(return_value=[])
    fake_repo.get_or_create_document = AsyncMock(return_value=SimpleNamespace(id="doc-cand-1"))
    fake_repo.append_chunk = AsyncMock(return_value=SimpleNamespace(id="chunk-cand-1"))

    @asynccontextmanager
    async def _ctx():
        yield fake_session

    mock_emb = MagicMock()
    mock_emb.embed_query = AsyncMock(return_value=[0.1] * 384)

    with patch("services.knowledge.knowledge_learning_service.async_session", new=_ctx), \
         patch("services.knowledge.knowledge_learning_service.DocumentRepository", return_value=fake_repo), \
         patch("services.knowledge.knowledge_learning_service.EmbeddingService", return_value=mock_emb), \
         patch("services.knowledge.knowledge_learning_service._maybe_sync_dump_snapshot", new=AsyncMock()):
        logged = await learner.process_turn_feedback(
            tenant_id="t1",
            call_uuid="call-2",
            user_text="What is the zero markup forex lounge quota?",
            reply_text="Let me check with a specialist for that card variant.",
            retrieval_tier="none",
            quality_ok=True,
        )

    assert logged is True
    append_kwargs = fake_repo.append_chunk.await_args.kwargs
    assert append_kwargs["meta"]["type"] == "unanswered_candidate"
    assert append_kwargs["meta"]["verified"] is False
    assert append_kwargs["meta"]["status"] == "pending_review"

    # Approval promotes the candidate chunk into verified_policy with new embedding
    candidate_chunk = DocumentChunk(
        id="chunk-cand-1",
        document_id="doc-cand-1",
        tenant_id="t1",
        chunk_index=0,
        content="[CANDIDATE - PENDING REVIEW]",
        meta_json=json.dumps(append_kwargs["meta"]),
    )
    fake_session.get = AsyncMock(return_value=candidate_chunk)
    fake_session.commit = AsyncMock()
    fake_session.refresh = AsyncMock()

    with patch("services.knowledge.knowledge_learning_service.async_session", new=_ctx), \
         patch("services.knowledge.knowledge_learning_service.EmbeddingService", return_value=mock_emb), \
         patch("services.knowledge.knowledge_learning_service._maybe_sync_dump_snapshot", new=AsyncMock()):
        promoted = await learner.approve_candidate_chunk(
            chunk_id="chunk-cand-1",
            verified_content="Zero markup forex cards include 2 complimentary international lounge visits per quarter.",
            category="cards",
            topic="forex_lounge_access",
        )

    assert promoted is not None
    assert promoted["verified"] is True
    assert promoted["meta"]["type"] == "verified_policy"
    assert promoted["meta"]["status"] == "approved"
    assert candidate_chunk.content.startswith("Zero markup forex cards")

