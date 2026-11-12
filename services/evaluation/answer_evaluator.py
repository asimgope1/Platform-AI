"""
Answer Groundedness & Policy Validator
======================================
Validates Gemma outputs against retrieved RAG chunks, script/language constraints,
and anti-hallucination guardrails before speaking back to callers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, List, Optional

from services.evaluation.retrieval_evaluator import RetrievalConfidenceTier, RetrievalEvaluation


@dataclass
class AnswerValidationResult:
    valid: bool
    grounded: bool
    groundedness_score: float
    issues: List[str] = field(default_factory=list)


_PLACEHOLDER_PATTERNS = (
    "mock response",
    "didn't quite catch that",
    "पूरी तरह समझ नहीं",
    "ସମ୍ପୂର୍ଣ୍ଣ ବୁଝିପାରିଲି ନାହିଁ",
)

# Matches fabricated balances/amounts unless grounded in RAG context or structured tools
_SPECIFIC_MONEY_OR_ID = re.compile(r"(?:₹|rs\.?|inr)\s*\d[\d,]*(?:\.\d+)?", re.IGNORECASE)


class AnswerEvaluator:
    """Checks generated LLM answers for grounding, script fidelity, and safety."""

    @staticmethod
    def _chunk_text(chunk: Any) -> str:
        if isinstance(chunk, str):
            return chunk
        if isinstance(chunk, dict):
            return str(chunk.get("content") or "")
        return str(getattr(chunk, "content", "") or "")

    def validate_response(
        self,
        response: str,
        retrieved_chunks: List[Any],
        retrieval_quality: RetrievalEvaluation,
        language: str = "en",
        required_token: Optional[str] = None,
        dynamic_tool_used: bool = False,
    ) -> AnswerValidationResult:
        clean = (response or "").strip()
        issues: List[str] = []

        if not clean or any(p in clean.lower() for p in _PLACEHOLDER_PATTERNS):
            issues.append("empty_or_placeholder_response")

        # Strict script checks matching session language
        if language == "hi" and not any("\u0900" <= ch <= "\u097f" for ch in clean):
            issues.append("language_mismatch_expected_hi")
        elif language == "or" and any("\u0900" <= ch <= "\u097f" for ch in clean):
            issues.append("language_mismatch_expected_or_not_hi")

        # Ensure structured dynamic tool outputs (e.g. Order ID last 4 digits) are preserved
        if required_token and required_token not in clean:
            issues.append(f"missing_required_token:{required_token}")

        combined_context = " ".join(self._chunk_text(c) for c in retrieved_chunks).lower()

        # Prevent fabricated monetary balances/refunds not present in retrieved context or tool results
        if not dynamic_tool_used:
            for m in _SPECIFIC_MONEY_OR_ID.finditer(clean):
                digits = re.sub(r"\D", "", m.group(0))
                if digits and digits not in re.sub(r"\D", "", combined_context):
                    if retrieval_quality.tier == RetrievalConfidenceTier.NONE:
                        issues.append("ungrounded_financial_amount")
                        break

        # Compute lexical groundedness against retrieved context
        resp_tokens = {w for w in re.findall(r"[a-zA-Z0-9\u0900-\u097f\u0b00-\u0b7f]{3,}", clean.lower())}
        ctx_tokens = {w for w in re.findall(r"[a-zA-Z0-9\u0900-\u097f\u0b00-\u0b7f]{3,}", combined_context)}
        if resp_tokens and ctx_tokens:
            overlap = len(resp_tokens.intersection(ctx_tokens)) / max(1, min(len(resp_tokens), 12))
            groundedness = round(min(1.0, overlap), 3)
        elif retrieval_quality.tier == RetrievalConfidenceTier.NONE:
            groundedness = 0.0
        else:
            groundedness = 0.5

        grounded = (
            "ungrounded_financial_amount" not in issues
            and (retrieval_quality.tier != RetrievalConfidenceTier.NONE or dynamic_tool_used or groundedness >= 0.15)
        )

        return AnswerValidationResult(
            valid=len(issues) == 0,
            grounded=grounded,
            groundedness_score=groundedness,
            issues=issues,
        )

