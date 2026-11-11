"""
Satisfaction Analyzer — keeps explicit customer feedback ratings distinct from
inferred conversational satisfaction.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class SatisfactionResult:
    satisfaction_score: Optional[float]  # [0.0, 1.0] or None when unknown
    satisfaction_label: str              # "low" | "moderate" | "high" | "unknown"
    explicit_rating: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SatisfactionAnalyzer:
    """Evaluates caller satisfaction from explicit ratings, emotion, and failed attempts."""

    def analyze(
        self,
        text_emotion: str,
        resolution_status: str,
        failed_attempts: int = 0,
        explicit_rating: Optional[int] = None,
        customer_confirmed: bool = False,
        prior_score: Optional[float] = None,
    ) -> SatisfactionResult:
        # 1. Explicit customer rating takes direct precedence
        if explicit_rating is not None:
            norm = round((max(1, min(5, explicit_rating)) - 1) / 4.0, 3)
            label = "high" if norm >= 0.70 else ("low" if norm < 0.40 else "moderate")
            return SatisfactionResult(
                satisfaction_score=norm,
                satisfaction_label=label,
                explicit_rating=explicit_rating,
            )

        # 2. Interpretable rule-based conversational estimate
        base = prior_score if prior_score is not None else 0.55

        if text_emotion in ("angry", "frustrated"):
            base -= 0.22
        elif text_emotion == "concerned":
            base -= 0.08
        elif text_emotion in ("satisfied", "happy"):
            base += 0.20

        if failed_attempts >= 2:
            base -= 0.15 * min(2, failed_attempts - 1)

        if customer_confirmed or resolution_status == "resolved":
            base = max(base, 0.80)

        score = float(max(0.05, min(1.0, round(base, 3))))
        if score < 0.38:
            label = "low"
        elif score >= 0.72:
            label = "high"
        else:
            label = "moderate"

        return SatisfactionResult(
            satisfaction_score=score,
            satisfaction_label=label,
            explicit_rating=None,
        )
