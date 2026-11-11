"""
Caller Certainty Analyzer — strictly separate from System Confidence and Emotion.

Measures how sure the caller appears about their own statement
(e.g. "I think ₹2,000 was deducted, not sure" -> low certainty vs
"Definitely ₹2,000 failed twice" -> high certainty).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List


UNCERTAINTY_PATTERNS = re.compile(
    r"\b(?:i think|maybe|perhaps|not sure|unsure|don't know|do not know|probably|might be|guess|forgot|"
    r"can't remember|could be)\b|"
    r"(?:शायद|लगता है|पता नहीं|याद नहीं|कंफर्म नहीं|ବୋଧହୁଏ|ଜଣା ନାହିଁ|ମନେ ନାହିଁ)",
    re.IGNORECASE,
)

CERTAINTY_PATTERNS = re.compile(
    r"\b(?:definitely|certain|100%|always|exact|confirmed|yes exactly)\b|(?<!\bnot\s)\bsure\b|"
    r"(?:बिल्कुल|पक्का|निश्चित|सही है|ନିଶ୍ଚିତ|ଠିକ୍)",
    re.IGNORECASE,
)


@dataclass
class CallerConfidenceResult:
    caller_certainty: float  # [0.0, 1.0]
    certainty_label: str     # "uncertain" | "moderate" | "certain"
    indicators: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CallerConfidenceAnalyzer:
    """Evaluates caller epistemic certainty independently of STT accuracy."""

    def analyze(self, text: str) -> CallerConfidenceResult:
        clean = (text or "").strip()
        if not clean:
            return CallerConfidenceResult(caller_certainty=0.50, certainty_label="moderate", indicators=[])

        score = 0.72
        indicators: List[str] = []

        if UNCERTAINTY_PATTERNS.search(clean):
            score -= 0.38
            indicators.append("hedging_phrase")

        if "?" in clean and len(clean.split()) <= 6:
            score -= 0.10
            indicators.append("short_question")

        if CERTAINTY_PATTERNS.search(clean):
            score += 0.20
            indicators.append("explicit_certainty")

        score = float(max(0.05, min(1.0, round(score, 3))))
        if score < 0.50:
            label = "uncertain"
        elif score >= 0.75:
            label = "certain"
        else:
            label = "moderate"

        return CallerConfidenceResult(
            caller_certainty=score,
            certainty_label=label,
            indicators=indicators,
        )
