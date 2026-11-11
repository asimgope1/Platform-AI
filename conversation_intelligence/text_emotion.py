"""
Multilingual Text Emotion Analyzer (English, Hindi, Odia, Hinglish).

Detects emotional polarity and caller frustration/confusion/satisfaction from
text and dialogue context, avoiding naive reliance on positive words alone
(e.g. "I am fine" spoken angrily or after repeated failures).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


FRUSTRATED_WORDS = re.compile(
    r"\b(?:frustrated|annoyed|useless|pathetic|worst|horrible|ridiculous|terrible|waste of time|stupid|"
    r"sick of|tired of|still not|again and again|same thing|not working|didn't work|broken)\b|"
    r"(?:परेशान|बकवास|बेकार|तंग|बार-बार|काम नहीं|नहीं हो रहा|गलत|ବିରକ୍ତ|ବେକାର|କାମ କରୁନି|ବାରମ୍ବାର)",
    re.IGNORECASE,
)

ANGRY_WORDS = re.compile(
    r"\b(?:angry|furious|cheating|looted|scam|unacceptable|nonsense|hell|idiot)\b|"
    r"(?:गुस्सा|धोखा|लूट|बंद करो|रागा|ପ୍ରତାରଣା)",
    re.IGNORECASE,
)

CONCERNED_WORDS = re.compile(
    r"\b(?:worried|concerned|confused|unsure|uncertain|stuck|missing|lost|deducted|debited|urgent|please help)\b|"
    r"(?:चिंता|कब तक|अटक|कट गया|कट गए|पता नहीं|समझ नहीं|ଚିନ୍ତା|କଟିଗଲା|ଅଟକି)",
    re.IGNORECASE,
)

HAPPY_SATISFIED_WORDS = re.compile(
    r"\b(?:great|awesome|wonderful|perfect|resolved|solved|worked|helpful|thank you|thanks|appreciate|excellent)\b|"
    r"(?:बढ़िया|बहुत अच्छा|मदद मिल|समाधान|धन्यवाद|शुक्रिया|खुश|ବହୁତ ଭଲ|ସମାଧାନ|ଧନ୍ୟବାଦ|ଖୁସି)",
    re.IGNORECASE,
)


@dataclass
class TextEmotionResult:
    primary_emotion: str
    confidence: float
    probabilities: Dict[str, float] = field(default_factory=dict)
    signals: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TextEmotionAnalyzer:
    """Fast lexical + contextual text emotion classifier."""

    def analyze(
        self,
        text: str,
        failed_attempts: int = 0,
        voice_emotion_hint: Optional[str] = None,
    ) -> TextEmotionResult:
        clean = (text or "").strip()
        if not clean:
            return TextEmotionResult(
                primary_emotion="neutral",
                confidence=0.40,
                probabilities={"neutral": 0.80, "frustrated": 0.05, "angry": 0.05, "concerned": 0.05, "satisfied": 0.05},
            )

        probs = {
            "neutral": 0.50,
            "frustrated": 0.05,
            "angry": 0.03,
            "concerned": 0.08,
            "satisfied": 0.05,
        }
        signals: List[str] = []

        if ANGRY_WORDS.search(clean):
            probs["angry"] += 0.65
            probs["frustrated"] += 0.30
            signals.append("angry_lexicon")

        if FRUSTRATED_WORDS.search(clean):
            probs["frustrated"] += 0.60
            signals.append("frustrated_lexicon")

        if CONCERNED_WORDS.search(clean):
            probs["concerned"] += 0.45
            signals.append("concerned_lexicon")

        # Repeated failure context boosts frustration even on polite phrasing
        if failed_attempts >= 2:
            probs["frustrated"] += 0.35
            signals.append("repeated_failures")

        # Guard against surface "I am fine / okay" sarcasm when acoustic voice is angry/frustrated
        has_positive = bool(HAPPY_SATISFIED_WORDS.search(clean))
        if has_positive:
            if voice_emotion_hint in ("angry", "frustrated") or failed_attempts >= 2:
                probs["frustrated"] += 0.25
                signals.append("masked_positive_discrepancy")
            else:
                probs["satisfied"] += 0.60
                signals.append("satisfied_lexicon")

        # Exclamation/question density hints at urgency
        if clean.count("!") >= 1:
            probs["frustrated"] += 0.12
            probs["angry"] += 0.08

        total = sum(probs.values())
        norm_probs = {k: round(v / total, 3) for k, v in probs.items()}
        primary = max(norm_probs, key=lambda k: norm_probs[k])
        return TextEmotionResult(
            primary_emotion=primary,
            confidence=round(norm_probs[primary], 3),
            probabilities=norm_probs,
            signals=signals,
        )
