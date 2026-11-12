"""
Transcript Intelligence & Calibrated Confidence Estimator (Milestone 2).

Validates Whisper transcripts using:
  • Audio quality metrics (SNR, VAD probability, clipping)
  • Whisper decoding diagnostics (avg_logprob, no_speech_prob, compression_ratio)
  • Banking domain dictionary & phonetic normalization
  • Language script & repetition hallucination checks
  • Contextual confirmation flags for critical financial details (amounts, status)

Never uses an LLM to silently rewrite uncertain caller transcripts.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from audio_intelligence.audio_quality import AudioQualityReport


BANKING_VOCAB_PATTERNS = [
    (r"\b(?:you\s*be\s*[ai]|you\s*pay|u\s*p\s*i|up\s*i|yupi|u\s*b\s*i)\b", "UPI"),
    (r"\b(?:(?:vedio|video)\s*k\s*y\s*c|kay\s*why\s*see)\b", "Video KYC"),
    (r"\b(?:k\s*y\s*c)\b", "KYC"),
    (r"\b(?:pen\s*card|pan\s*car)\b", "PAN card"),
    (r"\b(?:aadhar\s*card|adhar\s*card)\b", "Aadhaar card"),
    (r"\b(?:aadhar|adhar)\b", "Aadhaar"),
    (r"\b(?:e\s*m\s*i)\b", "EMI"),
    (r"\b(?:n\s*e\s*f\s*t)\b", "NEFT"),
    (r"\b(?:i\s*m\s*p\s*s)\b", "IMPS"),
    (r"\b(?:r\s*t\s*g\s*s)\b", "RTGS"),
    (r"\b(?:o\s*t\s*p)\b", "OTP"),
]

BANKING_TERMS_SET = {
    "upi", "kyc", "pan", "aadhaar", "refund", "payment", "transaction", "account",
    "balance", "card", "debit", "credit", "atm", "pin", "loan", "emi", "complaint",
    "bank", "branch", "cheque", "check", "transfer", "neft", "imps", "rtgs", "failed",
    "pending", "debited", "deducted", "blocked", "freeze", "agent", "representative",
    "order", "status", "delivery", "track", "tracking", "shipment", "courier", "parcel",
    "यूपीआई", "केवाईसी", "पैन", "आधार", "रिफंड", "खाता", "बैलेंस", "कार्ड", "ऋण", "शिकायत", "पेमेंट",
    "ऑर्डर", "डिलीवरी", "स्टेटस", "पार्सल", "ट्रैक",
    "ୟୁପିଆଇ", "କେୱାଇସି", "ଆଧାର", "ପ୍ୟାନ୍", "ଖାତା", "ବାଲାନ୍ସ", "କାର୍ଡ", "ଅଭିଯୋଗ",
    "ଅର୍ଡର", "ଡେଲିଭରି", "ସ୍ଥିତି", "ପାର୍ସଲ",
}

AMOUNT_PATTERN = re.compile(r"(?:₹|rs\.?|inr|rupees?|रुपये|टके|ଟଙ୍କା)\s*([\d,]+(?:\.\d{1,2})?)|([\d,]{2,7})\s*(?:rs\.?|inr|rupees?|रुपये|ଟଙ୍କା)", re.IGNORECASE)
PHONE_OR_ACCT_PATTERN = re.compile(r"\b\d{9,18}\b")
UPI_ID_PATTERN = re.compile(r"\b[\w.\-+]{2,}@[a-zA-Z]{2,}\b")


def mask_sensitive_text(text: str) -> str:
    """Mask long numeric account/phone digits and UPI IDs for safe logging."""
    if not text:
        return ""
    masked = UPI_ID_PATTERN.sub("[REDACTED_UPI_ID]", text)
    masked = PHONE_OR_ACCT_PATTERN.sub(lambda m: f"[REDACTED_{len(m.group(0))}DIGITS]", masked)
    return masked


@dataclass
class TranscriptAnalysis:
    raw_text: str
    normalized_text: str
    masked_text: str
    transcript_confidence: float
    language_match: bool
    is_meaningful: bool
    needs_clarification: bool
    needs_confirmation: bool
    detected_amounts: List[str] = field(default_factory=list)
    banking_terms_matched: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    suggested_confirmation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TranscriptAnalyzer:
    """Calibrated transcript validator and banking normalizer."""

    def __init__(
        self,
        clarify_threshold: float = 0.45,
        confirm_threshold: float = 0.60,
    ) -> None:
        self.clarify_threshold = clarify_threshold
        self.confirm_threshold = confirm_threshold

    def analyze(
        self,
        stt_result: Dict[str, Any] | str,
        audio_quality: Optional[AudioQualityReport] = None,
        selected_language: str = "en",
    ) -> TranscriptAnalysis:
        if isinstance(stt_result, str):
            raw_text = stt_result.strip()
            diagnostics: Dict[str, Any] = {}
        else:
            raw_text = str(stt_result.get("text") or "").strip()
            diagnostics = dict(stt_result.get("diagnostics") or {})

        if not raw_text or not re.search(r"\w", raw_text):
            return TranscriptAnalysis(
                raw_text=raw_text,
                normalized_text="",
                masked_text="",
                transcript_confidence=0.0,
                language_match=False,
                is_meaningful=False,
                needs_clarification=True,
                needs_confirmation=False,
                issues=["empty_transcript"],
            )

        normalized = raw_text
        for pattern, replacement in BANKING_VOCAB_PATTERNS:
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\s+", " ", normalized).strip()

        issues: List[str] = []
        confidence = 0.78  # Base calibrated confidence prior for decoded speech

        # 1. Audio quality coupling
        if audio_quality is not None:
            if not audio_quality.is_speech_present:
                confidence -= 0.30
                issues.append("weak_speech_signal")
            elif audio_quality.snr_db < 6.0:
                confidence -= 0.18
                issues.append("low_snr")
            elif audio_quality.snr_db >= 18.0:
                confidence += 0.08

            if audio_quality.clipping_pct > 5.0:
                confidence -= 0.12
                issues.append("clipped_audio")

        # 2. Whisper decoding diagnostics (if available)
        if diagnostics:
            avg_logprob = diagnostics.get("avg_logprob")
            if isinstance(avg_logprob, (int, float)):
                if avg_logprob < -1.0:
                    confidence -= 0.25
                    issues.append("low_decoder_logprob")
                elif avg_logprob > -0.35:
                    confidence += 0.08

            no_speech_prob = diagnostics.get("no_speech_prob")
            if isinstance(no_speech_prob, (int, float)) and no_speech_prob > 0.65:
                confidence -= 0.25
                issues.append("high_no_speech_prob")

            comp_ratio = diagnostics.get("compression_ratio")
            if isinstance(comp_ratio, (int, float)) and comp_ratio > 2.4:
                confidence -= 0.25
                issues.append("repetitive_decoding")

            if diagnostics.get("hallucination_filtered"):
                confidence -= 0.35
                issues.append("hallucination_filtered")

        # 3. Lexical coherence & repetition check
        tokens = re.findall(r"\w+", normalized.lower())
        if len(tokens) >= 4:
            unique_ratio = len(set(tokens)) / len(tokens)
            if unique_ratio < 0.40:
                confidence -= 0.28
                issues.append("low_lexical_diversity")

        if len(tokens) == 1 and len(tokens[0]) <= 2 and tokens[0] not in {"hi", "no", "ok"}:
            confidence -= 0.25
            issues.append("single_short_token")

        # 4. Language consistency check
        has_devanagari = any("\u0900" <= ch <= "\u097f" for ch in normalized)
        has_odia = any("\u0b00" <= ch <= "\u0b7f" for ch in normalized)
        has_latin = any("a" <= ch.lower() <= "z" for ch in normalized)
        is_pure_digits = bool(re.fullmatch(r"[\d\s\-.,#०-९୦-୯]+", normalized))

        language_match = True
        if selected_language == "hi" and not has_devanagari and not is_pure_digits:
            language_match = False
            confidence -= 0.18 if has_latin else 0.25
            issues.append("script_mismatch_hi")
        elif selected_language == "or" and not has_odia and not is_pure_digits:
            language_match = False
            confidence -= 0.18 if (has_devanagari or has_latin) else 0.25
            issues.append("script_mismatch_or")

        # 5. Domain overlap & monetary entity extraction
        matched_terms = sorted({t for t in tokens if t in BANKING_TERMS_SET})
        if matched_terms:
            confidence += min(0.10, 0.04 * len(matched_terms))

        detected_amounts: List[str] = []
        for m in AMOUNT_PATTERN.finditer(normalized):
            val = m.group(1) or m.group(2)
            if val:
                detected_amounts.append(val)

        confidence = float(max(0.0, min(1.0, round(confidence, 3))))
        is_meaningful = len(tokens) >= 1 and confidence >= 0.25

        needs_clarification = confidence < self.clarify_threshold or not is_meaningful
        needs_confirmation = (
            not needs_clarification
            and (confidence < self.confirm_threshold or bool(detected_amounts and confidence < 0.82))
        )

        suggested_confirmation = None
        if needs_confirmation and detected_amounts:
            amt = detected_amounts[0]
            if selected_language == "hi":
                suggested_confirmation = f"पुष्टि करने के लिए, क्या आपने {amt} रुपये की राशि बताई है?"
            elif selected_language == "or":
                suggested_confirmation = f"ନିଶ୍ଚିତ କରିବା ପାଇଁ, ଆପଣ {amt} ଟଙ୍କା ବିଷୟରେ କହୁଛନ୍ତି କି?"
            else:
                suggested_confirmation = f"Just to confirm, did you mention an amount of {amt} rupees?"

        return TranscriptAnalysis(
            raw_text=raw_text,
            normalized_text=normalized,
            masked_text=mask_sensitive_text(normalized),
            transcript_confidence=confidence,
            language_match=language_match,
            is_meaningful=is_meaningful,
            needs_clarification=needs_clarification,
            needs_confirmation=needs_confirmation,
            detected_amounts=detected_amounts,
            banking_terms_matched=matched_terms,
            issues=issues,
            suggested_confirmation=suggested_confirmation,
        )
