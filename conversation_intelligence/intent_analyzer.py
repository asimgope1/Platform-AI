"""
Intent Analyzer for Multilingual Customer Support Calls (English, Hindi, Odia).

Extracts:
  • Structured banking/IVR intent (`upi_payment_failed`, `upi_refund`, `video_kyc`,
    `card_block`, `card_pin`, `account_balance`, `loans_emi`, `branch_hours`,
    `complaint`, `human_agent`, `greeting`, `appreciation`, `confirmation_yes`,
    `confirmation_no`, `feedback_rating`, `unrelated_general`, `unknown`)
  • `explicit_agent_request` flag (immediate human transfer intent)
  • `requires_mandatory_escalation` flag (fraud, scam, unauthorized debit, OTP theft)
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class IntentResult:
    intent: str
    confidence: float
    explicit_agent_request: bool = False
    requires_mandatory_escalation: bool = False
    is_off_topic: bool = False
    rating_value: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


AGENT_PATTERNS = re.compile(
    r"\b(?:agent|representative|human|person|executive|operator|manager|talk to someone|speak to someone)\b|"
    r"(?:एजेंट|प्रतिनिधि|इंसान|किसी\s*से\s*बात|ପ୍ରତିନିଧି|ଏଜେଣ୍ଟ|ମଣିଷ)",
    re.IGNORECASE,
)

MANDATORY_ESCALATION_PATTERNS = re.compile(
    r"\b(?:fraud|scam|scammed|stolen|hacked|unauthorized|compromised|cyber\s*crime|phishing)\b|"
    r"(?:धोखाधड़ी|फ्रॉड|चोरी|हैक|अनधिकृत|ठगी|ପ୍ରତାରଣା|ଫ୍ରଡ୍|ଚୋରି)",
    re.IGNORECASE,
)

CONFIRM_YES_PATTERNS = re.compile(
    r"^(?:yes|yeah|yep|sure|correct|right|ok|okay|confirm|confirmed|haan|haanji|bilkul|ji|theek hai|thik hai|"
    r"हाँ|हाँजी|जी|बिल्कुल|सही|ठीक है|ହଁ|ଆଜ୍ଞା|ଠିକ୍ ଅଛି)[!.,? ]*$",
    re.IGNORECASE,
)

CONFIRM_NO_PATTERNS = re.compile(
    r"^(?:no|nope|not really|incorrect|wrong|cancel|nahin|nahi|galat|"
    r"नहीं|गलत|ନା|ନାହିଁ|ଭୁଲ୍)[!.,? ]*$",
    re.IGNORECASE,
)

RATING_PATTERN = re.compile(r"\b([1-5])\s*(?:out of 5|stars?|rating|में से 5|स्टार|ରେଟିଂ)?\b", re.IGNORECASE)
_RATING_WORDS: Dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "ek": 1, "do": 2, "teen": 3, "chaar": 4, "char": 4, "paanch": 5, "panch": 5,
    "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पांच": 5, "पाँच": 5,
    "ଏକ": 1, "ଦୁଇ": 2, "ତିନି": 3, "ଚାରି": 4, "ପାଞ୍ଚ": 5,
}
_RATING_WORD_PATTERN = re.compile(
    r"(?:^|\s)(" + "|".join(re.escape(w) for w in _RATING_WORDS) + r")(?:$|\s|[.!?,।])",
    re.IGNORECASE,
)

OFF_TOPIC_PATTERNS = re.compile(
    r"\b(?:weather|cricket|football|movie|recipe|joke|poem|capital of|president|prime minister|coding|python|java|physics|chemistry)\b|"
    r"(?:मौसम|क्रिकेट|फिल्म|चुटकुला|କବିତା|ପାଣିପାଗ|କ୍ରିକେଟ୍)",
    re.IGNORECASE,
)


class IntentAnalyzer:
    """Fast multilingual intent & escalation rule classifier."""

    def analyze(
        self,
        text: str,
        service_category: Optional[str] = None,
        awaiting_feedback: bool = False,
        awaiting_order_id_digits: bool = False,
    ) -> IntentResult:
        clean = (text or "").strip()
        lower = clean.lower()
        if not clean:
            return IntentResult(intent="unknown", confidence=0.0)

        # 1. Explicit feedback rating when expected or formatted as rating
        if awaiting_feedback:
            digit_match = RATING_PATTERN.search(clean)
            word_match = _RATING_WORD_PATTERN.search(lower)
            if word_match and (not digit_match or word_match.start() < digit_match.start()):
                word_rating = _RATING_WORDS.get(word_match.group(1).strip())
                if word_rating is not None:
                    return IntentResult(
                        intent="feedback_rating",
                        confidence=0.90,
                        rating_value=word_rating,
                    )
            if digit_match:
                return IntentResult(
                    intent="feedback_rating",
                    confidence=0.92,
                    rating_value=int(digit_match.group(1)),
                )

        # 2. Explicit agent request (highest priority for immediate routing)
        if AGENT_PATTERNS.search(clean):
            return IntentResult(
                intent="human_agent",
                confidence=0.95,
                explicit_agent_request=True,
            )

        # 3. Mandatory security escalation (fraud / unauthorized activity)
        if MANDATORY_ESCALATION_PATTERNS.search(clean):
            return IntentResult(
                intent="fraud_escalation",
                confidence=0.95,
                requires_mandatory_escalation=True,
            )

        # 4. Short confirmation responses
        if CONFIRM_YES_PATTERNS.match(clean):
            return IntentResult(intent="confirmation_yes", confidence=0.90)
        if CONFIRM_NO_PATTERNS.match(clean):
            return IntentResult(intent="confirmation_no", confidence=0.90)

        # 5. Off-topic non-banking guard
        if OFF_TOPIC_PATTERNS.search(clean) and not any(
            w in lower for w in ("upi", "bank", "kyc", "card", "account", "loan", "emi", "refund", "बैंक", "खाता", "ବ୍ୟାଙ୍କ")
        ):
            return IntentResult(intent="unrelated_general", confidence=0.88, is_off_topic=True)

        # 6. Banking domain intents
        if any(w in lower for w in ("upi", "यूपीआई", "ୟୁପିଆଇ", "gpay", "phonepe", "paytm", "bhim", "payment", "भुगतान", "पेमेंट", "ପେମେଣ୍ଟ")):
            if any(w in lower for w in ("fail", "failed", "stuck", "deduct", "debit", "विफल", "फेल", "कट", "ବିଫଳ", "କଟି")):
                return IntentResult(intent="upi_payment_failed", confidence=0.90)
            return IntentResult(intent="upi_refund", confidence=0.86)

        if any(w in lower for w in ("kyc", "केवाईसी", "କେୱାଇସି", "video kyc", "pan", "aadhaar", "aadhar", "पैन", "आधार", "ପ୍ୟାନ୍", "ଆଧାର")):
            return IntentResult(intent="video_kyc", confidence=0.90)

        if any(w in lower for w in ("block", "lost", "freeze", "ब्लॉक", "खो", "ବ୍ଲକ୍", "ହଜି")):
            return IntentResult(intent="card_block", confidence=0.88)

        if any(w in lower for w in ("atm", "pin", "पिन", "ପିନ୍", "card", "कार्ड", "କାର୍ଡ")):
            return IntentResult(intent="card_pin", confidence=0.84)

        if any(w in lower for w in ("balance", "statement", "बैलेंस", "खाता", "ବାଲାନ୍ସ", "ଖାତା")):
            return IntentResult(intent="account_balance", confidence=0.85)

        if any(w in lower for w in ("loan", "emi", "interest", "ऋण", "ईएमआई", "ଋଣ", "ଇଏମଆଇ")):
            return IntentResult(intent="loans_emi", confidence=0.86)

        if any(w in lower for w in (
            "order", "order id", "delivery", "parcel", "shipment", "courier", "track", "tracking",
            "ऑर्डर", "डिलीवरी", "पार्सल", "ट्रैक", "ଅର୍ଡର", "ଡେଲିଭରି", "ପାର୍ସଲ",
        )):
            return IntentResult(intent="order_status", confidence=0.90)

        if awaiting_order_id_digits and (
            re.search(r"[\d०-९୦-୯]", clean)
            or any(
                w in lower
                for w in (
                    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                    "शून्य", "एक", "दो", "तीन", "चार", "पांच", "पाँच", "छह", "सात", "आठ", "नौ",
                    "ଶୂନ", "ଏକ", "ଦୁଇ", "ତିନି", "ଚାରି", "ପାଞ୍ଚ", "ଛଅ", "ସାତ", "ଆଠ", "ନଅ",
                )
            )
        ):
            return IntentResult(intent="order_status", confidence=0.92)

        if any(w in lower for w in ("complaint", "issue", "problem", "ticket", "शिकायत", "समस्या", "ଅଭିଯୋଗ", "ସମସ୍ୟା")):
            return IntentResult(intent="complaint", confidence=0.80)

        if any(w in lower for w in ("thank", "thanks", "dhanyawad", "dhanyavad", "shukriya", "धन्यवाद", "शुक्रिया", "ଧନ୍ୟବାଦ")):
            return IntentResult(intent="appreciation", confidence=0.88)

        if any(w in lower for w in ("hi", "hello", "namaste", "नमस्ते", "ନମସ୍କାର")):
            if len(clean.split()) <= 4:
                return IntentResult(intent="greeting", confidence=0.85)

        if service_category:
            return IntentResult(intent=service_category, confidence=0.55)

        return IntentResult(intent="unknown", confidence=0.35)

