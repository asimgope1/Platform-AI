"""
Human-Agent Transfer Policy & PBX Handoff Builder.

Produces multilingual transfer prompts, agent availability checks, and structured
Asterisk WebSocket `{"event": "transfer", ...}` payloads preserving call_uuid
and a masked concise handoff summary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from telephony.intelligence.session_state import ConversationState


OFFER_TRANSFER_PROMPTS = {
    "en": "I understand this has been frustrating. Would you like me to transfer you to a live customer care agent?",
    "hi": "मैं समझती हूँ कि आपको परेशानी हो रही है। क्या आप चाहते हैं कि मैं आपको हमारे ग्राहक सेवा प्रतिनिधि से जोड़ दूँ?",
    "or": "ମୁଁ ବୁଝିପାରୁଛି ଆପଣ ଅସୁବିଧାରେ ଅଛନ୍ତି। ଆପଣ ଚାହାଁନ୍ତି କି ମୁଁ ଆପଣଙ୍କୁ ଆମ ପ୍ରତିନିଧିଙ୍କ ସହିତ ସଂଯୋଗ କରିବି?",
}

INITIATE_TRANSFER_PROMPTS = {
    "en": "Connecting you with a live customer care representative right away. Please hold the line.",
    "hi": "मैं आपको तुरंत हमारे ग्राहक सेवा प्रतिनिधि से कनेक्ट कर रही हूँ। कृपया लाइन पर बने रहें।",
    "or": "ମୁଁ ଆପଣଙ୍କୁ ତୁରନ୍ତ ଆମ ପ୍ରତିନିଧିଙ୍କ ସହିତ ସଂଯୋଗ କରୁଛି। ଦୟାକରି ଲାଇନରେ ରୁହନ୍ତୁ।",
}

TRANSFER_UNAVAILABLE_PROMPTS = {
    "en": "All agents are currently busy, so I will stay with you and help resolve this step by step. Could you share a bit more detail?",
    "hi": "अभी हमारे सभी प्रतिनिधि व्यस्त हैं, इसलिए मैं यहीं आपकी पूरी सहायता करूँगी। कृपया थोड़ा और विवरण बताएं।",
    "or": "ବର୍ତ୍ତମାନ ଆମର ସମସ୍ତ ପ୍ରତିନିଧି ବ୍ୟସ୍ତ ଅଛନ୍ତି, ତେଣୁ ମୁଁ ଏଠାରେ ଆପଣଙ୍କୁ ସାହାଯ୍ୟ କରିବି। ଦୟାକରି ଟିକେ ଅଧିକ ବିବରଣୀ କୁହନ୍ତୁ।",
}


@dataclass
class TransferPayload:
    call_uuid: str
    reason: str
    language: str
    summary: str
    emotion_at_transfer: str
    satisfaction_at_transfer: Optional[float]
    ws_event: Dict[str, Any]


class TransferPolicy:
    """Manages offer prompts and PBX handoff event generation."""

    def __init__(self, agent_available: bool = True) -> None:
        self.agent_available = agent_available

    def get_offer_prompt(self, language: str) -> str:
        return OFFER_TRANSFER_PROMPTS.get(language, OFFER_TRANSFER_PROMPTS["en"])

    def get_initiate_prompt(self, language: str) -> str:
        return INITIATE_TRANSFER_PROMPTS.get(language, INITIATE_TRANSFER_PROMPTS["en"])

    def get_unavailable_prompt(self, language: str) -> str:
        return TRANSFER_UNAVAILABLE_PROMPTS.get(language, TRANSFER_UNAVAILABLE_PROMPTS["en"])

    def build_handoff_payload(self, state: ConversationState, reason: str) -> TransferPayload:
        intent = state.intent or state.service_category or "general_support"
        emo = state.emotion.get("fused", "neutral")
        last_q = state.last_masked_transcript or "caller inquiry"
        summary = (
            f"Intent={intent} | Lang={state.language} | Emotion={emo} ({state.emotion.get('trend', 'stable')}) "
            f"| FailedAttempts={state.failed_attempts} | Reason={reason} | LastQuery='{last_q[:120]}'"
        )
        ws_event = {
            "event": "transfer",
            "uuid": state.call_uuid,
            "reason": reason,
            "language": state.language,
            "intent": intent,
            "emotion": emo,
            "summary": summary,
        }
        return TransferPayload(
            call_uuid=state.call_uuid,
            reason=reason,
            language=state.language,
            summary=summary,
            emotion_at_transfer=emo,
            satisfaction_at_transfer=state.satisfaction_score,
            ws_event=ws_event,
        )
