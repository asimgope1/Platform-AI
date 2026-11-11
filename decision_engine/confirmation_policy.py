"""
Confirmation & Clarification Policy.

Handles:
  • Low audio quality / noise repeat requests (CLARIFY)
  • Low transcript confidence / unclear amounts / low caller certainty (CONFIRM)
  • Resolution confirmation before initiating feedback (CONFIRM_RESOLUTION)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from telephony.intelligence.session_state import ConversationState


CLARIFY_AUDIO_PROMPTS = {
    "en": "I had trouble hearing that clearly due to background noise. Could you please repeat your question slowly?",
    "hi": "शोर के कारण आपकी आवाज़ साफ़ सुनाई नहीं दी। कृपया अपना प्रश्न थोड़ा धीरे दोहराएं?",
    "or": "ଶବ୍ଦ ଯୋଗୁଁ ଆପଣଙ୍କ କଥା ସ୍ପଷ୍ଟ ଶୁଣାଗଲା ନାହିଁ। ଦୟାକରି ଆପଣଙ୍କ ପ୍ରଶ୍ନଟି ଆଉଥରେ ଧୀରେ କୁହନ୍ତୁ?",
}

CONFIRM_RESOLUTION_PROMPTS = {
    "en": "Can you please confirm whether your issue is resolved now, or is there anything else I can assist with?",
    "hi": "कृपया पुष्टि करें, क्या आपकी समस्या का समाधान हो गया है या मैं कुछ और सहायता करूँ?",
    "or": "ଦୟାକରି ନିଶ୍ଚିତ କରନ୍ତୁ, ଆପଣଙ୍କ ସମସ୍ୟାର ସମାଧାନ ହୋଇଗଲା କି, ନା ଆଉ କିଛି ସାହାଯ୍ୟ ଦରକାର?",
}

UNCERTAIN_STATEMENT_PROMPTS = {
    "en": "Just to make sure I help accurately, would you like me to check your transaction status or confirm the exact details first?",
    "hi": "सही सहायता के लिए, क्या आप चाहेंगे कि मैं पहले आपके लेन-देन की स्थिति या विवरण की पुष्टि करूँ?",
    "or": "ସଠିକ୍ ସାହାଯ୍ୟ ପାଇଁ, ମୁଁ ପ୍ରଥମେ ଆପଣଙ୍କ କାରବାର ସ୍ଥିତି କିମ୍ବା ବିବରଣୀ ଯାଞ୍ଚ କରିବି କି?",
}


class ConfirmationPolicy:
    """Produces localized clarification and read-back confirmation prompts."""

    def get_clarify_audio_prompt(self, language: str) -> str:
        return CLARIFY_AUDIO_PROMPTS.get(language, CLARIFY_AUDIO_PROMPTS["en"])

    def get_resolution_confirm_prompt(self, language: str) -> str:
        return CONFIRM_RESOLUTION_PROMPTS.get(language, CONFIRM_RESOLUTION_PROMPTS["en"])

    def get_confirm_prompt(self, state: ConversationState) -> str:
        lang = state.language
        if state.detected_amounts:
            amt = state.detected_amounts[0]
            if lang == "hi":
                return f"पुष्टि करने के लिए, क्या आपने {amt} रुपये की राशि बताई है?"
            if lang == "or":
                return f"ନିଶ୍ଚିତ କରିବା ପାଇଁ, ଆପଣ {amt} ଟଙ୍କା ବିଷୟରେ କହୁଛନ୍ତି କି?"
            return f"Just to confirm, did you mention an amount of {amt} rupees?"

        if state.resolution_status == "potentially_resolved" and not state.customer_confirmed:
            return self.get_resolution_confirm_prompt(lang)

        if state.caller_certainty is not None and state.caller_certainty < 0.50:
            return UNCERTAIN_STATEMENT_PROMPTS.get(lang, UNCERTAIN_STATEMENT_PROMPTS["en"])

        # Generic low transcript confidence repeat/confirm
        snippet = state.last_masked_transcript.strip()
        if snippet and len(snippet.split()) <= 10:
            if lang == "hi":
                return f"क्या मैंने सही सुना कि आपने '{snippet}' के बारे में पूछा?"
            if lang == "or":
                return f"ମୁଁ କ'ଣ ଠିକ୍ ଶୁଣିଲି ଯେ ଆପଣ '{snippet}' ବିଷୟରେ ପଚାରୁଛନ୍ତି?"
            return f"Just to be sure, did you ask about {snippet}?"

        return self.get_clarify_audio_prompt(lang)
