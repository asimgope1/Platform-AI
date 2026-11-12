"""
Customer Feedback Policy — Triggers rating collection ONLY after resolution is confirmed.
"""
from __future__ import annotations

from typing import Optional


FEEDBACK_ASK_PROMPTS = {
    "en": "I'm glad your issue is resolved! On a scale of 1 to 5, how satisfied are you with today's support?",
    "hi": "यह जानकर खुशी हुई कि आपकी समस्या हल हो गई! 1 से 5 के पैमाने पर, आप आज की सहायता से कितने संतुष्ट हैं?",
    "or": "ଆପଣଙ୍କ ସମସ୍ୟାର ସମାଧାନ ହୋଇଥିବାରୁ ଖୁସି ଲାଗିଲା! ୧ ରୁ ୫ ମଧ୍ୟରେ ଆଜିର ସେବା ପାଇଁ ଆପଣ କେତେ ରେଟିଂ ଦେବେ?",
}

FEEDBACK_THANKS_PROMPTS = {
    "en": "Thank you for rating us {rating} out of 5! Have a wonderful day ahead.",
    "hi": "हमें 5 में से {rating} रेटिंग देने के लिए बहुत-बहुत धन्यवाद! आपका दिन शुभ हो।",
    "or": "ଆମକୁ ୫ ରୁ {rating} ରେଟିଂ ଦେଇଥିବାରୁ ଅଶେଷ ଧନ୍ୟବାଦ! ଆପଣଙ୍କ ଦିନଟି ଶୁଭମୟ ହେଉ।",
}


class FeedbackPolicy:
    """Handles post-resolution feedback prompts and acknowledgment."""

    def get_feedback_prompt(self, language: str) -> str:
        return FEEDBACK_ASK_PROMPTS.get(language, FEEDBACK_ASK_PROMPTS["en"])

    def get_feedback_thanks(self, language: str, rating: Optional[int]) -> str:
        if rating is None:
            if language == "hi":
                return "आपकी बहुमूल्य प्रतिक्रिया के लिए बहुत धन्यवाद! आपका दिन शुभ हो।"
            if language == "or":
                return "ଆପଣଙ୍କ ମତାମତ ପାଇଁ ଧନ୍ୟବାଦ! ଆପଣଙ୍କ ଦିନଟି ଶୁଭମୟ ହେଉ।"
            return "Thank you for your valuable feedback! Have a wonderful day."

        tmpl = FEEDBACK_THANKS_PROMPTS.get(language, FEEDBACK_THANKS_PROMPTS["en"])
        return tmpl.format(rating=rating)

