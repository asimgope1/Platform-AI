"""
Unit tests for the Autonomous IVR RAG Mini-Model.
Verifies self-contained conversational handling, multi-turn reasoning,
RAG chunk synthesis, and multilingual generation without external GPUs or APIs.
"""
import pytest
from platform_core.contracts.requests import ChatMessage, NormalizedChatRequest
from providers.chat.ivr_mini_model import IVRRAGMiniModel, IVRRAGMiniModelAdapter, ivr_mini_brain


class TestIVRRAGMiniModel:

    @pytest.fixture
    def brain(self):
        return IVRRAGMiniModel()

    def test_greeting_synthesis(self, brain):
        en_reply = brain.synthesize("hello", language="en")
        assert "customer support" in en_reply.lower()

        hi_reply = brain.synthesize("नमस्ते", language="hi")
        assert "सहायता" in hi_reply

        or_reply = brain.synthesize("ନମସ୍କାର", language="or")
        assert "ସାହାଯ୍ୟ" in or_reply

    def test_video_kyc_autonomous(self, brain):
        en_reply = brain.synthesize("What do I need for Video KYC?", language="en")
        assert "pan" in en_reply.lower() and "aadhaar" in en_reply.lower()

        hi_reply = brain.synthesize("वीडियो केवाईसी के लिए क्या चाहिए?", language="hi")
        assert "पैन कार्ड" in hi_reply and "आधार" in hi_reply

        or_reply = brain.synthesize("ଭିଡିଓ କେୱାଇସି ପାଇଁ କଣ ଦରକାର?", language="or")
        assert "ପ୍ୟାନ୍ କାର୍ଡ" in or_reply

    def test_upi_refund_autonomous(self, brain):
        en_reply = brain.synthesize("Money was debited but UPI transaction failed", language="en")
        assert "refund" in en_reply.lower() or "hours" in en_reply.lower()

        hi_reply = brain.synthesize("यूपीआई से पैसे कट गए वापस कब आएंगे?", language="hi")
        assert "खाते" in hi_reply and ("घंटों" in hi_reply or "रिफंड" in hi_reply)

    def test_card_blocking_autonomous(self, brain):
        en_reply = brain.synthesize("I lost my debit card please block it", language="en")
        assert "frozen" in en_reply.lower() or "block" in en_reply.lower()

        hi_reply = brain.synthesize("मेरा एटीएम कार्ड खो गया है ब्लॉक कर दो", language="hi")
        assert "ब्लॉक" in hi_reply or "कार्ड" in hi_reply

    def test_multi_turn_anaphoric_resolution(self, brain):
        history = [
            ChatMessage(role="user", content="Tell me about Video KYC"),
            ChatMessage(role="assistant", content="For Video KYC you need physical PAN and Aadhaar."),
        ]
        reply = brain.synthesize("How much time does it take?", messages=history, language="en")
        assert "2 minutes" in reply.lower() or "verification" in reply.lower()

        card_history = [
            ChatMessage(role="user", content="I need a replacement debit card"),
            ChatMessage(role="assistant", content="Your card has been blocked and replaced."),
        ]
        card_reply = brain.synthesize("Kitna time lagega?", messages=card_history, language="hi")
        assert "5 से 7" in card_reply or "कार्यदिवसों" in card_reply

    def test_rag_chunk_synthesis_and_metadata_stripping(self, brain):
        rag_context = (
            "--- BEGIN UNTRUSTED RETRIEVED CONTEXT ---\n"
            "[Source 1]: Audio Prompt ID: PROMPT-001\n"
            "English Voice Transcript: For home loans interest rates start at 8.4 percent per annum.\n"
            "Category: Loans\n"
            "--- END RETRIEVED CONTEXT ---"
        )
        reply = brain.synthesize("What is home loan interest rate?", rag_context=rag_context, language="en")
        assert "PROMPT-001" not in reply
        assert "Audio Prompt ID" not in reply
        assert "8.4 percent" in reply

    def test_appreciation_closing(self, brain):
        en_reply = brain.synthesize("Thank you for your help", language="en")
        assert "welcome" in en_reply.lower() or "wonderful day" in en_reply.lower()

        hi_reply = brain.synthesize("धन्यवाद", language="hi")
        assert "धन्यवाद" in hi_reply or "शुभ हो" in hi_reply

    def test_substring_hi_does_not_trigger_greeting(self, brain):
        reply = brain.synthesize("what think you can do", language="en")
        assert "welcome to customer support" not in reply.lower()
        assert "video kyc" in reply.lower() or "assist" in reply.lower()

    def test_mid_dialogue_greeting_does_not_repeat_welcome(self, brain):
        history = [
            ChatMessage(role="user", content="I have a question about UPI"),
            ChatMessage(role="assistant", content="Failed UPI refunds are credited back in 24 to 48 hours."),
            ChatMessage(role="user", content="hello"),
        ]
        reply = brain.synthesize("hello", messages=history, language="en")
        assert "welcome to customer support" not in reply.lower()
        assert "listening" in reply.lower()

    def test_emotion_frustration_empathy(self, brain):
        reply = brain.synthesize("I am so stressed and frustrated my money was debited!", language="en")
        assert "understand your concern" in reply.lower() or "don't worry" in reply.lower()

        hi_reply = brain.synthesize("मैं बहुत परेशान हूँ मेरे पैसे कट गए!", language="hi")
        assert "परेशानी समझ सकती हूँ" in hi_reply or "चिंता मत कीजिए" in hi_reply

    def test_feedback_handling(self, brain):
        reply = brain.synthesize("I want to give feedback about your service", language="en")
        assert "feedback" in reply.lower() or "appreciate" in reply.lower()

        hi_reply = brain.synthesize("मेरा एक सुझाव है", language="hi")
        assert "प्रतिक्रिया" in hi_reply or "सुझाव" in hi_reply

    def test_satisfaction_acknowledgment(self, brain):
        reply = brain.synthesize("That was very helpful and I am satisfied", language="en")
        assert "glad to hear that" in reply.lower() or "help with anything else" in reply.lower()

    def test_confirmation_handling(self, brain):
        reply = brain.synthesize("Are you sure? Please confirm", language="en")
        assert "confirm" in reply.lower() or "proceed" in reply.lower()

    def test_unrelated_general_knowledge_handling(self, brain):
        reply = brain.synthesize("What is the capital of France?", language="en")
        assert "exclusively in bank customer support" in reply.lower() or "banking" in reply.lower()

    def test_closing_with_followup_question_does_not_close(self, brain):
        # User says thank you but immediately asks another question
        assert not brain._is_closing_intent("thank you, but what about home loans?")
        assert not brain._is_closing_intent("thanks, can you tell me more?")
        assert not brain._is_closing_intent("धन्यवाद, लेकिन क्या आप मदद करेंगे?")

    def test_phonetic_misrecognition_resolution(self, brain):
        # Transcripts where Whisper mishears UPI as 'you be a' or batao as 'better'
        reply = brain.synthesize("you be a payment better", language="en")
        assert "refund" in reply.lower() or "24 to 48" in reply.lower()

        # Service category guided resolution
        cat_reply = brain.synthesize("better details", language="en", service_category="upi_payments")
        assert "refund" in cat_reply.lower() or "24 to 48" in cat_reply.lower()


@pytest.mark.asyncio
class TestIVRRAGMiniModelAdapter:

    async def test_adapter_chat_completion(self):
        adapter = IVRRAGMiniModelAdapter()
        req = NormalizedChatRequest(
            model_alias="ivr-rag-mini",
            messages=[ChatMessage(role="user", content="How do I reset my ATM PIN?")],
        )
        resp = await adapter.chat_completion(req)
        assert resp.choices[0].message.content
        assert "pin" in resp.choices[0].message.content.lower()

    async def test_adapter_streaming(self):
        adapter = IVRRAGMiniModelAdapter()
        req = NormalizedChatRequest(
            model_alias="ivr-rag-mini",
            messages=[ChatMessage(role="user", content="Branch timings?")],
        )
        chunks = []
        async for chunk in adapter.stream_chat_completion(req):
            if chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)
        full_text = "".join(chunks)
        assert "10:00" in full_text or "branch" in full_text.lower()

    async def test_fallback_intent_detected_for_unknown_queries(self):
        adapter = IVRRAGMiniModelAdapter()
        req = NormalizedChatRequest(
            model_alias="ivr-rag-mini",
            messages=[ChatMessage(role="user", content="What is quantum mechanics?")],
        )
        resp = await adapter.chat_completion(req)
        # Should return polite clarification fallback
        assert "clarify" in resp.choices[0].message.content.lower() or "video kyc" in resp.choices[0].message.content.lower()
