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

    def test_rag_chunk_rejects_bot_clarification_questions(self, brain):
        rag_context = (
            "Customer Spoken Inquiry: \"What are the Video KYicy documents needed?\"\n"
            "Verified Resolution: \"I can certainly look up information on Video KYC documents for you; could you tell me what specific details you are looking for?\"\n"
            "Source: Live Telephone Interaction (a88675ca-65e0-4bba-95c1-c543718e6d35)\n\n"
            "- Mandatory Prerequisites:\n"
            "- Original physical PAN card (scans, laminations, or soft copies are rejected by regulatory rules).\n"
            "- Active Aadhaar-linked mobile device to receive and input the live OTP.\n"
        )
        reply = brain.synthesize("What are the Video KYicy documents needed?", rag_context=rag_context, language="en")
        assert "what specific details you are looking for" not in reply.lower()
        assert "pan" in reply.lower()

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
        assert "understand your concern" in reply.lower() or "don't worry" in reply.lower() or "understand your frustration" in reply.lower()

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

    def test_complaint_status_direct_cmp_query(self, brain):
        reply = brain.synthesize("What is the status of complaint CMP/0001/26-27?", language="en")
        assert "ticket number 1" in reply.lower()
        assert "cmp 0001" in reply.lower()
        assert "canceled" in reply.lower()
        # Verify it does not recite awkward raw strings like slash zero zero zero one
        assert "slash" not in reply.lower()
        assert "26-27" not in reply.lower()

    def test_complaint_status_spoken_and_short_numbers(self, brain):
        reply_num = brain.synthesize("What is the status of complaint 8?", language="en")
        assert "ticket number 8" in reply_num.lower() or "cmp 0008" in reply_num.lower()

        reply_spoken = brain.synthesize("Status of ticket one", language="en")
        assert "ticket number 1" in reply_spoken.lower()

    def test_complaint_status_phone_lookup(self, brain):
        reply = brain.synthesize("Check my complaint status for phone 8976453421", language="en")
        assert "8976453421" in reply
        assert "ticket 1" in reply.lower() and "ticket 2" in reply.lower()

    def test_complaint_status_multi_turn_prompting(self, brain):
        # Turn 1: user asks without ID -> assistant asks for ticket or phone
        turn1_reply = brain.synthesize("I want to check my complaint status", language="en")
        assert "complaint ticket number" in turn1_reply.lower()
        assert "phone number" in turn1_reply.lower()

        # Turn 2: user responds with bare number "1"
        history = [
            ChatMessage(role="user", content="I want to check my complaint status"),
            ChatMessage(role="assistant", content=turn1_reply),
        ]
        turn2_reply = brain.synthesize("1", messages=history, language="en")
        assert "ticket number 1" in turn2_reply.lower()
        assert "canceled" in turn2_reply.lower()

    def test_complaint_status_multilingual(self, brain):
        hi_reply = brain.synthesize("शिकायत नंबर 1 की स्थिति क्या है?", language="hi")
        assert "टिकट नंबर 1" in hi_reply or "सीएमपी 0001" in hi_reply

        or_reply = brain.synthesize("ଅଭିଯୋଗ ଟିକେଟ୍ ୧ ର ସ୍ଥିତି କଣ?", language="or")
        assert "ଟିକେଟ୍ ନମ୍ବର ୧" in or_reply or "ସିଏମ୍‌ପି ୦୦୦୧" in or_reply

    def test_complaint_status_video_mishearing_recovery(self, brain):
        history = [
            ChatMessage(role="user", content="Check my complaint"),
            ChatMessage(role="assistant", content="Please tell me your complaint ticket number (for example 1 or 0001), or your registered phone number."),
        ]
        # Whispered 'zero zero zero 1' transcribed acoustically as 'Video, video, video 1.'
        reply = brain.synthesize("Video, video, video 1.", messages=history, language="en")
        assert "ticket number 1" in reply.lower()
        assert "canceled" in reply.lower()
        # Verify it did not divert to Video KYC documents
        assert "aadhaar" not in reply.lower()
        assert "physical pan" not in reply.lower()

    def test_complaint_status_first_turn_service_category(self, brain):
        # Turn 1 right after selecting Menu Option 2 (complaint), without preceding history in messages
        reply = brain.synthesize("Zero, zero zero one.", service_category="complaint", language="en")
        assert "ticket number 1" in reply.lower()
        assert "canceled" in reply.lower()

    def test_complaint_status_spaced_digits_recovery(self, brain):
        history = [
            ChatMessage(role="user", content="Check my complaint"),
            ChatMessage(role="assistant", content="Please tell me your complaint ticket number (for example 1 or 0001), or your registered phone number."),
        ]
        reply_spaced = brain.synthesize("0 0 0 1", messages=history, language="en")
        assert "ticket number 1" in reply_spaced.lower()

        reply_commas = brain.synthesize("0, 0, 0, 1", messages=history, language="en")
        assert "ticket number 1" in reply_commas.lower()

    def test_complaint_registration_flow_turn1_prompts_details(self, brain):
        # Caller speaks single word 'Register.' as instructed by menu prompt
        bare_reply = brain.synthesize("Register.", service_category="complaint", language="en")
        assert "describe the issue" in bare_reply.lower() or "problem you are facing" in bare_reply.lower()
        assert "ticket number" not in bare_reply.lower()

        # Caller expresses intent to register a new complaint
        en_reply = brain.synthesize("Yes, I want to register a new complaint.", language="en")
        assert "describe the issue" in en_reply.lower() or "problem you are facing" in en_reply.lower()

        hi_reply = brain.synthesize("मुझे एक नई शिकायत दर्ज करनी है", language="hi")
        assert "समस्या का विवरण" in hi_reply

        or_reply = brain.synthesize("ମୁଁ ଏକ ନୂଆ ଅଭିଯୋଗ ଦାଖଲ କରିବାକୁ ଚାହୁଁଛି", language="or")
        assert "ସମସ୍ୟା ବିଷୟରେ" in or_reply

    def test_complaint_registration_flow_turn2_registers_and_lookups_status(self, brain):
        history = [
            ChatMessage(role="user", content="Yes, I want to register a new complaint."),
            ChatMessage(role="assistant", content="Certainly! Please describe the issue or problem you are facing, and I will register your complaint right away."),
        ]
        # Turn 2: Caller describes the issue
        reply = brain.synthesize(
            "My broadband internet is down since yesterday.",
            messages=history,
            language="en",
            caller_phone="9937234512",
        )
        assert "successfully registered" in reply.lower()
        assert "ticket number" in reply.lower()
        assert "cmp" in reply.lower()

        # Extract assigned ticket number from reply or catalog
        from providers.chat.ivr_mini_model import extract_complaint_identifier, COMPLAINT_STATUS_CATALOG
        tkt_id = extract_complaint_identifier(reply)
        assert tkt_id is not None
        assert tkt_id in COMPLAINT_STATUS_CATALOG

        # Turn 3: Caller asks for status of the newly registered ticket
        status_reply = brain.synthesize(f"What is the status of ticket {tkt_id}?", language="en")
        assert f"ticket number {tkt_id}" in status_reply.lower()
        assert "broadband internet is down" in status_reply.lower()

    def test_complaint_registration_single_turn_embedded_description(self, brain):
        reply = brain.synthesize(
            "I want to register a complaint that my broadband internet is not working.",
            language="en",
        )
        assert "successfully registered" in reply.lower()
        assert "ticket number" in reply.lower()

    def test_conversational_branch_status_flow(self, brain):
        history = [
            ChatMessage(
                role="assistant",
                content="Welcome to Complaint Services. Would you like to check the status of an existing complaint, or register a new one?",
            )
        ]
        # Turn 1: Caller chooses status branch
        reply1 = brain.synthesize("Check status", messages=history, service_category="complaint", language="en")
        assert "ticket number" in reply1.lower() or "registered phone" in reply1.lower()
        history.append(ChatMessage(role="user", content="Check status"))
        history.append(ChatMessage(role="assistant", content=reply1))

        # Turn 2: Caller provides spaced digits
        reply2 = brain.synthesize("0 0 0 1", messages=history, service_category="complaint", language="en")
        assert "ticket number 1" in reply2.lower()
        assert "cmp 0001" in reply2.lower()
        assert "swagatika swain" in reply2.lower()
        history.append(ChatMessage(role="user", content="0 0 0 1"))
        history.append(ChatMessage(role="assistant", content=reply2))

        # Turn 3: Caller concludes with gratitude
        reply3 = brain.synthesize("Thank you", messages=history, service_category="complaint", language="en")
        assert "welcome" in reply3.lower()

    def test_conversational_branch_registration_flow(self, brain):
        history = [
            ChatMessage(
                role="assistant",
                content="Welcome to Complaint Services. Would you like to check the status of an existing complaint, or register a new one?",
            )
        ]
        # Turn 1: Caller says 'new one' or 'register a new one'
        reply1 = brain.synthesize("new one", messages=history, service_category="complaint", language="en")
        assert "describe the issue" in reply1.lower() or "problem you are facing" in reply1.lower()
        assert "ticket number 1" not in reply1.lower()  # MUST NOT extract digit 1 from 'one'
        history.append(ChatMessage(role="user", content="new one"))
        history.append(ChatMessage(role="assistant", content=reply1))

        # Turn 2: Caller describes their issue
        reply2 = brain.synthesize(
            "My wifi router has a red blinking light and no connection",
            messages=history,
            service_category="complaint",
            language="en",
        )
        assert "successfully registered" in reply2.lower()
        assert "ticket number" in reply2.lower()
        history.append(ChatMessage(role="user", content="My wifi router has a red blinking light and no connection"))
        history.append(ChatMessage(role="assistant", content=reply2))

        # Turn 3: Caller says 'No that is all, thank you'
        reply3 = brain.synthesize("No that is all, thank you", messages=history, service_category="complaint", language="en")
        assert "welcome" in reply3.lower()

    def test_conversational_branch_multilingual(self, brain):
        hi_history = [
            ChatMessage(
                role="assistant",
                content="शिकायत सेवा में आपका स्वागत है। क्या आप अपनी पुरानी शिकायत की स्थिति देखना चाहते हैं, या नई शिकायत दर्ज करना चाहते हैं?",
            )
        ]
        hi_status = brain.synthesize("पुरानी शिकायत की स्थिति देखनी है", messages=hi_history, language="hi", service_category="complaint")
        assert "टिकट नंबर" in hi_status

        hi_reg = brain.synthesize("नया वाला", messages=hi_history, language="hi", service_category="complaint")
        assert "विवरण" in hi_reg or "समस्या" in hi_reg

        or_history = [
            ChatMessage(
                role="assistant",
                content="ଅଭିଯୋଗ ସେବାକୁ ସ୍ୱାଗତ। ଆପଣ ପୂର୍ବ ଅଭିଯୋଗର ସ୍ଥିତି ଜାଣିବାକୁ ଚାହାଁନ୍ତି, କିମ୍ବା ନୂଆ ଅଭିଯୋଗ ଦାଖଲ କରିବାକୁ ଚାହାଁନ୍ତି?",
            )
        ]
        or_status = brain.synthesize("ପୂର୍ବ ଅଭିଯୋଗର ସ୍ଥିତି", messages=or_history, language="or", service_category="complaint")
        assert "ଟିକେଟ୍ ନମ୍ବର" in or_status

        or_reg = brain.synthesize("ନୂଆ ଗୋଟିଏ", messages=or_history, language="or", service_category="complaint")
        assert "ସମସ୍ୟା" in or_reg or "ଅଭିଯୋଗ" in or_reg

    def test_response_generator_does_not_stall_on_complaint_registration(self):
        from services.conversation.response_generator import _needs_live_action_fallback
        # Must return False so Turn 1 synthesizes instantly in < 2ms without slow Ollama tool stalls
        assert not _needs_live_action_fallback("Yes, I want to register a new complaint.", "complaint")
        assert not _needs_live_action_fallback("lets register new complain", "complaint")
        assert not _needs_live_action_fallback("file a complaint", "complaint")


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

    async def test_live_crm_cache_sync(self):
        adapter = IVRRAGMiniModelAdapter()
        synced = await adapter.brain.sync_live_crm_cache(force=True)
        # If CRM DB is available, catalog reflects live records
        if synced:
            reply8 = adapter.brain.lookup_complaint_status("8", lang="en")
            assert "ticket number 8" in reply8.lower()
            assert "solved" in reply8.lower()
            reply10 = adapter.brain.lookup_complaint_status("10", lang="en")
            assert "ticket number 10" in reply10.lower()
            reply_phone = adapter.brain.lookup_complaint_status("9937234512", lang="en")
            assert "9937234512" in reply_phone
