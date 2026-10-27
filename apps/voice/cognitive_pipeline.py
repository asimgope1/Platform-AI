"""
Com-one Platform v2.0 - 4-Stage Cognitive Pipeline
Executes:
01. Perceive: Real-Time ASR (Whisper)
02. Understand: Intent Classification, Emotion & Sentiment Scoring, Escalation Detection
03. Respond: pgvector RAG + Gemma 4 LLM Generation + Neural TTS
04. Act: Autonomous CRM Execution & Agent Handover Packet Generation
"""
import time
import base64
import logging
from typing import Dict, Any, Optional, List
from django.utils import timezone
from apps.telephony.models import CallSession
from apps.telephony.bridge import AsteriskBridge
from apps.conversations.models import ConversationTurn
from apps.providers.services import ProviderBridge
from apps.knowledge_base.services import KnowledgeBaseService
from apps.crm_bridge.services import CrmBridgeService
from asgiref.sync import sync_to_async

logger = logging.getLogger(__name__)

class CognitiveVoicePipeline:
    @classmethod
    async def process_call_turn(
        cls,
        session: CallSession,
        audio_bytes: Optional[bytes] = None,
        text_input: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute all 4 cognitive stages for a single caller interaction."""
        t_start = time.time()
        result: Dict[str, Any] = {
            "call_id": session.call_id,
            "stage_timings_ms": {},
            "escalated": False,
        }

        # -------------------------------------------------------------
        # STAGE 01: PERCEIVE (Real-time Speech -> Text)
        # -------------------------------------------------------------
        t0 = time.time()
        caller_text = ""
        stt_meta = {}
        if audio_bytes:
            stt_res = await ProviderBridge.transcribe_audio(audio_bytes)
            caller_text = stt_res.get("text", "").strip()
            stt_meta = stt_res
        elif text_input:
            caller_text = text_input.strip()

        if not caller_text:
            caller_text = "..."

        t_perceive = int((time.time() - t0) * 1000)
        result["stage_timings_ms"]["01_perceive"] = t_perceive
        result["caller_transcript"] = caller_text

        # Record caller turn in database via sync_to_async
        await sync_to_async(cls._record_turn)(
            session=session,
            speaker="caller",
            text=caller_text,
            latency_ms=t_perceive
        )

        # -------------------------------------------------------------
        # STAGE 02: UNDERSTAND (Intent + Emotion + Context + Escalation)
        # -------------------------------------------------------------
        t1 = time.time()
        intent, sentiment_score, emotion = cls._analyze_sentiment_and_intent(caller_text)
        await sync_to_async(cls._update_session_intent_emotion)(session, intent, sentiment_score, emotion)

        # Check Escalation Signals (Deck Page 4):
        # 1. Explicit request: "let me speak to someone", "agent", "बात करवाओ", "प्रतिनिधि"
        # 2. Negative Sentiment Spike (sentiment_score <= -0.6)
        # 3. Conversation Loop: repeating question 2x
        escalation_triggered = False
        escalation_reason = None

        lower_text = caller_text.lower()
        if any(w in lower_text for w in ["agent", "human", "representative", "बात करवाओ", "प्रतिनिधि", "इंसान"]):
            escalation_triggered = True
            escalation_reason = "Explicit Human Request detected ('Let me speak to someone')"
        elif sentiment_score <= -0.6 or emotion in ["frustrated", "angry"]:
            escalation_triggered = True
            escalation_reason = f"Negative Sentiment Spike detected (Score: {sentiment_score:.2f}, Emotion: {emotion})"

        t_understand = int((time.time() - t1) * 1000)
        result["stage_timings_ms"]["02_understand"] = t_understand
        result["intent"] = intent
        result["sentiment_score"] = sentiment_score
        result["emotion"] = emotion

        # -------------------------------------------------------------
        # STAGE 03: RESPOND (RAG Context + Gemma LLM + Neural TTS)
        # -------------------------------------------------------------
        t2 = time.time()
        rag_context = ""
        retrieved_chunks = []
        if not escalation_triggered:
            retrieved_chunks = await KnowledgeBaseService.retrieve_ivr_context(
                query=caller_text,
                tenant_id=session.tenant_id,
                top_k=2
            )
            if retrieved_chunks:
                rag_context = "\n".join([f"- {c['content']}" for c in retrieved_chunks])

        # Generate response via Gemma 4 LLM
        system_instruction = (
            "You are Com-one AI Voice Agent for automated IVR call assistance. "
            "Keep replies concise, clear, polite, and under 2-3 sentences so it sounds natural on telephone calls. "
            "If caller speaks Hindi, reply in natural conversational Hindi. "
            "Use the provided knowledge context accurately without guessing."
        )

        messages = [
            {"role": "system", "content": system_instruction},
        ]
        if rag_context:
            messages.append({"role": "system", "content": f"Verified Knowledge Context:\n{rag_context}"})

        # Multi-turn history
        past_turns = await sync_to_async(cls._get_recent_turns)(session, limit=4)
        for past_speaker, past_text in past_turns:
            role = "user" if past_speaker == "caller" else "assistant"
            messages.append({"role": role, "content": past_text})

        if escalation_triggered:
            ai_reply = (
                "मैं समझता हूँ। मैं तुरंत आपकी कॉल हमारे वरिष्ठ ग्राहक सेवा प्रतिनिधि (Customer Care Specialist) "
                "को ट्रांसफर कर रहा हूँ। कृपया एक क्षण प्रतीक्षा करें।"
                if session.language == "hi" else
                "I completely understand. I am transferring your call directly to a senior customer care specialist right away. Please hold."
            )
        else:
            chat_res = await ProviderBridge.generate_chat_response(messages=messages, max_tokens=150)
            ai_reply = chat_res.get("content", "Thank you for contacting customer support. How may I help you?")

        # Synthesize audio (TTS)
        audio_reply_bytes = await ProviderBridge.synthesize_speech(text=ai_reply)

        t_respond = int((time.time() - t2) * 1000)
        result["stage_timings_ms"]["03_respond"] = t_respond
        result["ai_response_text"] = ai_reply
        result["ai_response_audio_base64"] = base64.b64encode(audio_reply_bytes).decode("utf-8") if audio_reply_bytes else None

        # Record AI agent turn via sync_to_async
        await sync_to_async(cls._record_turn)(
            session=session,
            speaker="ai_agent",
            text=ai_reply,
            intent=intent,
            sentiment_score=sentiment_score,
            emotion=emotion,
            chunks=retrieved_chunks,
            latency_ms=t_respond
        )

        # -------------------------------------------------------------
        # STAGE 04: ACT (CRM Execution & Escalation Handover)
        # -------------------------------------------------------------
        t3 = time.time()
        crm_action_summary = {}

        if escalation_triggered:
            await sync_to_async(AsteriskBridge.escalate_call)(session, reason=escalation_reason)
            result["escalated"] = True
            result["escalation_handover_packet"] = await sync_to_async(cls.generate_agent_handover_packet)(session, escalation_reason)
            crm_action_summary = CrmBridgeService.execute_midcall_action(
                "escalate_to_human",
                {"call_id": session.call_id, "reason": escalation_reason}
            )
        else:
            # Autonomous CRM action based on intent
            if "KYC" in intent:
                crm_action_summary = CrmBridgeService.execute_midcall_action("video_kyc_link_dispatched", {"phone": session.caller_number})
            elif "UPI" in intent or "REFUND" in intent:
                crm_action_summary = CrmBridgeService.execute_midcall_action("refund_status_checked", {"phone": session.caller_number})

        t_act = int((time.time() - t3) * 1000)
        result["stage_timings_ms"]["04_act"] = t_act
        result["crm_action"] = crm_action_summary
        result["total_turn_latency_ms"] = int((time.time() - t_start) * 1000)

        return result

    @classmethod
    def _record_turn(cls, session, speaker, text, intent=None, sentiment_score=0.0, emotion="calm", chunks=None, latency_ms=0):
        turn_count = session.turns.count() + 1
        return ConversationTurn.objects.create(
            call_session=session,
            turn_index=turn_count,
            speaker=speaker,
            text=text,
            detected_intent=intent,
            sentiment_score=sentiment_score,
            detected_emotion=emotion,
            retrieved_chunks=chunks or [],
            latency_ms=latency_ms,
        )

    @classmethod
    def _update_session_intent_emotion(cls, session, intent, sentiment_score, emotion):
        session.current_intent = intent
        session.caller_sentiment = sentiment_score
        session.caller_emotion = emotion
        session.save()

    @classmethod
    def _get_recent_turns(cls, session, limit=4):
        turns = session.turns.order_by("-turn_index")[:limit]
        return [(t.speaker, t.text) for t in reversed(list(turns))]

    @classmethod
    def _analyze_sentiment_and_intent(cls, text: str) -> (str, float, str):
        """Analyze text to extract intent, sentiment score (-1.0 to 1.0) and emotion."""
        lower = text.lower()
        # Intent detection
        if any(w in lower for w in ["upi", "पैसा", "refund", "रिफंड", "कट गया"]):
            intent = "UPI_REFUND_INQUIRY"
        elif any(w in lower for w in ["kyc", "केवाईसी", "pan", "aadhar", "पैन कार्ड"]):
            intent = "VIDEO_KYC_VERIFICATION"
        elif any(w in lower for w in ["order", "ऑर्डर", "tracking", "डिलीवरी", "cancel"]):
            intent = "ORDER_TRACKING_CANCELLATION"
        elif any(w in lower for w in ["agent", "human", "representative", "बात", "प्रतिनिधि"]):
            intent = "LIVE_AGENT_TRANSFER"
        else:
            intent = "GENERAL_INQUIRY"

        # Emotion & sentiment scoring
        if any(w in lower for w in ["गुस्सा", "बेकार", "stupid", "worst", "fraud", "bad", "धोखा", "लूट"]):
            return intent, -0.85, "angry"
        elif any(w in lower for w in ["agent se baat karao", "talk to human", "not helping", "नहीं समझ रहे", "तुरंत बात"]):
            return intent, -0.65, "frustrated"
        elif any(w in lower for w in ["thank", "धन्यवाद", "शुक्रिया", "great", "good", "सहायक"]):
            return intent, 0.75, "calm"
        else:
            return intent, 0.05, "neutral"

    @classmethod
    def generate_agent_handover_packet(cls, session: CallSession, flagged_reason: str) -> Dict[str, Any]:
        """
        Builds the 5-part Agent Screen Handover Packet (Page 4 of Com-one deck):
        01. Full Verbatim Transcript
        02. Escalation Reason Highlighted
        03. Intent, Sentiment & Emotion Tag
        04. CRM Record Auto-Surfaced
        05. AI-Suggested Resolution Path
        """
        turns = session.turns.all()
        transcript_history = []
        for t in turns:
            transcript_history.append({
                "turn": t.turn_index,
                "speaker": t.speaker,
                "text": t.text,
                "time": t.created_at.strftime("%H:%M:%S")
            })

        crm_record = CrmBridgeService.lookup_customer_by_number(session.caller_number)

        suggested_action = (
            "Verify UPI transaction reference ID (RRN) in bank gateway and initiate manual dispute if deduction exceeded 24 hours."
            if "UPI" in (session.current_intent or "") else
            "Offer manual verification via secure link and expedite account access."
        )

        return {
            "01_full_verbatim_transcript": transcript_history,
            "02_escalation_reason_highlighted": flagged_reason,
            "03_intent_sentiment_emotion": {
                "intent": session.current_intent or "Live Support Request",
                "sentiment_score": session.caller_sentiment,
                "emotion_tag": session.caller_emotion,
            },
            "04_crm_record_auto_surfaced": crm_record,
            "05_ai_suggested_resolution_path": {
                "recommended_action": suggested_action,
                "recommended_by": "Com-one Escalation Intelligence"
            }
        }
