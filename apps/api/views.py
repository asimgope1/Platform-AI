"""
Com-one Platform v2.0 - REST API Views
Handles Telephony Call Lifecycle, Real-Time Audio Turns, CRM Auto-Surfacing, and CDR Reporting.
"""
import io
import json
import base64
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import get_object_or_404
from asgiref.sync import async_to_sync

from apps.telephony.models import CallSession, CDRRecord
from apps.telephony.bridge import AsteriskBridge
from apps.voice.cognitive_pipeline import CognitiveVoicePipeline
from apps.crm_bridge.services import CrmBridgeService

class StartCallView(APIView):
    """Initiate a call session (simulated or via Asterisk ARI/AMI event)."""
    def post(self, request):
        caller_number = request.data.get("caller_number", "+919876543210")
        did_number = request.data.get("did_number", "1800-COM-ONE")
        language = request.data.get("language", "hi")
        tenant_id = request.data.get("tenant_id", "879e9bda-8049-4824-9aa0-fc5e9f0012d9")

        session = AsteriskBridge.start_call(
            caller_number=caller_number,
            did_number=did_number,
            tenant_id=tenant_id,
            language=language
        )

        crm_profile = CrmBridgeService.lookup_customer_by_number(caller_number)

        # Initial IVR greeting based on language
        greeting_text = (
            "नमस्ते! हमारी ग्राहक सेवा हेल्पलाइन में आपका स्वागत है। मैं आपकी क्या सहायता कर सकता हूँ?"
            if language == "hi" else
            "Thank you for calling Customer Support. How may I assist you today?"
        )

        from asgiref.sync import async_to_sync
        from apps.providers.services import ProviderBridge

        greeting_audio_b64 = None
        try:
            greeting_audio_bytes = async_to_sync(ProviderBridge.synthesize_speech)(greeting_text)
            if greeting_audio_bytes:
                greeting_audio_b64 = base64.b64encode(greeting_audio_bytes).decode("ascii")
        except Exception as e:
            greeting_audio_b64 = None

        # Eagerly pre-warm Whisper and Embeddings while greeting is playing aloud
        import threading
        threading.Thread(target=ProviderBridge.warmup_models, daemon=True).start()

        return Response({
            "status": "active",
            "call_id": session.call_id,
            "caller_number": session.caller_number,
            "language": session.language,
            "crm_customer_surfaced": crm_profile,
            "initial_greeting": greeting_text,
            "initial_greeting_audio_base64": greeting_audio_b64,
            "started_at": session.started_at.isoformat(),
        }, status=status.HTTP_201_CREATED)

class ProcessTurnView(APIView):
    """
    Execute 4-Stage Cognitive Pipeline on incoming caller turn:
    01. Perceive: Speech-to-Text (Whisper) or raw text
    02. Understand: Intent, Sentiment (-1.0 to 1.0), Emotion Tag, Escalation Check
    03. Respond: pgvector RAG + Gemma 4 LLM Generation + Neural TTS
    04. Act: Autonomous CRM Updates & Agent Handover Packet
    """
    def post(self, request, call_id):
        session = get_object_or_404(CallSession, call_id=call_id)
        if session.status in ["completed", "failed"]:
            return Response({"error": "Call is already closed."}, status=status.HTTP_400_BAD_REQUEST)

        # Handle either audio file upload or text payload
        audio_bytes = None
        if "audio" in request.FILES:
            audio_bytes = request.FILES["audio"].read()
        elif "audio_base64" in request.data:
            audio_bytes = base64.b64decode(request.data["audio_base64"])

        text_input = request.data.get("text")

        if not audio_bytes and not text_input:
            return Response({"error": "Must provide 'audio' file or 'text' query."}, status=status.HTTP_400_BAD_REQUEST)

        # Run async 4-stage pipeline in sync view
        pipeline_output = async_to_sync(CognitiveVoicePipeline.process_call_turn)(
            session=session,
            audio_bytes=audio_bytes,
            text_input=text_input
        )

        return Response(pipeline_output, status=status.HTTP_200_OK)

class EndCallView(APIView):
    """Terminate call, calculate duration, and generate CDR + Post-call CSAT record."""
    def post(self, request, call_id):
        session = get_object_or_404(CallSession, call_id=call_id)
        csat = request.data.get("csat_rating", 5)

        cdr = AsteriskBridge.end_call(session, csat=csat)

        return Response({
            "status": "completed",
            "call_id": session.call_id,
            "duration_seconds": cdr.duration_seconds,
            "disposition": cdr.disposition,
            "csat_rating": cdr.csat_rating,
            "is_escalated": session.is_escalated,
            "assigned_agent": session.assigned_agent,
            "created_at": cdr.created_at.isoformat()
        }, status=status.HTTP_200_OK)

class AgentHandoverView(APIView):
    """Retrieve 5-part auto-surfaced Handover Packet for Human Agent Screen (Deck Page 4)."""
    def get(self, request, call_id):
        session = get_object_or_404(CallSession, call_id=call_id)
        packet = CognitiveVoicePipeline.generate_agent_handover_packet(
            session=session,
            flagged_reason=session.escalation_reason or "Escalation requested by supervisor/caller"
        )
        return Response(packet, status=status.HTTP_200_OK)

class SupervisorLiveMonitorView(APIView):
    """Supervisor Live Monitor (Deck Page 7): View all active AI calls with live sentiment & intent."""
    def get(self, request):
        active_sessions = CallSession.objects.filter(status__in=["active", "escalated"]).order_by("-started_at")[:50]
        data = []
        for s in active_sessions:
            crm = CrmBridgeService.lookup_customer_by_number(s.caller_number)
            data.append({
                "call_id": s.call_id,
                "caller_number": s.caller_number,
                "caller_name": crm.get("name", "Unknown"),
                "status": s.status,
                "intent": s.current_intent or "Connecting...",
                "sentiment_score": s.caller_sentiment,
                "emotion": s.caller_emotion,
                "is_escalated": s.is_escalated,
                "escalation_reason": s.escalation_reason,
                "assigned_agent": s.assigned_agent,
                "started_at": s.started_at.isoformat(),
            })
        return Response({
            "total_active_calls": len(data),
            "calls": data
        }, status=status.HTTP_200_OK)
