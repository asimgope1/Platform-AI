"""
WebSocket route: /ws/ivr

Bridges an Asterisk / FreePBX ARI WebSocket session to the AI platform.

Protocol (inbound from PBX)
───────────────────────────
  Binary frame  → raw PCM (s16le, mono, 8 kHz, usually 320 bytes / 20 ms)
  JSON text     → {"event": "call_start", "uuid": "…"}
               → {"event": "transcript",  "text": "…"}   (optional)
               → {"event": "call_end",    "uuid": "…"}

Protocol (outbound to PBX)
──────────────────────────
  JSON text     → {"event": "play", "uuid": "…"}   (optional header)
  Binary frames → raw PCM s16le mono 8 kHz  (320-byte chunks, 20 ms each)
"""
from __future__ import annotations

import json
import logging
import os

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from audio.service import AudioService
from orchestration.chat_orchestrator import ChatOrchestrator
from telephony.ivr_session import IVRCallSession

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Telephony IVR"])

# Module-level singletons (created once, shared across calls)
_orchestrator: ChatOrchestrator | None = None
_audio_service: AudioService | None = None


def _get_orchestrator() -> ChatOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ChatOrchestrator()
    return _orchestrator


def _get_audio_service() -> AudioService:
    global _audio_service
    if _audio_service is None:
        _audio_service = AudioService()
    return _audio_service


@router.websocket("/ws/ivr")
async def ivr_websocket(ws: WebSocket):
    """
    WebSocket endpoint consumed by Asterisk / FreePBX ARI.

    The PBX dials extension 8000, which proxies audio to this endpoint.
    Each binary frame is 320 bytes of s16le PCM at 8 kHz.
    Text frames carry JSON control events (call_start, call_end, transcript).
    """
    await ws.accept()

    chat_model = os.getenv("IVR_CHAT_MODEL", "mock-chat")
    language   = os.getenv("IVR_LANGUAGE", "en")

    session = IVRCallSession(
        ws=ws,
        orchestrator=_get_orchestrator(),
        audio_service=_get_audio_service(),
        chat_model=chat_model,
        language=language,
    )

    logger.info("[IVR] WebSocket connection accepted")

    try:
        while True:
            message = await ws.receive()

            # ── Binary: raw PCM from caller ──────────────────────────────────
            if "bytes" in message and message["bytes"] is not None:
                await session.on_pcm_frame(message["bytes"])

            # ── Text: JSON control events ────────────────────────────────────
            elif "text" in message and message["text"]:
                try:
                    event = json.loads(message["text"])
                except json.JSONDecodeError:
                    logger.warning(f"[IVR] Non-JSON text frame ignored: {message['text'][:80]}")
                    continue

                event_type = event.get("event", "").lower()

                if event_type == "call_start":
                    session.on_call_start(event)

                elif event_type == "transcript":
                    # PBX already ran STT — skip our Whisper step
                    await session.on_transcript(event)

                elif event_type == "call_end":
                    session.on_call_end(event)
                    break   # close the WebSocket after call ends

                else:
                    logger.debug(f"[IVR] Unknown event type '{event_type}' — ignored")

    except WebSocketDisconnect:
        logger.info(f"[IVR] WebSocket disconnected (call={session.call_uuid})")
    except Exception as exc:
        logger.error(f"[IVR] Unexpected error (call={session.call_uuid}): {exc}", exc_info=True)
    finally:
        try:
            await ws.close()
        except Exception:
            pass
        logger.info(f"[IVR] Session cleaned up for call={session.call_uuid}")

