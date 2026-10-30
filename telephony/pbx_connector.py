"""
PBX Persistent Connector
========================
Maintains a live WebSocket connection to the Asterisk PBX at
ws://192.168.1.133:9093 (or whatever PBX_WS_URL is set to).

Flow when MizoDroid dials extension 8000:
  MizoDroid (SIP) → Asterisk PBX → sends PCM over this WebSocket
  → IVRCallSession: STT → AI → TTS → sends PCM back on same socket
  → Asterisk plays reply audio to the caller

This module runs as a background asyncio task, started in FastAPI lifespan.
It auto-reconnects on disconnect with exponential backoff.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os

logger = logging.getLogger(__name__)

# ── config ────────────────────────────────────────────────────────────────────
PBX_WS_URL      = os.getenv("PBX_WS_URL",      "ws://192.168.1.133:9093")
IVR_CHAT_MODEL  = os.getenv("IVR_CHAT_MODEL",  "mock-chat")
IVR_LANGUAGE    = os.getenv("IVR_LANGUAGE",    "en")

RECONNECT_MIN   = 2.0    # seconds before first retry
RECONNECT_MAX   = 30.0   # cap on backoff


# ── connector ─────────────────────────────────────────────────────────────────

class PBXConnector:
    """
    Persistent WebSocket client that connects to the Asterisk PBX and
    runs an IVRCallSession for every call received on that socket.
    """

    def __init__(self):
        self._task: asyncio.Task | None = None
        self._running = False

    def start(self):
        """Called from FastAPI lifespan — launches the background loop."""
        self._running = True
        self._task = asyncio.create_task(self._run_loop(), name="pbx-connector")
        logger.info(f"[PBX] Connector started → {PBX_WS_URL}")

    def stop(self):
        """Called on server shutdown."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        logger.info("[PBX] Connector stopped")

    async def _run_loop(self):
        """Outer reconnect loop — keeps trying forever until stop() is called."""
        delay = RECONNECT_MIN
        while self._running:
            try:
                await self._connect_and_serve()
                delay = RECONNECT_MIN          # successful session → reset backoff
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning(f"[PBX] Connection lost ({exc}); retrying in {delay:.0f} s…")
                await asyncio.sleep(delay)
                delay = min(delay * 2, RECONNECT_MAX)

    async def _connect_and_serve(self):
        """Single connection lifetime — connects, reads hello, then handles calls."""
        try:
            import websockets
        except ImportError:
            logger.error("[PBX] 'websockets' package not installed — connector disabled")
            await asyncio.sleep(60)
            return

        logger.info(f"[PBX] Connecting to {PBX_WS_URL} …")
        # Ensure direct connection to PBX; disable websockets keepalive ping so it doesn't drop during long calls
        async with websockets.connect(
            PBX_WS_URL,
            proxy=None,
            ping_interval=None,
            open_timeout=10,
        ) as ws:
            logger.info("[PBX] ✓ Connected to Asterisk PBX")

            # ── Read hello frame ────────────────────────────────────────────
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=5)
                hello = json.loads(raw) if isinstance(raw, str) else {}
                logger.info(f"[PBX] hello: {hello.get('message','?')} | "
                            f"format={hello.get('audio_format','?')} "
                            f"{hello.get('rate','?')}Hz")
            except Exception:
                pass   # some PBX versions skip the hello

            # ── Build session (one session per WebSocket connection) ─────────
            from audio.service import AudioService
            from orchestration.chat_orchestrator import ChatOrchestrator
            from telephony.ivr_session import IVRCallSession
            from telephony.pbx_ws_adapter import PBXWebSocketAdapter

            orchestrator  = ChatOrchestrator()
            audio_service = AudioService()

            # Wrap the raw websockets.WebSocketClientProtocol in a thin adapter
            # so IVRCallSession can call ws.send_text / ws.send_bytes on it.
            adapted_ws = PBXWebSocketAdapter(ws)

            session = IVRCallSession(
                ws=adapted_ws,
                orchestrator=orchestrator,
                audio_service=audio_service,
                chat_model=IVR_CHAT_MODEL,
                language=IVR_LANGUAGE,
            )

            logger.info("[PBX] Waiting for calls … (dial ext 8000 on MizoDroid)")

            # ── Main receive loop ────────────────────────────────────────────
            async for message in ws:
                if isinstance(message, bytes):
                    # Raw PCM from caller
                    await session.on_pcm_frame(message)

                elif isinstance(message, str):
                    try:
                        event = json.loads(message)
                    except json.JSONDecodeError:
                        continue

                    etype = event.get("event", "").lower()
                    logger.debug(f"[PBX] event: {etype}")

                    if etype == "call_start":
                        caller = (
                            event.get("caller")
                            or event.get("caller_id")
                            or event.get("caller_num")
                            or event.get("caller_number")
                            or event.get("from")
                            or event.get("channel")
                            or "Unknown"
                        )
                        logger.info(f"[PBX] 📞 Call started from caller: '{caller}' | uuid={event.get('uuid','?')} | event={event}")
                        session.on_call_start(event)

                    elif etype == "transcript":
                        # Only process complete sentences / final transcripts!
                        # Ignore partial streaming snippets ("final": false)
                        is_final = event.get("final", True)
                        if is_final is not False:
                            await session.on_transcript(event)
                        else:
                            logger.debug(f"[PBX] Partial transcript ignored: {event.get('text')}")

                    elif etype in ("dtmf", "keypad", "dtmf_digit") or "digit" in event or "dtmf" in event:
                        digit = str(
                            event.get("digit")
                            or event.get("dtmf")
                            or event.get("key")
                            or event.get("value")
                            or ""
                        ).strip()
                        if digit:
                            logger.info(f"[PBX] 🔢 DTMF keypad digit received: '{digit}' | uuid={event.get('uuid','?')}")
                            await session.on_dtmf(digit, event)

                    elif etype == "call_end":
                        session.on_call_end(event)
                        logger.info("[PBX] 📵 Call ended")
                        # Reset session for next call on same socket
                        session = IVRCallSession(
                            ws=adapted_ws,
                            orchestrator=orchestrator,
                            audio_service=audio_service,
                            chat_model=IVR_CHAT_MODEL,
                            language=IVR_LANGUAGE,
                        )

                    elif etype == "hello":
                        pass   # re-hello after reconnect

                    else:
                        logger.debug(f"[PBX] Ignored unknown event: {etype}")


# ── module-level singleton ────────────────────────────────────────────────────
pbx_connector = PBXConnector()

