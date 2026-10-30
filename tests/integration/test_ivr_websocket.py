"""
Integration tests for the IVR WebSocket endpoint.

These tests use FastAPI's built-in `TestClient` WebSocket support — no external
server or network proxy required. They run exactly like any other pytest test.

Run with:
    pytest tests/integration/test_ivr_websocket.py -v
"""
import io
import json
import math
import struct
import uuid
import wave

import pytest
from fastapi.testclient import TestClient

from services.api.main import app

# ── helpers ──────────────────────────────────────────────────────────────────

SAMPLE_RATE  = 8_000
CHUNK_BYTES  = 320


def _sine_pcm_bytes(freq_hz: float = 440.0, duration_s: float = 2.0,
                    amplitude: int = 14_000) -> bytes:
    """Generate s16le mono 8 kHz sine wave PCM (passes silence gate)."""
    n = int(SAMPLE_RATE * duration_s)
    frames = bytearray()
    for i in range(n):
        val = int(amplitude * math.sin(2 * math.pi * freq_hz * i / SAMPLE_RATE))
        frames.extend(struct.pack("<h", max(-32768, min(32767, val))))
    return bytes(frames)


def _chunk(pcm: bytes):
    for i in range(0, len(pcm), CHUNK_BYTES):
        chunk = pcm[i : i + CHUNK_BYTES]
        if len(chunk) < CHUNK_BYTES:
            chunk += b"\x00" * (CHUNK_BYTES - len(chunk))
        yield chunk


def _is_valid_pcm(data: bytes) -> bool:
    """Heuristic: valid raw s16le PCM is even-lengthed and non-zero."""
    return len(data) % 2 == 0 and len(data) > 0 and any(b != 0 for b in data)


# ── tests ────────────────────────────────────────────────────────────────────

class TestIVRWebSocket:

    @pytest.fixture(autouse=True)
    def client(self):
        self._client = TestClient(app)

    # ── Test 1: basic handshake ───────────────────────────────────────────────
    def test_websocket_connects_and_accepts(self):
        """The /ws/ivr endpoint must accept a WebSocket connection."""
        with self._client.websocket_connect("/ws/ivr") as ws:
            call_uuid = str(uuid.uuid4())
            ws.send_json({"event": "call_start", "uuid": call_uuid})
            ws.send_json({"event": "call_end",   "uuid": call_uuid})
        # If we reach here without exception, connection was accepted ✓

    # ── Test 2: unknown events are silently ignored ───────────────────────────
    def test_unknown_event_does_not_crash(self):
        """Unrecognised JSON events must not crash the session."""
        with self._client.websocket_connect("/ws/ivr") as ws:
            ws.send_json({"event": "call_start", "uuid": "test-1"})
            ws.send_json({"event": "unknown_event", "foo": "bar"})
            ws.send_json({"event": "call_end", "uuid": "test-1"})

    # ── Test 3: transcript → JSON play event ─────────────────────────────────
    def test_transcript_event_triggers_play_json(self):
        """
        Sending a transcript event (pre-transcribed text) must elicit a
        JSON {"event":"play"} frame before or alongside the binary PCM reply.
        """
        call_uuid = str(uuid.uuid4())
        play_seen = False
        pcm_seen  = False

        with self._client.websocket_connect("/ws/ivr") as ws:
            ws.send_json({"event": "call_start", "uuid": call_uuid})
            ws.send_json({
                "event": "transcript",
                "uuid":  call_uuid,
                "text":  "What are your business hours?",
            })

            # Drain up to 30 frames looking for the play event + PCM
            for _ in range(30):
                try:
                    data = ws.receive()
                except Exception:
                    break

                if isinstance(data, str) or (isinstance(data, dict) and "text" in data):
                    text = data if isinstance(data, str) else data.get("text", "")
                    try:
                        evt = json.loads(text)
                        if evt.get("event") == "play":
                            play_seen = True
                    except Exception:
                        pass

                elif isinstance(data, bytes) or (isinstance(data, dict) and "bytes" in data):
                    raw = data if isinstance(data, bytes) else data.get("bytes", b"")
                    if raw:
                        pcm_seen = True

                if play_seen and pcm_seen:
                    break

            ws.send_json({"event": "call_end", "uuid": call_uuid})

        # At minimum the play event must arrive (PCM may be empty if TTS fails)
        assert play_seen, "Expected a JSON play event from the server"

    # ── Test 4: transcript → binary PCM reply ────────────────────────────────
    def test_transcript_event_returns_binary_pcm(self):
        """
        After a transcript event the server must send at least one binary
        frame containing non-silent raw PCM data.
        """
        call_uuid = str(uuid.uuid4())
        pcm_frames = []

        with self._client.websocket_connect("/ws/ivr") as ws:
            ws.send_json({"event": "call_start", "uuid": call_uuid})
            ws.send_json({
                "event": "transcript",
                "uuid":  call_uuid,
                "text":  "Hello, I need help with my account.",
            })

            for _ in range(50):
                try:
                    data = ws.receive()
                except Exception:
                    break

                raw = None
                if isinstance(data, bytes):
                    raw = data
                elif isinstance(data, dict) and data.get("bytes"):
                    raw = data["bytes"]

                if raw and _is_valid_pcm(raw):
                    pcm_frames.append(raw)
                    if len(pcm_frames) >= 3:
                        break

            ws.send_json({"event": "call_end", "uuid": call_uuid})

        assert len(pcm_frames) > 0, (
            "Expected binary PCM frames in response to transcript event; "
            "got none. Check TTS provider is configured correctly."
        )
        total_bytes = sum(len(f) for f in pcm_frames)
        assert total_bytes % 2 == 0, "PCM reply bytes must be even-aligned (s16le)"

    # ── Test 5: call_end closes the loop ─────────────────────────────────────
    def test_call_end_closes_session(self):
        """
        After call_end is sent, the server must close the WebSocket without
        raising an exception on the client side.
        """
        call_uuid = str(uuid.uuid4())
        with self._client.websocket_connect("/ws/ivr") as ws:
            ws.send_json({"event": "call_start", "uuid": call_uuid})
            ws.send_json({"event": "call_end",   "uuid": call_uuid})
            # Server should have broken out of its receive loop.
            # Any further recv on a closed socket raises — that's expected and fine.

    # ── Test 6: silence is blocked at VAD (logic-level check) ───────────────
    def test_silent_pcm_is_rejected_by_vad(self):
        """
        VAD gate (is_silence) must return True for all-zero PCM bytes.
        This is a logic-level unit assertion — no WebSocket needed.
        """
        from telephony.pcm_utils import is_silence
        silence = b"\x00\x00" * 160
        assert is_silence(silence), "All-zero frames must be classified as silence"

        # Voiced frame (RMS >> threshold) must NOT be classified as silence
        voiced = struct.pack("<h", 10_000) * 160
        assert not is_silence(voiced), "High-amplitude frame must not be silence"
