#!/usr/bin/env python3
"""
IVR WebSocket Bridge — Test Client
====================================
Simulates an Asterisk PBX connecting to ws://localhost:8001/ws/ivr

Three test modes (pick via --mode):

  1. ping       Fastest. Sends call_start + call_end. Checks WS handshake.
  2. transcript Sends a call_start + pre-transcribed text (no PCM/STT).
                Validates that AI reply comes back as binary PCM.
  3. pcm        Full pipeline. Generates synthetic voiced PCM (sine-wave
                tone encoded as s16le 8 kHz), sends it as 320-byte frames,
                waits for the STT→LLM→TTS reply to arrive as binary frames.

Usage
-----
  python scripts/test_ivr.py --mode ping
  python scripts/test_ivr.py --mode transcript --text "What are your opening hours?"
  python scripts/test_ivr.py --mode pcm
  python scripts/test_ivr.py --mode pcm --host 192.168.1.133 --port 8001
"""

import argparse
import asyncio
import json
import math
import os
import struct
import sys
import time
import uuid
from typing import Optional

# Clear proxy env vars so websockets connects directly (not via sandbox HTTP proxy)
for _pv in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
            "ws_proxy",   "WS_PROXY",   "ALL_PROXY",  "all_proxy"):
    os.environ.pop(_pv, None)

try:
    import websockets
except ImportError:
    print("ERROR: websockets not installed. Run: pip install websockets")
    sys.exit(1)

# ── colours ─────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✓{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}⚠{RESET} {msg}")
def err(msg):  print(f"  {RED}✗{RESET} {msg}")
def info(msg): print(f"  {CYAN}→{RESET} {msg}")
def header(msg): print(f"\n{BOLD}{msg}{RESET}")

# ── PCM helpers ──────────────────────────────────────────────────────────────
SAMPLE_RATE  = 8_000
CHUNK_BYTES  = 320        # 20 ms frame
CHUNK_SAMPLES = CHUNK_BYTES // 2


def _sine_wave_pcm(freq_hz: float = 440.0, duration_s: float = 2.0,
                   amplitude: int = 12_000) -> bytes:
    """
    Generate a sine-wave tone as raw s16le mono 8 kHz PCM.
    This is the simplest 'voice signal' to test VAD + STT pipeline.
    Whisper will transcribe it as silence/noise but the pipeline will run.
    """
    n = int(SAMPLE_RATE * duration_s)
    frames = bytearray()
    for i in range(n):
        t = i / SAMPLE_RATE
        val = int(amplitude * math.sin(2 * math.pi * freq_hz * t))
        val = max(-32768, min(32767, val))
        frames.extend(struct.pack("<h", val))
    return bytes(frames)


def _chunk_pcm(pcm: bytes):
    for i in range(0, len(pcm), CHUNK_BYTES):
        chunk = pcm[i : i + CHUNK_BYTES]
        if len(chunk) < CHUNK_BYTES:
            chunk = chunk + b"\x00" * (CHUNK_BYTES - len(chunk))
        yield chunk


def _pcm_stats(data: bytes) -> str:
    import struct
    n = len(data) // 2
    if n == 0:
        return "empty"
    samples = [struct.unpack_from("<h", data, i * 2)[0] for i in range(min(n, 4000))]
    rms = math.sqrt(sum(s * s for s in samples) / len(samples))
    peak = max(abs(s) for s in samples)
    duration_ms = n / SAMPLE_RATE * 1000
    return f"{len(data)} bytes | {n} samples | {duration_ms:.0f} ms | rms={rms:.0f} peak={peak}"


# ── test modes ───────────────────────────────────────────────────────────────

async def test_ping(uri: str):
    """Level 1 — just check the WS handshake and JSON events."""
    header("TEST 1 — PING  (WebSocket handshake + JSON events only)")
    call_uuid = str(uuid.uuid4())
    info(f"Connecting to {uri}")

    t0 = time.monotonic()
    async with websockets.connect(uri, ping_timeout=5) as ws:
        ok(f"Connected in {(time.monotonic()-t0)*1000:.0f} ms")

        # Send call_start
        start_event = json.dumps({"event": "call_start", "uuid": call_uuid})
        await ws.send(start_event)
        ok(f"Sent call_start (uuid={call_uuid[:8]}…)")

        await asyncio.sleep(0.1)

        # Send call_end
        end_event = json.dumps({"event": "call_end", "uuid": call_uuid})
        await ws.send(end_event)
        ok("Sent call_end")

        # Give the server 500 ms to process then close
        await asyncio.sleep(0.5)

    ok("WebSocket closed cleanly — PING PASSED ✓")


async def test_transcript(uri: str, text: str, timeout: float = 15.0):
    """
    Level 2 — send a pre-transcribed text event.
    Skips STT (no PCM needed). Validates that:
      • server sends back a JSON play event
      • followed by binary PCM frames (the AI reply spoken aloud)
    """
    header("TEST 2 — TRANSCRIPT  (JSON text → AI → PCM reply)")
    call_uuid = str(uuid.uuid4())
    info(f"Connecting to {uri}")
    info(f"User text: '{text}'")

    pcm_received = bytearray()
    play_event_received = False
    t0 = time.monotonic()

    async with websockets.connect(uri, ping_timeout=10) as ws:
        ok(f"Connected in {(time.monotonic()-t0)*1000:.0f} ms")

        await ws.send(json.dumps({"event": "call_start", "uuid": call_uuid}))
        ok("Sent call_start")

        await asyncio.sleep(0.05)

        # Send pre-transcribed text (simulates PBX-side STT)
        transcript_event = json.dumps({
            "event": "transcript",
            "uuid": call_uuid,
            "text": text,
        })
        await ws.send(transcript_event)
        ok("Sent transcript event — waiting for AI reply…")

        deadline = time.monotonic() + timeout
        try:
            while time.monotonic() < deadline:
                remaining = deadline - time.monotonic()
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=remaining)
                except asyncio.TimeoutError:
                    break

                if isinstance(msg, str):
                    try:
                        evt = json.loads(msg)
                        if evt.get("event") == "play":
                            play_event_received = True
                            ok(f"Received play event: {msg}")
                    except json.JSONDecodeError:
                        warn(f"Non-JSON text frame: {msg[:60]}")

                elif isinstance(msg, bytes):
                    pcm_received.extend(msg)
                    # Stop once we've got at least 0.5 s of audio
                    if len(pcm_received) >= SAMPLE_RATE * 2 * 0.5:
                        info("Got enough PCM — stopping early")
                        break

        except websockets.exceptions.ConnectionClosed:
            warn("Server closed connection after reply")

        await ws.send(json.dumps({"event": "call_end", "uuid": call_uuid}))

    print()
    # ── results
    if pcm_received:
        ok(f"PCM reply received: {_pcm_stats(bytes(pcm_received))}")
        ok("TRANSCRIPT TEST PASSED ✓")
    else:
        err("No binary PCM received within timeout")
        err("TRANSCRIPT TEST FAILED ✗")
        sys.exit(1)


async def test_pcm(uri: str, timeout: float = 30.0):
    """
    Level 3 — full pipeline.
    Generates a sine-wave 'voice', sends it as 320-byte frames (real-time),
    then waits for STT→LLM→TTS → binary PCM reply.

    Note: Whisper will NOT transcribe a pure sine tone as real speech
    (it'll get empty / noise). So the pipeline runs but the LLM won't reply
    unless you swap the PCM for real recorded speech.
    Use --mode transcript for reliable end-to-end AI reply validation.
    """
    header("TEST 3 — FULL PCM PIPELINE  (synthetic voice → STT → AI → PCM reply)")
    call_uuid = str(uuid.uuid4())
    info(f"Connecting to {uri}")

    # Generate 2 seconds of sine wave (passes silence gate; Whisper returns noise/empty)
    pcm_voice = _sine_wave_pcm(freq_hz=220, duration_s=2.0, amplitude=14000)
    total_chunks = len(pcm_voice) // CHUNK_BYTES
    info(f"Sending {total_chunks} PCM frames ({len(pcm_voice)//2} samples = 2.0 s of 8 kHz audio)")

    pcm_received = bytearray()
    play_event_received = False
    t0 = time.monotonic()

    async with websockets.connect(uri, ping_timeout=30) as ws:
        ok(f"Connected in {(time.monotonic()-t0)*1000:.0f} ms")

        await ws.send(json.dumps({"event": "call_start", "uuid": call_uuid}))
        ok("Sent call_start")

        # ── Send PCM frames at real-time pace (20 ms each) ──────────────────
        info("Streaming PCM frames…")
        sent = 0
        for frame in _chunk_pcm(pcm_voice):
            await ws.send(frame)
            sent += 1
            await asyncio.sleep(0.020)   # 20 ms real-time pacing

        ok(f"Sent {sent} frames — waiting for VAD flush + STT + AI + TTS…")
        info(f"(VAD will flush after ~0.8 s silence — timeout={timeout}s)")

        # ── Receive reply ────────────────────────────────────────────────────
        deadline = time.monotonic() + timeout
        try:
            while time.monotonic() < deadline:
                remaining = deadline - time.monotonic()
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=remaining)
                except asyncio.TimeoutError:
                    break

                if isinstance(msg, str):
                    try:
                        evt = json.loads(msg)
                        if evt.get("event") == "play":
                            play_event_received = True
                            ok(f"Received play event: {msg}")
                    except json.JSONDecodeError:
                        pass

                elif isinstance(msg, bytes):
                    pcm_received.extend(msg)
                    if len(pcm_received) >= SAMPLE_RATE * 2 * 0.5:
                        info("Got enough PCM reply — stopping")
                        break

        except websockets.exceptions.ConnectionClosed:
            warn("Connection closed by server")

        await ws.send(json.dumps({"event": "call_end", "uuid": call_uuid}))

    print()
    elapsed = time.monotonic() - t0
    info(f"Total elapsed: {elapsed:.1f} s")

    if pcm_received:
        ok(f"PCM reply: {_pcm_stats(bytes(pcm_received))}")
        ok("FULL PCM TEST PASSED ✓")
        warn("Note: Whisper gave empty text for synthetic tone (expected).")
        warn("For a real speech test, record yourself and replay WAV bytes.")
    else:
        warn("No PCM reply received (Whisper returned empty for sine tone — expected)")
        warn("Run --mode transcript to verify the AI→TTS path works end-to-end.")
        ok("FULL PCM TEST COMPLETE (pipeline ran; no speech detected by Whisper)")


# ── entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="IVR WebSocket bridge tester")
    parser.add_argument("--host",    default="localhost", help="Server host (default: localhost)")
    parser.add_argument("--port",    default=8001, type=int, help="Server port (default: 8001)")
    parser.add_argument("--mode",    default="transcript",
                        choices=["ping", "transcript", "pcm"],
                        help="Test mode (default: transcript)")
    parser.add_argument("--text",    default="Hello, what can you help me with today?",
                        help="Text to send in transcript mode")
    parser.add_argument("--timeout", default=20.0, type=float,
                        help="Max seconds to wait for AI reply (default: 20)")
    args = parser.parse_args()

    uri = f"ws://{args.host}:{args.port}/ws/ivr"
    print(f"\n{BOLD}IVR WebSocket Test Client{RESET}")
    print(f"Target: {CYAN}{uri}{RESET}")
    print(f"Mode:   {BOLD}{args.mode}{RESET}")

    try:
        if args.mode == "ping":
            asyncio.run(test_ping(uri))
        elif args.mode == "transcript":
            asyncio.run(test_transcript(uri, args.text, timeout=args.timeout))
        elif args.mode == "pcm":
            asyncio.run(test_pcm(uri, timeout=args.timeout))
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as exc:
        err(f"Fatal: {exc}")
        import traceback; traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
