#!/usr/bin/env python3
"""
End-to-End Voice Intelligence Load, Concurrency & Fault-Resilience Benchmark.

Measures per-turn latency (p50, p95, p99), throughput, CPU time, and RSS peak memory
across 1, 5, 10, and 20 concurrent synthetic IVR calls through the real
`IVRCallSession`, `AudioIntelligencePipeline`, `TelephonyIntelligencePipeline`,
and `IVRRAGMiniModel` pipeline, plus fault-injection tests (STT timeout, Gemma/LLM
timeout fallback, and abrupt PBX WebSocket disconnect).

Clearly separates synthetic STT/TTS benchmarks from real local Whisper STT mode
(`--real-stt`). Real SIP/Asterisk trunk concurrency is NOT covered here.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orchestration.chat_orchestrator import ChatOrchestrator
from platform_core.contracts.requests import (
    ChatMessage,
    NormalizedChatResponse,
    NormalizedChoice,
    UsageInfo,
)
from providers.chat.ivr_mini_model import ivr_mini_brain
from telephony.ivr_audio_cache import get_cached_prompt_pcm
from telephony.ivr_menu import IVRState
from telephony.ivr_prompts import HUMAN_TRANSFER
from telephony.ivr_session import IVRCallSession


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return round(ordered[idx], 2)


def _rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports ru_maxrss in bytes; Linux reports KiB
    if sys.platform == "darwin":
        return round(rss / (1024.0 * 1024.0), 2)
    return round(rss / 1024.0, 2)


def _load_sample_pcm() -> bytes:
    cached = get_cached_prompt_pcm(HUMAN_TRANSFER["en"])
    if cached:
        return cached
    sr = 8000
    t = np.linspace(0.0, 1.0, sr, endpoint=False, dtype=np.float32)
    sig = 6000.0 * (0.6 * np.sin(2 * np.pi * 320 * t) + 0.4 * np.sin(2 * np.pi * 680 * t))
    return sig.astype("<i2").tobytes()


class _BenchmarkOrchestrator:
    """Routes turns through IVRRAGMiniModel without network dependency."""

    async def execute(self, request, context=None, enable_rag: bool = True, **kwargs):
        user_msg = request.messages[-1].content if request.messages else ""
        reply_text = ivr_mini_brain.synthesize(
            user_query=user_msg,
            messages=request.messages,
            language="en",
            rag_context="",
        )
        return NormalizedChatResponse(
            id="bench-resp",
            model=request.model_alias,
            created=int(time.time()),
            choices=[
                NormalizedChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=reply_text),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(prompt_tokens=20, completion_tokens=15, total_tokens=35),
        )


async def _run_single_call(
    call_idx: int,
    turns_per_call: int,
    sample_pcm: bytes,
    stt_adapter: Any = None,
    with_db: bool = False,
) -> List[float]:
    ws = MagicMock()
    ws.send_text = AsyncMock()
    ws.send_bytes = AsyncMock()

    audio_svc = MagicMock()
    if stt_adapter is not None:
        async def _real_transcribe(audio_bytes: bytes, filename: str = "call.wav", language: str = "en"):
            return await stt_adapter.transcribe(audio_bytes=audio_bytes, language=language)

        audio_svc.transcribe = _real_transcribe
    else:
        audio_svc.transcribe = AsyncMock(
            return_value={
                "text": "My UPI payment of 5000 rupees failed, please check refund status",
                "language": "en",
                "diagnostics": {"inference_ms": 1.0},
            }
        )
    # Empty audio short-circuits real-time 20ms playback sleep while exercising full intelligence path
    audio_svc.synthesize = AsyncMock(return_value=b"")

    sess = IVRCallSession(
        ws=ws,
        orchestrator=_BenchmarkOrchestrator(),
        audio_service=audio_svc,
        language="en",
    )
    sess.intel_pipeline.persistence_enabled = with_db
    sess.on_call_start({"uuid": f"bench-call-{call_idx}", "caller_id": "9876543210"})
    sess.menu_machine.state = IVRState.AI_CONVERSATION
    sess.selected_language = "en"

    turn_latencies: List[float] = []
    for _ in range(turns_per_call):
        t0 = time.perf_counter()
        await sess._run_stt_ai_tts(sample_pcm)
        turn_latencies.append((time.perf_counter() - t0) * 1000.0)
    if with_db:
        await asyncio.sleep(0.05)  # flush background telemetry queue
    sess.on_call_end({"uuid": f"bench-call-{call_idx}"})
    return turn_latencies


async def benchmark_concurrency(
    concurrency: int,
    turns_per_call: int,
    sample_pcm: bytes,
    stt_adapter: Any = None,
    with_db: bool = False,
) -> Dict[str, Any]:
    cpu_before = os.times()
    wall_t0 = time.perf_counter()
    results = await asyncio.gather(
        *[
            _run_single_call(
                i,
                turns_per_call,
                sample_pcm,
                stt_adapter=stt_adapter,
                with_db=with_db,
            )
            for i in range(concurrency)
        ]
    )
    wall_ms = (time.perf_counter() - wall_t0) * 1000.0
    cpu_after = os.times()
    flat = [lat for call_lats in results for lat in call_lats]
    return {
        "concurrent_calls": concurrency,
        "total_turns": len(flat),
        "wall_ms": round(wall_ms, 2),
        "throughput_turns_per_sec": round(len(flat) / max(0.001, wall_ms / 1000.0), 2),
        "p50_ms": _percentile(flat, 50),
        "p95_ms": _percentile(flat, 95),
        "p99_ms": _percentile(flat, 99),
        "max_ms": round(max(flat), 2) if flat else 0.0,
        "cpu_user_s": round(cpu_after.user - cpu_before.user, 3),
        "cpu_sys_s": round(cpu_after.system - cpu_before.system, 3),
        "rss_peak_mb": _rss_mb(),
    }


async def run_fault_injection_checks(sample_pcm: bytes) -> Dict[str, Any]:
    """Verify resilience under STT timeout, Gemma timeout fallback, and WS disconnect."""
    outcomes: Dict[str, Any] = {}

    # 1. STT Timeout resilience
    ws = MagicMock()
    ws.send_text = AsyncMock()
    ws.send_bytes = AsyncMock()
    audio_timeout = MagicMock()
    audio_timeout.transcribe = AsyncMock(side_effect=asyncio.TimeoutError("Whisper timed out"))
    audio_timeout.synthesize = AsyncMock(return_value=b"")
    s1 = IVRCallSession(ws=ws, orchestrator=_BenchmarkOrchestrator(), audio_service=audio_timeout)
    s1.intel_pipeline.persistence_enabled = False
    s1.on_call_start({"uuid": "fault-stt-timeout"})
    s1.menu_machine.state = IVRState.AI_CONVERSATION
    await s1._run_stt_ai_tts(sample_pcm)
    outcomes["stt_timeout_nonfatal"] = True
    s1.on_call_end({"uuid": "fault-stt-timeout"})

    # 2. Orchestrator / Gemma timeout -> falls back to ivr_mini_brain cleanly
    orch_timeout = MagicMock()
    orch_timeout.execute = AsyncMock(side_effect=asyncio.TimeoutError("Gemma upstream timeout"))
    audio_ok = MagicMock()
    audio_ok.transcribe = AsyncMock(
        return_value={"text": "Where is my UPI refund", "diagnostics": {"inference_ms": 5.0}}
    )
    audio_ok.synthesize = AsyncMock(return_value=b"")
    s2 = IVRCallSession(ws=ws, orchestrator=orch_timeout, audio_service=audio_ok)
    s2.intel_pipeline.persistence_enabled = False
    s2.on_call_start({"uuid": "fault-llm-timeout"})
    s2.menu_machine.state = IVRState.AI_CONVERSATION
    await s2._run_stt_ai_tts(sample_pcm)
    outcomes["llm_timeout_fallback_replied"] = audio_ok.synthesize.await_count >= 1
    s2.on_call_end({"uuid": "fault-llm-timeout"})

    # 3. Abrupt WebSocket disconnect during PCM playback
    ws_dead = MagicMock()
    ws_dead.send_text = AsyncMock(side_effect=ConnectionResetError("PBX socket closed"))
    ws_dead.send_bytes = AsyncMock(side_effect=ConnectionResetError("PBX socket closed"))
    audio_pcm = MagicMock()
    audio_pcm.transcribe = AsyncMock(
        return_value={"text": "Check my debit card", "diagnostics": {"inference_ms": 5.0}}
    )
    # 1600 bytes raw s16le PCM -> _send_pcm_reply will try send_bytes and hit ConnectionResetError
    audio_pcm.synthesize = AsyncMock(return_value=b"\x01\x02" * 800)
    s3 = IVRCallSession(ws=ws_dead, orchestrator=_BenchmarkOrchestrator(), audio_service=audio_pcm)
    s3.intel_pipeline.persistence_enabled = False
    s3.on_call_start({"uuid": "fault-ws-disconnect"})
    s3.menu_machine.state = IVRState.AI_CONVERSATION
    await s3._run_stt_ai_tts(sample_pcm)
    await asyncio.sleep(0.05)
    outcomes["ws_disconnect_handled_gracefully"] = not s3._is_playing
    s3.on_call_end({"uuid": "fault-ws-disconnect"})

    return outcomes


async def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Hybrid Voice Intelligence pipeline.")
    parser.add_argument(
        "--concurrencies",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20],
        help="Concurrent call counts to test.",
    )
    parser.add_argument("--turns-per-call", type=int, default=3)
    parser.add_argument(
        "--real-stt",
        action="store_true",
        help="Use real local Whisper STT adapter instead of synthetic instant STT.",
    )
    parser.add_argument(
        "--with-db",
        action="store_true",
        help="Allow real PostgreSQL RAG + Intelligence persistence writes.",
    )
    args = parser.parse_args()

    sample_pcm = _load_sample_pcm()
    stt_adapter = None
    if args.real_stt:
        from providers.speech_to_text.whisper_local import WhisperLocalSpeechToTextAdapter

        stt_adapter = WhisperLocalSpeechToTextAdapter(
            "bench-whisper",
            {"model": os.getenv("WHISPER_MODEL", "openai/whisper-base")},
        )
        stt_adapter.warmup()

    ctx_mgr = None
    if not args.with_db:
        from contextlib import nullcontext
        from unittest.mock import patch

        class _NoopAsyncSession:
            async def __aenter__(self):
                return MagicMock()

            async def __aexit__(self, *a):
                return False

        ctx_mgr = patch("telephony.ivr_session.async_session", return_value=_NoopAsyncSession())
        search_patch = patch("retrieval.search.SemanticSearchService.search", AsyncMock(return_value=[]))
        ctx_mgr.__enter__()
        search_patch.__enter__()
    else:
        search_patch = None

    try:
        # Warm up first turn to avoid one-time JIT / numpy allocation bias
        await benchmark_concurrency(1, 1, sample_pcm, stt_adapter=stt_adapter, with_db=args.with_db)

        load_results = []
        for c in args.concurrencies:
            res = await benchmark_concurrency(
                c,
                turns_per_call=args.turns_per_call,
                sample_pcm=sample_pcm,
                stt_adapter=stt_adapter,
                with_db=args.with_db,
            )
            load_results.append(res)

        fault_results = await run_fault_injection_checks(sample_pcm)
    finally:
        if search_patch is not None:
            search_patch.__exit__(None, None, None)
        if ctx_mgr is not None:
            ctx_mgr.__exit__(None, None, None)
    report = {
        "mode": "real_local_whisper_stt_synthetic_tts" if args.real_stt else "synthetic_stt_tts_real_audio_conv_decision",
        "turns_per_call": args.turns_per_call,
        "load_benchmarks": load_results,
        "fault_injection": fault_results,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
