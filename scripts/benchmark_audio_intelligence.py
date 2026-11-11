"""
Milestone 1 Benchmark Script — Audio Intelligence Pipeline Evaluation.

Evaluates Original vs. Enhanced PCM across:
  • Synthetic test conditions (clean, soft, fan noise, traffic/60 Hz hum, clipped)
  • Real cached multilingual 8 kHz IVR voice recordings (English, Hindi, Odia)

Usage:
    python scripts/benchmark_audio_intelligence.py
    python scripts/benchmark_audio_intelligence.py --with-whisper
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import wave
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Ensure repo root on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from audio_intelligence import AudioIntelligencePipeline
from telephony.ivr_audio_cache import get_cached_prompt_pcm
from telephony.ivr_prompts import get_service_ack_prompt


SAMPLE_RATE = 8000


def _synth_speech(duration_s: float = 1.2, amplitude: float = 7000.0, f0_hz: float = 170.0) -> bytes:
    n = int(SAMPLE_RATE * duration_s)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    env = 0.55 + 0.45 * np.sin(2.0 * np.pi * 4.0 * t)
    sig = (
        0.60 * np.sin(2.0 * np.pi * f0_hz * t)
        + 0.28 * np.sin(2.0 * np.pi * (2.0 * f0_hz) * t)
        + 0.12 * np.sin(2.0 * np.pi * (3.0 * f0_hz) * t)
    ) * env * amplitude
    pad = np.zeros(int(SAMPLE_RATE * 0.25), dtype=np.float32)
    full = np.concatenate([pad, sig, pad])
    return np.clip(full, -32767.0, 32767.0).astype("<i2").tobytes()


def _add_noise_and_hum(pcm: bytes, noise_std: float = 900.0, hum_amp: float = 0.0, seed: int = 7) -> bytes:
    rng = np.random.default_rng(seed)
    arr = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    t = np.arange(len(arr), dtype=np.float32) / SAMPLE_RATE
    noise = rng.normal(0.0, noise_std, size=len(arr))
    hum = hum_amp * np.sin(2.0 * np.pi * 60.0 * t) if hum_amp > 0 else 0.0
    mixed = np.clip(arr + noise + hum, -32767.0, 32767.0)
    return mixed.astype("<i2").tobytes()


def _pcm_to_wav(pcm: bytes, sample_rate: int = 8000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def collect_cases() -> List[Tuple[str, str, bytes]]:
    cases: List[Tuple[str, str, bytes]] = []

    clean = _synth_speech(amplitude=7500.0)
    cases.append(("synth_clean", "en", clean))
    cases.append(("synth_soft_voice", "en", _synth_speech(amplitude=380.0)))
    cases.append(("synth_fan_noise_only", "en", _add_noise_and_hum(b"\x00\x00" * (SAMPLE_RATE * 1), noise_std=450.0)))
    cases.append(("synth_speech_plus_fan", "en", _add_noise_and_hum(clean, noise_std=950.0)))
    cases.append(("synth_speech_plus_60hz_hum", "en", _add_noise_and_hum(clean, noise_std=400.0, hum_amp=2200.0)))
    cases.append(("synth_clipped", "en", _synth_speech(amplitude=60000.0)))

    for lang in ("en", "hi", "or"):
        prompt_text = get_service_ack_prompt(lang, "upi_payment")
        cached = get_cached_prompt_pcm(prompt_text)
        if cached:
            # Cap clip at 3.5 seconds for crisp turn-level benchmarking
            clip = cached[: SAMPLE_RATE * 2 * 4]
            cases.append((f"cached_{lang}_clean", lang, clip))
            cases.append((f"cached_{lang}_noisy", lang, _add_noise_and_hum(clip, noise_std=1100.0, hum_amp=900.0)))

    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Milestone 1 Audio Intelligence Pipeline")
    parser.add_argument("--with-whisper", action="store_true", help="Also run Whisper STT on Original vs Enhanced PCM")
    args = parser.parse_args()

    pipeline = AudioIntelligencePipeline(sample_rate=SAMPLE_RATE, enabled=True, noise_suppression_enabled=True)
    cases = collect_cases()

    whisper = None
    if args.with_whisper:
        from providers.speech_to_text.whisper_local import WhisperLocalSpeechToTextAdapter

        whisper = WhisperLocalSpeechToTextAdapter("benchmark-whisper", {"model": os.getenv("WHISPER_MODEL", "openai/whisper-base")})

    header = (
        f"{'Case':<28} | {'Lang':<4} | {'Orig SNR':>8} | {'Enh SNR':>8} | "
        f"{'Score':>5} | {'Speech?':<7} | {'Latency':>8} | {'Applied'}"
    )
    print("\n=== Milestone 1: Audio Intelligence Benchmark ===")
    print(header)
    print("-" * len(header))

    for name, lang, pcm in cases:
        orig_q = pipeline.analyze_sync(pcm)
        result = pipeline.enhance_sync(pcm, quality=orig_q)
        enh_q = pipeline.analyze_sync(result.enhanced_pcm)

        applied_str = ",".join(result.enhancement_applied) if result.enhancement_applied else ("bypassed" if result.bypassed else "passthrough")
        print(
            f"{name:<28} | {lang:<4} | {orig_q.snr_db:>6.1f}dB | {enh_q.snr_db:>6.1f}dB | "
            f"{orig_q.audio_quality_score:>5.2f} | {str(orig_q.is_speech_present):<7} | "
            f"{result.processing_ms:>6.2f}ms | {applied_str}"
        )

        if whisper is not None and orig_q.is_speech_present:
            orig_tx = whisper._transcribe_sync(_pcm_to_wav(result.original_pcm), language=lang).get("text", "")
            enh_tx = whisper._transcribe_sync(_pcm_to_wav(result.segmented_pcm), language=lang).get("text", "")
            print(f"    Whisper Orig: {orig_tx}")
            print(f"    Whisper Enh : {enh_tx}")


if __name__ == "__main__":
    main()

