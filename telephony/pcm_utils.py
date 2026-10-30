"""
PCM audio codec utilities for the IVR WebSocket bridge.

Asterisk / PBX wire format:  signed 16-bit little-endian, mono, 8 000 Hz
Whisper needs:               float32, mono, 16 000 Hz
TTS output (WAV/MP3/bytes):  must be converted back to s16le 8 kHz before
                             sending binary frames on the WebSocket.
"""
from __future__ import annotations

import io
import struct
import wave
import math
import asyncio
import logging
import subprocess
import tempfile
import os
from typing import Generator

import numpy as np

logger = logging.getLogger(__name__)

# Asterisk PCM spec
PBX_SAMPLE_RATE = 8_000
PBX_CHANNELS = 1
PBX_SAMPLE_WIDTH = 2          # bytes (s16le)
PBX_CHUNK_BYTES = 320          # 20 ms frame = 160 samples × 2 bytes
PBX_CHUNK_SAMPLES = PBX_CHUNK_BYTES // PBX_SAMPLE_WIDTH  # 160

# Whisper's native sample rate
WHISPER_SAMPLE_RATE = 16_000


# ──────────────────────────────────────────────────────────────────────────────
# Resampling and Smoothing Utilities
# ──────────────────────────────────────────────────────────────────────────────

def resample_audio(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """
    Resample audio array using high-quality polyphase FIR anti-aliasing filter.
    Prevents Gibbs ringing, boundary clicks, and high-frequency spectral aliasing.
    """
    if source_rate == target_rate or len(samples) == 0:
        return samples
    import scipy.signal
    from math import gcd
    g = gcd(source_rate, target_rate)
    up = target_rate // g
    down = source_rate // g
    try:
        return scipy.signal.resample_poly(samples, up, down).astype(np.float32)
    except Exception as exc:
        logger.warning(f"resample_poly failed ({exc}); falling back to linear/fft resample")
        target_len = int(len(samples) * target_rate / source_rate)
        return scipy.signal.resample(samples, target_len).astype(np.float32)


def apply_smooth_fades(samples: np.ndarray, fade_samples: int = 80) -> np.ndarray:
    """
    Apply smooth Hann raised-cosine fade-in and fade-out to prevent boundary clicks/pops.
    Default 80 samples = 10 ms at 8 kHz.
    """
    n = len(samples)
    if n == 0:
        return samples
    actual_fade = min(fade_samples, n // 4)
    if actual_fade > 0:
        hann_curve = 0.5 * (1.0 - np.cos(np.pi * np.arange(actual_fade) / actual_fade))
        samples[:actual_fade] *= hann_curve
        samples[-actual_fade:] *= hann_curve[::-1]
    return samples


def crossfade_concat_pcm(segments: list[bytes], silence_gap_ms: int = 250) -> bytes:
    """
    Concatenate multiple 8 kHz s16le PCM segments with smooth fade ramps
    and clean comfort silence gaps to eliminate clicks between phrases.
    """
    if not segments:
        return b""
    gap_samples = int(PBX_SAMPLE_RATE * silence_gap_ms / 1000)
    gap_bytes = b"\x00\x00" * gap_samples

    cleaned_segments = []
    for seg in segments:
        if not seg or len(seg) < 4:
            continue
        arr = np.frombuffer(seg, dtype="<i2").astype(np.float32)
        arr = apply_smooth_fades(arr, fade_samples=80)
        arr = np.clip(arr, -32767.0, 32767.0)
        cleaned_segments.append(arr.astype("<i2").tobytes())

    return gap_bytes.join(cleaned_segments)


# ──────────────────────────────────────────────────────────────────────────────
# Raw PCM → float32 array (for Whisper)
# ──────────────────────────────────────────────────────────────────────────────

def pcm_bytes_to_float32(pcm: bytes, source_rate: int = PBX_SAMPLE_RATE) -> np.ndarray:
    """
    Convert raw s16le PCM bytes → float32 numpy array resampled to 16 kHz.

    Parameters
    ----------
    pcm:         Raw bytes from the WebSocket binary frame(s).
    source_rate: Sample rate of the incoming bytes (default 8 000 Hz).

    Returns
    -------
    float32 numpy array at 16 000 Hz, values in [-1, 1].
    """
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0

    # Resample to 16 kHz for Whisper using polyphase filter
    if source_rate != WHISPER_SAMPLE_RATE and len(samples) > 0:
        samples = resample_audio(samples, source_rate, WHISPER_SAMPLE_RATE)

    return np.clip(samples, -1.0, 1.0)


def is_silence(pcm: bytes, rms_threshold: float = 100.0) -> bool:
    """
    Quick silence gate on raw s16le bytes.
    Keeps CPU usage near-zero for muted/hold channels.
    """
    if len(pcm) < PBX_SAMPLE_WIDTH * 2:
        return True
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    rms = float(np.sqrt(np.mean(samples ** 2)))
    return rms < rms_threshold


# ──────────────────────────────────────────────────────────────────────────────
# WAV / MP3 / generic audio → raw s16le PCM at 8 kHz (for sending back)
# ──────────────────────────────────────────────────────────────────────────────

def wav_bytes_to_pcm8k(wav_bytes: bytes) -> bytes:
    """
    Convert a WAV file (any rate/channels) to raw s16le mono 8 kHz PCM bytes.
    Features:
      - Polyphase anti-aliasing resampling
      - DC offset removal
      - Safe -1.5 dBFS normalization headroom (peak 27,500) to prevent DAC distortion
      - Boundary Hann fade-in and fade-out to eliminate start/end clicks
      - Strict clipping to prevent signed int16 overflow/wrap glitches
    """
    try:
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            raw = wf.readframes(wf.getnframes())

        # Parse samples
        dtype = "<i2" if sampwidth == 2 else "<i4"
        samples = np.frombuffer(raw, dtype=dtype).astype(np.float32)

        # Stereo → mono
        if n_channels == 2:
            samples = samples.reshape(-1, 2).mean(axis=1)
        elif n_channels > 2:
            samples = samples.reshape(-1, n_channels).mean(axis=1)

        # Resample to 8 kHz first (using polyphase anti-aliasing filter)
        if framerate != PBX_SAMPLE_RATE:
            samples = resample_audio(samples, framerate, PBX_SAMPLE_RATE)

        # Remove DC offset
        samples = samples - np.mean(samples)

        # Apply smooth fade ramps to start and end (10ms = 80 samples)
        samples = apply_smooth_fades(samples, fade_samples=80)

        # Normalize to -1.5 dBFS (~27,500) for clean telephone playback without DAC clipping
        max_val = float(np.max(np.abs(samples)))
        if max_val > 1e-4:
            target_peak = 27500.0
            gain = min(target_peak / max_val, 10.0)
            samples = samples * gain

        # Strict clipping to [-32767.0, 32767.0] to guarantee ZERO integer overflow wrapping
        samples = np.clip(samples, -32767.0, 32767.0)

        return samples.astype("<i2").tobytes()

    except Exception as exc:
        logger.warning(f"wav_bytes_to_pcm8k failed ({exc}); using afconvert fallback")
        return _afconvert_to_pcm8k(wav_bytes, ext=".wav")


def audio_bytes_to_pcm8k(audio_bytes: bytes, ext: str = ".mp3") -> bytes:
    """
    Convert arbitrary audio bytes (mp3 / m4a / wav / …) to raw s16le mono 8 kHz
    using macOS `afconvert` → WAV → wav_bytes_to_pcm8k pipeline.
    Falls back to a 0.5-second silence tone on error.
    """
    try:
        # Try wav path first (fastest, no subprocess)
        if ext.lower() in (".wav", ".wave"):
            return wav_bytes_to_pcm8k(audio_bytes)

        return _afconvert_to_pcm8k(audio_bytes, ext=ext)
    except Exception as exc:
        logger.error(f"audio_bytes_to_pcm8k: conversion failed ({exc}); returning silence")
        return _silence_pcm(duration_ms=500)


def _afconvert_to_pcm8k(audio_bytes: bytes, ext: str = ".mp3") -> bytes:
    """macOS afconvert → WAV → numpy → raw PCM 8 kHz."""
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as in_f:
        in_path = in_f.name
        in_f.write(audio_bytes)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out_f:
        out_path = out_f.name
    try:
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16@8000", "-c", "1", in_path, out_path],
            check=True,
            capture_output=True,
            timeout=10,
        )
        with open(out_path, "rb") as f:
            wav_bytes = f.read()
        return wav_bytes_to_pcm8k(wav_bytes)
    finally:
        for p in (in_path, out_path):
            try:
                os.remove(p)
            except OSError:
                pass


def _silence_pcm(duration_ms: int = 500) -> bytes:
    """Return silent raw s16le PCM for the given duration."""
    n_samples = int(PBX_SAMPLE_RATE * duration_ms / 1000)
    return b"\x00\x00" * n_samples


# ──────────────────────────────────────────────────────────────────────────────
# Chunker
# ──────────────────────────────────────────────────────────────────────────────

def chunk_pcm(pcm: bytes, chunk_size: int = PBX_CHUNK_BYTES) -> Generator[bytes, None, None]:
    """
    Yield fixed-size PCM byte chunks (default 320 bytes = 20 ms at 8 kHz).
    Pads the last chunk with zeros if it is shorter than chunk_size.
    """
    for start in range(0, len(pcm), chunk_size):
        frame = pcm[start : start + chunk_size]
        if len(frame) < chunk_size:
            frame = frame + b"\x00" * (chunk_size - len(frame))
        yield frame

