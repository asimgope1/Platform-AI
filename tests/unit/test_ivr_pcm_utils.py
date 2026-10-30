"""
Unit tests for the IVR telephony PCM utility functions.
No network, no Whisper model, no LLM required.
"""
import io
import struct
import wave

import numpy as np
import pytest

from telephony.pcm_utils import (
    PBX_CHUNK_BYTES,
    PBX_SAMPLE_RATE,
    chunk_pcm,
    is_silence,
    pcm_bytes_to_float32,
    wav_bytes_to_pcm8k,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_pcm_s16le(n_samples: int, value: int = 4000) -> bytes:
    """Create a raw s16le PCM buffer filled with a constant value."""
    return struct.pack(f"<{n_samples}h", *([value] * n_samples))


def _make_wav(n_samples: int, sample_rate: int = PBX_SAMPLE_RATE, value: int = 4000) -> bytes:
    """Create a minimal WAV file with a constant-value PCM payload."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(_make_pcm_s16le(n_samples, value))
    return buf.getvalue()


# ── tests ────────────────────────────────────────────────────────────────────

class TestIssilence:
    def test_silence_zeros(self):
        pcm = b"\x00\x00" * 160
        assert is_silence(pcm) is True

    def test_voice_above_threshold(self):
        # RMS of 4000 ≈ 4000, well above default 100
        pcm = _make_pcm_s16le(160, value=4000)
        assert is_silence(pcm) is False

    def test_very_short_buffer(self):
        assert is_silence(b"\x00") is True


class TestPcmBytesToFloat32:
    def test_shape_and_dtype(self):
        pcm = _make_pcm_s16le(160, value=1000)
        arr = pcm_bytes_to_float32(pcm, source_rate=PBX_SAMPLE_RATE)
        assert arr.dtype == np.float32
        # Resampled from 8k → 16k → should be ~double the samples
        assert arr.shape[0] > 100

    def test_values_in_range(self):
        pcm = _make_pcm_s16le(160, value=32767)
        arr = pcm_bytes_to_float32(pcm, source_rate=PBX_SAMPLE_RATE)
        assert arr.max() <= 1.0 + 1e-4


class TestWavBytesToPcm8k:
    def test_passthrough_8k_wav(self):
        """WAV already at 8 kHz should survive round-trip."""
        wav = _make_wav(n_samples=800, sample_rate=8000, value=2000)
        pcm = wav_bytes_to_pcm8k(wav)
        assert isinstance(pcm, bytes)
        assert len(pcm) > 0
        # Each sample is 2 bytes; we expect ~800 samples worth
        assert len(pcm) == pytest.approx(800 * 2, abs=16)

    def test_downsample_22k_to_8k(self):
        """WAV at 22 050 Hz should be downsampled to 8 kHz."""
        n_samples_22k = 22050  # exactly 1 second at 22 kHz
        wav = _make_wav(n_samples=n_samples_22k, sample_rate=22050, value=3000)
        pcm = wav_bytes_to_pcm8k(wav)
        # 1 second at 8 kHz = 8000 samples × 2 bytes = 16 000 bytes
        assert len(pcm) == pytest.approx(16_000, rel=0.05)

    def test_stereo_wav_produces_mono_pcm(self):
        """Stereo WAV should be mixed down to mono PCM."""
        buf = io.BytesIO()
        n = 400
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(8000)
            stereo = struct.pack(f"<{n * 2}h", *([1000, -1000] * n))
            wf.writeframes(stereo)
        pcm = wav_bytes_to_pcm8k(buf.getvalue())
        # mono → 400 samples × 2 bytes = 800 bytes
        assert len(pcm) == pytest.approx(400 * 2, abs=16)


class TestChunkPcm:
    def test_exact_multiple(self):
        pcm = b"\x01\x02" * (PBX_CHUNK_BYTES // 2 * 3)  # exactly 3 full chunks
        chunks = list(chunk_pcm(pcm))
        assert len(chunks) == 3
        assert all(len(c) == PBX_CHUNK_BYTES for c in chunks)

    def test_partial_last_chunk_is_padded(self):
        # 1.5 chunks worth of data
        pcm = b"\xAA\xBB" * (PBX_CHUNK_BYTES // 2 + PBX_CHUNK_BYTES // 4)
        chunks = list(chunk_pcm(pcm))
        assert len(chunks) == 2
        assert len(chunks[-1]) == PBX_CHUNK_BYTES  # padded to full frame

    def test_empty_input_yields_nothing(self):
        chunks = list(chunk_pcm(b""))
        assert chunks == []

