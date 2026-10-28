import os
import asyncio
from typing import Dict, Any, Optional
import httpx
from platform_core.contracts.providers import BaseTextToSpeechAdapter
from platform_core.errors.exceptions import ProviderError


import io
import math
import shutil
import struct
import subprocess
import tempfile
import wave


_TTS_CACHE: Dict[str, bytes] = {}

class MockTextToSpeechAdapter(BaseTextToSpeechAdapter):
    """Produces ultra-natural neural spoken audio (Edge-TTS hi-IN-SwaraNeural / en-IN-NeerjaNeural) with caching and local fallback."""

    async def _synthesize_edge_tts(self, text: str, voice: str) -> Optional[bytes]:
        try:
            import edge_tts
            communicate = edge_tts.Communicate(text, voice)
            chunks = []
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    chunks.append(chunk["data"])
            audio = b"".join(chunks)
            if audio and len(audio) > 100:
                return audio
        except Exception:
            return None
        return None

    def _synthesize_say_sync(self, text: str, voice: str) -> Optional[bytes]:
        if not shutil.which("say"):
            return None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                out_path = tf.name
            subprocess.run(
                ["say", "-v", voice, "-r", "185", "-o", out_path, "--data-format=LEI16@22050", text],
                check=True,
                timeout=5,
            )
            with open(out_path, "rb") as f:
                data = f.read()
            if os.path.exists(out_path):
                os.remove(out_path)
            if data and len(data) > 100:
                return data
        except Exception:
            return None
        return None

    @classmethod
    def _odia_to_phonetic(cls, text: str) -> str:
        """Transliterates Odia (ଓଡ଼ିଆ) Unicode script to Brahmic phonetic equivalent for neural TTS voice engines."""
        special = {
            'ୟ': 'य', 'ଯ': 'ज', 'ୱ': 'व', 'ଳ': 'ल', 'ଡ଼': 'ड़', 'ଢ଼': 'ढ़',
            'ଡ଼': 'ड़', 'ଢ଼': 'ढ़', 'ୠ': 'ॠ', 'ୡ': 'ॡ', 'ଵ': 'व', 'ଁ': 'ँ',
            'ଂ': 'ं', 'ଃ': 'ः'
        }
        res = []
        i = 0
        while i < len(text):
            matched = False
            for k, v in special.items():
                if text[i:i+len(k)] == k:
                    res.append(v)
                    i += len(k)
                    matched = True
                    break
            if matched:
                continue
            cp = ord(text[i])
            if 0x0B00 <= cp <= 0x0B7F:
                res.append(chr(cp - 0x0200))
            else:
                res.append(text[i])
            i += 1
        return ''.join(res)

    async def synthesize(
        self,
        text: str,
        model: str = "tts-1",
        voice: str = "alloy",
        response_format: str = "mp3",
    ) -> bytes:
        clean_text = text.strip()
        cache_key = f"{voice}:{clean_text}"
        if cache_key in _TTS_CACHE:
            return _TTS_CACHE[cache_key]

        # Detect language / script for natural neural voice selection
        has_odia = any('\u0b00' <= ch <= '\u0b7f' for ch in clean_text)
        has_hindi = any('\u0900' <= ch <= '\u097f' for ch in clean_text)
        spoken_text = self._odia_to_phonetic(clean_text) if has_odia else clean_text
        neural_voice = "hi-IN-SwaraNeural" if (has_hindi or has_odia) else "en-IN-NeerjaNeural"

        # Primary: Ultra-realistic neural speech with human breathing and natural intonation
        try:
            edge_audio = await asyncio.wait_for(
                self._synthesize_edge_tts(spoken_text, neural_voice),
                timeout=4.0
            )
            if edge_audio:
                if len(_TTS_CACHE) < 250:
                    _TTS_CACHE[cache_key] = edge_audio
                return edge_audio
        except Exception:
            pass

        # Secondary Fallback: macOS system voice if edge-tts network is slow
        say_voice = "Lekha" if (has_hindi or has_odia) else "Samantha"
        say_audio = await asyncio.to_thread(self._synthesize_say_sync, spoken_text, say_voice)
        if say_audio:
            if len(_TTS_CACHE) < 250:
                _TTS_CACHE[cache_key] = say_audio
            return say_audio

        # Tertiary Fallback: Generate audible RIFF WAVE PCM tone
        buf = io.BytesIO()
        sample_rate = 22050
        duration = min(3.0, max(0.5, len(clean_text) * 0.05))
        n_samples = int(sample_rate * duration)
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            frames = bytearray()
            for i in range(n_samples):
                t = i / sample_rate
                val = int(16000.0 * (math.sin(2.0 * math.pi * 440.0 * t) * 0.6 + math.sin(2.0 * math.pi * 880.0 * t) * 0.4) * math.exp(-t))
                frames.extend(struct.pack("<h", max(-32767, min(32767, val))))
            wf.writeframes(frames)
        return buf.getvalue()


class OpenAITextToSpeechAdapter(BaseTextToSpeechAdapter):
    """Text-to-speech adapter targeting OpenAI-compatible TTS endpoints."""

    def __init__(self, provider_id: str, config: Dict[str, Any]):
        super().__init__(provider_id, config)
        self.base_url = (
            os.getenv(config.get("base_url_env", "HOSTED_AI_BASE_URL"))
            or config.get("base_url")
            or "https://api.openai.com/v1"
        )
        self.api_key = (
            os.getenv(config.get("api_key_env", "HOSTED_AI_API_KEY"))
            or config.get("api_key")
        )
        self.timeout = float(config.get("timeout_seconds", 60.0))

    async def synthesize(
        self,
        text: str,
        model: str = "tts-1",
        voice: str = "alloy",
        response_format: str = "mp3",
    ) -> bytes:
        if not self.api_key:
            mock = MockTextToSpeechAdapter(self.provider_id, self.config)
            return await mock.synthesize(text, model, voice, response_format)

        url = f"{self.base_url.rstrip('/')}/audio/speech"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "input": text,
            "voice": voice,
            "response_format": response_format,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                return resp.content
            except Exception as exc:
                raise ProviderError(self.provider_id, f"TTS upstream error: {exc}")
