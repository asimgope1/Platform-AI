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
_ODIA_VITS_MODEL = None
_ODIA_VITS_TOKENIZER = None

def _get_odia_vits():
    global _ODIA_VITS_MODEL, _ODIA_VITS_TOKENIZER
    if _ODIA_VITS_MODEL is None:
        try:
            import torch
            from transformers import VitsModel, AutoTokenizer
            model_id = "facebook/mms-tts-ory"
            _ODIA_VITS_TOKENIZER = AutoTokenizer.from_pretrained(model_id)
            _ODIA_VITS_MODEL = VitsModel.from_pretrained(model_id)
            _ODIA_VITS_MODEL.eval()
        except Exception:
            _ODIA_VITS_MODEL = False
    return _ODIA_VITS_MODEL, _ODIA_VITS_TOKENIZER

class MockTextToSpeechAdapter(BaseTextToSpeechAdapter):
    """Produces native Odia VITS neural speech (facebook/mms-tts-ory) and Edge-TTS voices with caching."""

    @classmethod
    def warmup(cls):
        """Pre-warm native Odia VITS model so first turn is instant."""
        try:
            _get_odia_vits()
        except Exception:
            pass

    def _synthesize_native_odia_vits_sync(self, text: str) -> Optional[bytes]:
        """Synthesizes authentic native Odia speech directly from Odia Unicode text."""
        model, tokenizer = _get_odia_vits()
        if not model or not tokenizer:
            return None
        try:
            import re
            import torch
            import numpy as np

            # Expand numbers/digits to spoken native Odia words so the acoustic model pronounces them
            odia_digits = {
                '0': ' ଶୂନ ', '1': ' ଏକ ', '2': ' ଦୁଇ ', '3': ' ତିନି ', '4': ' ଚାରି ',
                '5': ' ପାଞ୍ଚ ', '6': ' ଛଅ ', '7': ' ସାତ ', '8': ' ଆଠ ', '9': ' ନଅ ',
                '୦': ' ଶୂନ ', '୧': ' ଏକ ', '୨': ' ଦୁଇ ', '୩': ' ତିନି ', '୪': ' ଚାରି ',
                '୫': ' ପାଞ୍ଚ ', '୬': ' ଛଅ ', '୭': ' ସାତ ', '୮': ' ଆଠ ', '୯': ' ନଅ ',
            }
            clean_text = text
            for d, w in odia_digits.items():
                clean_text = clean_text.replace(d, w)
            clean_text = re.sub(r"\s+", " ", clean_text).strip()

            inputs = tokenizer(clean_text, return_tensors="pt")
            with torch.no_grad():
                output = model(**inputs).waveform
            arr = (output.squeeze().cpu().numpy() * 32767).clip(-32768, 32767).astype(np.int16)
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(model.config.sampling_rate)
                wf.writeframes(arr.tobytes())
            wav_bytes = buf.getvalue()
            if wav_bytes and len(wav_bytes) > 100:
                return wav_bytes
        except Exception:
            return None
        return None

    async def _synthesize_edge_tts(self, text: str, voice: str, rate: str = "+0%") -> Optional[bytes]:
        try:
            import edge_tts
            communicate = edge_tts.Communicate(text, voice, rate=rate)
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
                ["say", "-v", voice, "-r", "165", "-o", out_path, "--data-format=LEI16@22050", text],
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
        """Transliterates Odia (ଓଡ଼ିଆ) Unicode script to natural Brahmic phonetic equivalent for fallback neural TTS engines."""
        special = {
            'ଆଜ୍ଞା': 'आग्याँ',
            'କ’ଣ': 'कण',
            'କ\'ଣ': 'कण',
            'କଣ': 'कण',
            'ହୁଅନ୍ତୁନି': 'हुअन्तुनि',
            'କରୁଛି': 'करुछि',
            'ଶୁଭୁନି': 'शुभुनि',
            'ଟିକେ': 'टिके',
            'ପହଞ୍ଚିଯିବ': 'पहुँचियिब',
            'ଘଣ୍ଟାଏ': 'घण्टाए',
            'ଭିତରେ': 'भितरे',
            'ୟୁପିଆଇ': 'यूपीआई',
            'ୟୁଟିଆର୍': 'यूटीआर',
            'କେୱାଇସି': 'केवाईसी',
            'ପ୍ୟାନ୍': 'पैन',
            'ଆଧାର': 'आधार',
            'ବାବୁ': 'बाबू',
            'ୟ': 'य', 'ଯ': 'ज', 'ୱ': 'व', 'ଳ': 'ल', 'ଡ଼': 'ड़', 'ଢ଼': 'ढ़',
            'ଡ଼': 'ड़', 'ଢ଼': 'ढ़', 'ୠ': 'ॠ', 'ୡ': 'ॡ', 'ଵ': 'व', 'ଁ': 'ँ',
            'ଂ': 'ं', 'ଃ': 'ः'
        }
        res = []
        i = 0
        keys = sorted(special.keys(), key=len, reverse=True)
        while i < len(text):
            matched = False
            for k in keys:
                if text[i:i+len(k)] == k:
                    res.append(special[k])
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

        # Detect language / script
        has_odia = any('\u0b00' <= ch <= '\u0b7f' for ch in clean_text)
        has_hindi = any('\u0900' <= ch <= '\u097f' for ch in clean_text)
        has_tamil = any('\u0b80' <= ch <= '\u0bff' for ch in clean_text)
        has_telugu = any('\u0c00' <= ch <= '\u0c7f' for ch in clean_text)
        has_kannada = any('\u0c80' <= ch <= '\u0cff' for ch in clean_text)

        # Primary for pure Odia text: Native Odia neural VITS model (facebook/mms-tts-ory)
        is_pure_odia = has_odia and not any('a' <= ch.lower() <= 'z' for ch in clean_text)
        if is_pure_odia:
            try:
                native_odia_audio = await asyncio.to_thread(self._synthesize_native_odia_vits_sync, clean_text)
                if native_odia_audio and len(native_odia_audio) > 500:
                    if len(_TTS_CACHE) < 250:
                        _TTS_CACHE[cache_key] = native_odia_audio
                    return native_odia_audio
            except Exception:
                pass

        has_indian_context = has_hindi or has_odia or any(w in clean_text.lower() for w in ("namaste", "dabayein", "dabantu", "paayein", "odia", "hindi"))
        spoken_text = self._odia_to_phonetic(clean_text) if has_odia else clean_text
        neural_voice = "hi-IN-SwaraNeural" if has_indian_context else "en-IN-NeerjaNeural"
        speech_rate = "-6%"

        # Secondary: Ultra-realistic neural speech
        try:
            edge_audio = await asyncio.wait_for(
                self._synthesize_edge_tts(spoken_text, neural_voice, rate=speech_rate),
                timeout=7.0
            )
            if edge_audio:
                if len(_TTS_CACHE) < 250:
                    _TTS_CACHE[cache_key] = edge_audio
                return edge_audio
        except Exception:
            pass

        # Secondary Fallback: native macOS system voice for respective Indian languages
        if has_tamil:
            say_voice = "Vani"
        elif has_telugu:
            say_voice = "Geeta"
        elif has_kannada:
            say_voice = "Soumya"
        elif has_hindi or has_odia:
            say_voice = "Lekha"
        else:
            say_voice = "Tara"

        say_audio = await asyncio.to_thread(self._synthesize_say_sync, spoken_text, say_voice)
        if not say_audio:
            # Fallback to Rishi or Samantha
            say_audio = await asyncio.to_thread(self._synthesize_say_sync, spoken_text, "Rishi") or await asyncio.to_thread(self._synthesize_say_sync, spoken_text, "Samantha")
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
