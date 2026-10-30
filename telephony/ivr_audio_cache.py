"""
High-Performance Audio Cache for IVR Menu Prompts.
Eliminates dead-air delays by serving pre-rendered PCM audio from memory/disk in <1ms.
Integrates native Odia VITS (facebook/mms-tts-ory) and neural multilingual speech.
"""
from __future__ import annotations

import os
import hashlib
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "ivr_audio_cache"
_PROMPT_PCM_CACHE: Dict[str, bytes] = {}


def _get_cache_path(text: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"prompt_{h}.pcm"


def get_cached_prompt_pcm(text: str) -> Optional[bytes]:
    """Retrieves 8kHz PCM bytes for a prompt with 0ms latency."""
    clean = text.strip()
    # 1. RAM Cache
    if clean in _PROMPT_PCM_CACHE:
        return _PROMPT_PCM_CACHE[clean]

    # 2. Disk Cache
    disk_path = _get_cache_path(clean)
    if disk_path.exists():
        try:
            pcm = disk_path.read_bytes()
            if pcm and len(pcm) > 100:
                _PROMPT_PCM_CACHE[clean] = pcm
                return pcm
        except Exception as e:
            logger.warning(f"[IVR Cache] Failed reading disk cache: {e}")

    return None


def store_cached_prompt_pcm(text: str, pcm: bytes) -> None:
    """Stores rendered 8kHz PCM bytes in RAM and disk."""
    if not pcm or len(pcm) < 100:
        return
    clean = text.strip()
    _PROMPT_PCM_CACHE[clean] = pcm
    try:
        disk_path = _get_cache_path(clean)
        disk_path.write_bytes(pcm)
    except Exception as e:
        logger.warning(f"[IVR Cache] Failed writing disk cache: {e}")


async def render_composite_multilingual_welcome(adapter) -> bytes:
    """
    Renders the multilingual welcome prompt using authentic native voices:
      1. English intro + 'Press one for English' via en-IN-NeerjaNeural (fallback: Tara/Rishi)
      2. 'हिंदी के लिए दो दबाएं' via hi-IN-SwaraNeural (fallback: Lekha)
      3. 'ଓଡ଼ିଆ ପାଇଁ ତିନି ଦବାନ୍ତୁ' via native facebook/mms-tts-ory VITS model
      4. 'To repeat this menu, press zero' via en-IN-NeerjaNeural (fallback: Tara/Rishi)
    Uses smooth crossfading to ensure zero pops or clicks between language transitions.
    """
    from telephony.ivr_session import _tts_output_to_pcm8k
    from telephony.pcm_utils import crossfade_concat_pcm

    en_intro = "Namaste! Thank you for calling customer support. Please choose your preferred language to continue. Press one for English."
    hi_part = "हिंदी के लिए दो दबाएं।"
    or_part = "ଓଡ଼ିଆ ପାଇଁ ତିନି ଦବାନ୍ତୁ।"
    en_outro = "To repeat this menu, press zero."

    async def _render_voice(text: str, edge_voice: str, say_voice: str) -> bytes:
        try:
            audio = await adapter._synthesize_edge_tts(text, edge_voice, rate="-6%")
            if audio and len(audio) > 500:
                pcm = _tts_output_to_pcm8k(audio)
                if len(pcm) > 1000:
                    return pcm
        except Exception:
            pass
        # Fallback to local system voice
        try:
            audio = adapter._synthesize_say_sync(text, say_voice)
            if audio and len(audio) > 500:
                pcm = _tts_output_to_pcm8k(audio)
                if len(pcm) > 1000:
                    return pcm
        except Exception:
            pass
        return b""

    en_pcm = await _render_voice(en_intro, "en-IN-NeerjaNeural", "Tara")
    hi_pcm = await _render_voice(hi_part, "hi-IN-SwaraNeural", "Lekha")

    # Native Odia VITS
    or_audio = adapter._synthesize_native_odia_vits_sync(or_part)
    or_pcm = _tts_output_to_pcm8k(or_audio) if or_audio else b""

    outro_pcm = await _render_voice(en_outro, "en-IN-NeerjaNeural", "Tara")

    # Strict validation: guarantee all 4 language parts are present and healthy
    if len(en_pcm) < 5000 or len(hi_pcm) < 5000 or len(or_pcm) < 5000 or len(outro_pcm) < 5000:
        raise ValueError(
            f"Composite welcome missing segments: en={len(en_pcm)}, hi={len(hi_pcm)}, "
            f"or={len(or_pcm)}, outro={len(outro_pcm)}"
        )

    return crossfade_concat_pcm([en_pcm, hi_pcm, or_pcm, outro_pcm], silence_gap_ms=350)


async def warm_up_ivr_prompts(audio_service) -> None:
    """
    Background worker that pre-renders any uncached IVR prompts
    so callers never experience network delays during calls.
    """
    from telephony.ivr_prompts import (
        WELCOME_AND_LANGUAGE,
        SERVICE_MENUS,
        SERVICE_ACKS,
        SERVICE_NAMES,
        INVALID_INPUT,
        TIMEOUT_PROMPTS,
        MAX_RETRIES_EXCEEDED,
        HUMAN_TRANSFER,
        get_service_ack_prompt,
    )
    from telephony.ivr_session import _tts_output_to_pcm8k
    from providers.registry import provider_registry

    # 1. First warm up the composite multilingual welcome with native Odia VITS
    tts_adapter = provider_registry.get_tts_adapter("default")
    if tts_adapter and hasattr(tts_adapter, "_synthesize_native_odia_vits_sync"):
        try:
            logger.info("[IVR Cache] Pre-rendering composite multilingual welcome (with native Odia VITS)...")
            comp_pcm = await render_composite_multilingual_welcome(tts_adapter)
            if comp_pcm and len(comp_pcm) > 1000:
                store_cached_prompt_pcm(WELCOME_AND_LANGUAGE, comp_pcm)
                logger.info("[IVR Cache] ✓ Composite multilingual welcome cached with native Odia voice!")
        except Exception as exc:
            logger.warning(f"[IVR Cache] Failed composite welcome rendering: {exc}")

    prompts_to_warm = [
        SERVICE_MENUS["en"],
        SERVICE_MENUS["hi"],
        SERVICE_MENUS["or"],
        HUMAN_TRANSFER["en"],
        HUMAN_TRANSFER["hi"],
        HUMAN_TRANSFER["or"],
        INVALID_INPUT["en"],
        INVALID_INPUT["hi"],
        INVALID_INPUT["or"],
        TIMEOUT_PROMPTS["en"],
        TIMEOUT_PROMPTS["hi"],
        TIMEOUT_PROMPTS["or"],
        MAX_RETRIES_EXCEEDED["en"],
        MAX_RETRIES_EXCEEDED["hi"],
        MAX_RETRIES_EXCEEDED["or"],
    ]

    # Add service acknowledgements
    for s_key in SERVICE_NAMES:
        for lang in ("en", "hi", "or"):
            prompts_to_warm.append(get_service_ack_prompt(lang, s_key))

    cached_count = 0
    generated_count = 0

    for prompt in prompts_to_warm:
        if get_cached_prompt_pcm(prompt) is not None:
            cached_count += 1
            continue

        try:
            logger.info(f"[IVR Cache] Pre-rendering prompt ({len(prompt)} chars): '{prompt[:45]}...'")
            tts_bytes = await audio_service.synthesize(
                text=prompt,
                model="tts-1",
                voice="alloy",
                response_format="wav",
            )
            pcm = _tts_output_to_pcm8k(tts_bytes)
            if pcm:
                store_cached_prompt_pcm(prompt, pcm)
                generated_count += 1
        except Exception as e:
            logger.warning(f"[IVR Cache] Could not pre-render prompt: {e}")

    logger.info(
        f"[IVR Cache] Pre-warming complete: {cached_count} already cached, {generated_count} newly rendered."
    )
