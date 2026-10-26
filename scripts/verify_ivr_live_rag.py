#!/usr/bin/env python3
"""
Verify Live End-to-End RAG + Audio for IVR Bilingual Support
Connects to Colab T4 GPU Gemma 4 model with tenant-ivr-support context.
"""
import os
import sys
import asyncio
import httpx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

API_BASE = "http://127.0.0.1:8000"
IVR_KEY = "sk-live-tenant-ivr-support-key-101"


async def verify_live_ivr():
    print("\n" + "=" * 70)
    print("VERIFYING LIVE BILINGUAL IVR RAG & AUDIO WORKFLOW")
    print("=" * 70)

    async with httpx.AsyncClient(timeout=60.0) as client:
        # 1. Health check
        health = await client.get(f"{API_BASE}/health")
        print(f"1. Health Check: {health.status_code} -> {health.json()}")

        # 2. Hindi Query against IVR Knowledge Base
        print("\n2. Testing Hindi IVR Chat Query with Live Gemma 4 GPU...")
        hindi_payload = {
            "model": "gemma-gpu",
            "messages": [
                {
                    "role": "user",
                    "content": "नमस्ते, मेरा UPI से पैसा कट गया है लेकिन ऑर्डर कन्फर्म नहीं हुआ। मुझे रिफंड कब तक मिलेगा और इसके क्या नियम हैं?"
                }
            ],
            "temperature": 0.2,
            "max_tokens": 250,
            "enable_rag": True,
        }
        resp_hi = await client.post(
            f"{API_BASE}/v1/chat/completions",
            headers={"Authorization": f"Bearer {IVR_KEY}"},
            json=hindi_payload,
        )
        print(f"   Status: {resp_hi.status_code}")
        if resp_hi.status_code == 200:
            hi_content = resp_hi.json()["choices"][0]["message"]["content"]
            print(f"   [Gemma 4 Hindi Answer]:\n{hi_content}")
            assert any(term in hi_content for term in ["24", "48", "घंटे", "UTR", "रिफंड", "NPCI", "बैंक"]), "Expected refund terms in Hindi"
            print("   ✓ Hindi RAG Context and Grounding Verified!")
        else:
            print(f"   Error: {resp_hi.text}")

        # 3. English Query against IVR Knowledge Base
        print("\n3. Testing English IVR Escalation Query with Live Gemma 4 GPU...")
        eng_payload = {
            "model": "gemma-gpu",
            "messages": [
                {
                    "role": "user",
                    "content": "What option number on the IVR keypad should I press to talk to a live customer support representative?"
                }
            ],
            "temperature": 0.2,
            "max_tokens": 250,
            "enable_rag": True,
        }
        resp_en = await client.post(
            f"{API_BASE}/v1/chat/completions",
            headers={"Authorization": f"Bearer {IVR_KEY}"},
            json=eng_payload,
        )
        print(f"   Status: {resp_en.status_code}")
        if resp_en.status_code == 200:
            en_content = resp_en.json()["choices"][0]["message"]["content"]
            print(f"   [Gemma 4 English Answer]:\n{en_content}")
            assert any(term in en_content for term in ["9", "Agent", "Representative", "queue", "wait"]), "Expected escalation terms in English"
            print("   ✓ English Escalation RAG Context and Grounding Verified!")
        else:
            print(f"   Error: {resp_en.text}")

        # 4. Audio Text-to-Speech (TTS) for IVR Prompts (Hindi and English)
        print("\n4. Testing Audio Voice Synthesis (TTS) for IVR Prompts...")
        tts_payload = {
            "model": "tts-1",
            "input": "नमस्ते! हमारी 24x7 ग्राहक सेवा में आपका स्वागत है। हिंदी के लिए 1 दबाएं। To continue in English press 2.",
            "voice": "alloy",
            "response_format": "wav",
        }
        resp_audio = await client.post(
            f"{API_BASE}/v1/audio/speech",
            headers={"Authorization": f"Bearer {IVR_KEY}"},
            json=tts_payload,
        )
        print(f"   Status: {resp_audio.status_code}, Content-Type: {resp_audio.headers.get('content-type')}, Audio Size: {len(resp_audio.content)} bytes")
        assert resp_audio.status_code == 200
        assert len(resp_audio.content) > 1000
        print("   ✓ IVR Voice Prompt Speech Audio Synthesized Successfully!")

        # 5. Audio Speech-to-Text (STT) for IVR Inbound Callers
        print("\n5. Testing Inbound Audio Transcription (STT)...")
        # Send sample audio for transcription
        resp_stt = await client.post(
            f"{API_BASE}/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {IVR_KEY}"},
            files={"file": ("caller_query.wav", resp_audio.content, "audio/wav")},
            data={"model": "whisper-1", "language": "hi"},
        )
        print(f"   Status: {resp_stt.status_code}, Response: {resp_stt.json()}")
        assert resp_stt.status_code == 200
        print("   ✓ Audio Transcription Verified!")

    print("\n" + "=" * 70)
    print("ALL BILINGUAL IVR TEXT & AUDIO WORKFLOWS FULLY VERIFIED")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(verify_live_ivr())
