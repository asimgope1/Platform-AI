import asyncio
import os
import sys
import wave
import io
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from providers.speech_to_text.whisper_local import WhisperLocalSpeechToTextAdapter, _clean_hallucinations_and_loops
from storage.database import async_session
from storage.repositories.document_repo import DocumentRepository
from retrieval.embedding_service import EmbeddingService
from retrieval.search import SemanticSearchService


def pcm_to_wav(pcm_data: bytes, sample_rate: int = 8000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


async def test_stt_and_rag():
    print("=" * 65)
    print("  TEST 1: Repetition Deduplication & Hallucination Filter")
    print("=" * 65)
    test_cases = [
        ("paribu paribu paribu paribu", "paribu"),
        ("video kyc video kyc video kyc kya hai", "video kyc kya hai"),
        ("नमस्ते नमस्ते नमस्ते जी", "नमस्ते जी"),
        ("normal single sentence without repetitions", "normal single sentence without repetitions"),
    ]
    for raw, expected in test_cases:
        cleaned = _clean_hallucinations_and_loops(raw)
        print(f"RAW:     '{raw}'")
        print(f"CLEANED: '{cleaned}' (Pass: {expected in cleaned})")
    print("\n")

    print("=" * 65)
    print("  TEST 2: Whisper STT Initialization & Transcription")
    print("=" * 65)
    adapter = WhisperLocalSpeechToTextAdapter("whisper_local", {"model": "openai/whisper-base"})
    
    cache_dir = Path("data/ivr_audio_cache")
    hindi_pcm_file = cache_dir / "prompt_795dcf74c43984d2.pcm"
    odia_pcm_file = cache_dir / "prompt_3f8eaeb9a8ae5156.pcm"

    # Test Hindi Transcription
    if hindi_pcm_file.exists():
        pcm_bytes = hindi_pcm_file.read_bytes()
        # Take first 5 seconds (5 * 8000 * 2 = 80,000 bytes)
        wav_bytes = pcm_to_wav(pcm_bytes[:80000], sample_rate=8000)
        res_hi = await adapter.transcribe(wav_bytes, filename="hindi_prompt.wav", language="hi")
        print(f"🎙️ Spoken Hindi Transcription:\n   '{res_hi['text']}'")
        has_arabic = any('\u0600' <= ch <= '\u06ff' for ch in res_hi['text'])
        print(f"   Contains Perso-Arabic/Urdu script? -> {has_arabic} (Must be False)")
        print(f"   Duration: {res_hi['duration']}s\n")
    else:
        print("Hindi PCM file not found in cache.")

    # Test Odia Translation
    if odia_pcm_file.exists():
        pcm_bytes = odia_pcm_file.read_bytes()
        # Take first 6 seconds
        wav_bytes = pcm_to_wav(pcm_bytes[:96000], sample_rate=8000)
        res_or = await adapter.transcribe(wav_bytes, filename="odia_prompt.wav", language="or")
        print(f"🎙️ Spoken Odia Translation (task=translate):\n   '{res_or['text']}'")
        print(f"   Duration: {res_or['duration']}s\n")
    else:
        print("Odia PCM file not found in cache.")

    print("=" * 65)
    print("  TEST 3: pgvector Semantic Search on Real Queries")
    print("=" * 65)
    ivr_tenant = os.getenv("IVR_TENANT_ID", "879e9bda-8049-4824-9aa0-fc5e9f0012d9")
    rag_threshold = 0.22

    queries = [
        ("video kyc documents required", "Video KYC English"),
        ("वीडियो केवाईसी के लिए कौन से दस्तावेज चाहिए", "Video KYC Hindi"),
        ("ଭିଡିଓ କେୱାଇସି ପାଇଁ କଣ ଡକ୍ୟୁମେଣ୍ଟ ଦରକାର", "Video KYC Odia"),
        ("what documents are required for video kyc", "Whisper Translated Odia"),
        ("upi payment failed refund status", "UPI Payment English"),
        ("यूपीआई पेमेंट कट गया रिफंड कब मिलेगा", "UPI Payment Hindi"),
    ]

    async with async_session() as session:
        doc_repo = DocumentRepository(session)
        emb_service = EmbeddingService(provider_id="semantic-default")
        search_service = SemanticSearchService(doc_repo=doc_repo, embedding_service=emb_service)

        for q, label in queries:
            results = await search_service.search(tenant_id=ivr_tenant, query=q, top_k=2)
            top_score = max((score for _, score in results), default=0.0)
            matched = top_score >= rag_threshold
            top_chunk_snippet = results[0][0].content[:70].replace('\n', ' ') if results else "None"
            print(f"Query [{label}]: '{q}'")
            print(f"   -> Top Score: {top_score:.3f} (>= {rag_threshold}? {matched}) | Snippet: {top_chunk_snippet}...")
    
    print("\n" + "=" * 65)
    print("  ALL VERIFICATIONS COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(test_stt_and_rag())
