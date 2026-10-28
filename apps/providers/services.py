"""
Unified Providers Bridge for Django AI Platform
Connects Django apps directly to swappable LLM, STT, TTS, and Embedding engines.
"""
import logging
from typing import List, Dict, Any, Optional
from platform_core.contracts.requests import NormalizedChatRequest, ChatMessage
from providers.registry import provider_registry

logger = logging.getLogger(__name__)

class ProviderBridge:
    _initialized = False

    @classmethod
    def ensure_initialized(cls):
        if not cls._initialized:
            try:
                provider_registry.initialize_from_file("config/providers.yaml")
                cls._initialized = True
                logger.info("ProviderBridge successfully initialized providers from config/providers.yaml")
            except Exception as e:
                logger.warning(f"ProviderBridge config load fallback: {e}")

    @classmethod
    async def generate_chat_response(
        cls,
        messages: List[Dict[str, str]],
        model_alias: str = "gemma-gpu",
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Generate response via configured model (e.g. google/gemma-4-E2B-it or local smoke)."""
        cls.ensure_initialized()
        try:
            if model_alias in provider_registry.models:
                meta = provider_registry.get_model_metadata(model_alias)
                adapter = provider_registry.get_chat_adapter(meta.provider_id)
            else:
                meta = provider_registry.get_model_metadata("mock-chat")
                adapter = provider_registry.get_chat_adapter(meta.provider_id)

            req = NormalizedChatRequest(
                model_alias=model_alias,
                messages=[ChatMessage(role=m["role"], content=m["content"]) for m in messages],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            res = await adapter.chat_completion(req, meta)
            content = res.choices[0].message.content if res.choices else ""
            return {
                "content": content,
                "model": res.model,
                "finish_reason": res.choices[0].finish_reason if res.choices else "stop",
                "prompt_tokens": res.usage.prompt_tokens,
                "completion_tokens": res.usage.completion_tokens,
            }
        except Exception as e:
            logger.warning(f"Chat model {model_alias} failed ({e}), falling back to mock provider.")
            try:
                meta = provider_registry.get_model_metadata("mock-chat")
                adapter = provider_registry.get_chat_adapter(meta.provider_id)
                req = NormalizedChatRequest(
                    model_alias="mock-chat",
                    messages=[ChatMessage(role=m["role"], content=m["content"]) for m in messages],
                    max_tokens=max_tokens,
                )
                res = await adapter.chat_completion(req, meta)
                content = res.choices[0].message.content if res.choices else ""
                return {
                    "content": content,
                    "model": "fallback-mock",
                    "finish_reason": "stop",
                    "prompt_tokens": 10,
                    "completion_tokens": 10,
                }
            except Exception:
                return {
                    "content": "Customer support representative available. How can I assist you?",
                    "model": "static-fallback",
                    "finish_reason": "stop",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                }

    @classmethod
    async def transcribe_audio(
        cls,
        audio_bytes: bytes,
        filename: str = "caller_audio.wav",
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Perceive Stage: Transcribe caller audio bytes with WhisperLocalSpeechToTextAdapter."""
        cls.ensure_initialized()
        stt_adapter = provider_registry.get_stt_adapter("whisper_local")
        return await stt_adapter.transcribe(audio_bytes=audio_bytes, filename=filename, language=language)

    @classmethod
    async def synthesize_speech(cls, text: str, voice: str = "alloy") -> bytes:
        """Respond Stage: Synthesize spoken response via TTS adapter."""
        cls.ensure_initialized()
        tts_adapter = provider_registry.get_tts_adapter("mock_tts")
        audio_data = await tts_adapter.synthesize(text=text, voice=voice, response_format="wav")
        if isinstance(audio_data, bytes):
            return audio_data
        elif hasattr(audio_data, "audio_bytes"):
            return audio_data.audio_bytes
        return bytes(audio_data)

    @classmethod
    def warmup_models(cls):
        """Warm up Whisper and Embeddings in background so live caller turns are sub-second."""
        cls.ensure_initialized()
        try:
            stt = provider_registry.get_stt_adapter("whisper_local")
            if hasattr(stt, "warmup"):
                stt.warmup()
        except Exception as e:
            logger.warning(f"STT warmup notice: {e}")

        try:
            emb = provider_registry.get_embedding_adapter("google-embeddings")
            if hasattr(emb, "warmup"):
                emb.warmup()
            from apps.knowledge_base.services import KnowledgeBaseService
            import asyncio
            emb_service = KnowledgeBaseService.get_embedding_service()
            try:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(emb_service.embed_query("warmup"))
                loop.close()
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"Embedding warmup notice: {e}")

        try:
            tts = provider_registry.get_tts_adapter("mock_tts")
            if hasattr(tts, "synthesize"):
                import asyncio
                phrases = [
                    "ନମସ୍କାର ରାହୁଲ ବାବୁ! କୁହନ୍ତୁ ଆଜ୍ଞା, ଆଜି କ’ଣ ସାହାଯ୍ୟ କରିବି?",
                    "ନମସ୍କାର ଆଜ୍ଞା! କୁହନ୍ତୁ, ଆଜି କ’ଣ ସାହାଯ୍ୟ କରିବି?",
                    "ଆଜ୍ଞା ରାହୁଲ ବାବୁ, ଜମା ବ୍ୟସ୍ତ ହୁଅନ୍ତୁନି! ୟୁପିଆଇରେ ଟଙ୍କା କଟିଯାଇଥିଲେ ୨୪ ରୁ ୪୮ ଘଣ୍ଟା ଭିତରେ ସିଧା ଆପଣଙ୍କ ଖାତାକୁ ଫେରିଆସିବ। ଆପଣ ଆପ୍‌ରୁ ୟୁଟିଆର୍ ନମ୍ବର ଦେଖି ବି ଷ୍ଟାଟସ୍ ଚେକ୍ କରିପାରିବେ।",
                    "ହଁ ଆଜ୍ଞା ରାହୁଲ ବାବୁ, ଭିଡିଓ କେୱାଇସି ବହୁତ ସହଜରେ ହୋଇଯିବ। ଖାଲି ଅସଲି ପ୍ୟାନ୍ କାର୍ଡ ଆଉ ଆଧାର ଥିବା ଫୋନ୍ ପାଖରେ ରଖନ୍ତୁ। ବାସ୍ ୨ ମିନିଟ୍‌ର କଲ୍ ହେବ, ଆଉ ଘଣ୍ଟାଏ ଭିତରେ ଖାତା ଚାଲୁ ହୋଇଯିବ।",
                    "ହଁ ଆଜ୍ଞା ରାହୁଲ ବାବୁ, ମୁଁ ବୁଝିପାରୁଛି ଆପଣଙ୍କ ଅସୁବିଧା। ମୁଁ ଏବେ ହିଁ ଆମର ବରିଷ୍ଠ ଏଜେଣ୍ଟଙ୍କୁ କଲ୍ ଟ୍ରାନ୍ସଫର କରୁଛି, ଟିକେ ଲାଇନ୍‌ରେ ରୁହନ୍ତୁ।",
                    "नमस्ते! हमारी ग्राहक सेवा हेल्पलाइन में आपका स्वागत है। बताइए, आज मैं आपकी क्या सहायता करूँ?",
                    "Hello! Welcome to our customer helpline. How may I assist you today?",
                    "जी बिल्कुल, आप चिंता मत कीजिए! अगर बैंक खाते से यूपीआई पैसे कट गए हैं, तो 24 से 48 घंटे में वो स्वतः आपके खाते में वापस आ जाते हैं। आप अपनी यूपीआई ऐप में 12 अंकों के यूटीआर नंबर से भी स्टेटस चेक कर सकते हैं।",
                    "Don't worry at all! If your UPI payment was debited, the amount will be automatically refunded back to your bank account within 24 to 48 hours. You can also track it anytime using your 12-digit UTR reference.",
                    "जी, वीडियो केवाईसी करना बहुत ही आसान है! बस अपना ओरिजिनल पैन कार्ड और आधार से लिंक मोबाइल पास रखिएगा। सिर्फ 2 मिनट का वीडियो कॉल होगा, और 1 घंटे में आपका खाता चालू हो जाएगा।",
                    "Video KYC is super easy! Just keep your original physical PAN card and Aadhaar-linked mobile ready. It takes only 2 minutes and your account will be activated within an hour."
                ]
                loop = asyncio.new_event_loop()
                for p in phrases:
                    try:
                        loop.run_until_complete(tts.synthesize(text=p))
                    except Exception:
                        pass
                loop.close()
        except Exception as e:
            logger.warning(f"TTS warmup notice: {e}")
