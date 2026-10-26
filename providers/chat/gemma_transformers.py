import time
import uuid
import asyncio
import torch
from threading import Lock
from typing import Dict, Any, Optional, List, AsyncIterator
from platform_core.contracts.capabilities import ModelMetadata
from platform_core.contracts.requests import (
    NormalizedChatRequest,
    NormalizedChatResponse,
    NormalizedChoice,
    ChatMessage,
    UsageInfo,
    StreamChunk,
)
from platform_core.contracts.providers import BaseChatAdapter
from platform_core.errors.exceptions import ProviderError


class GemmaTransformersChatAdapter(BaseChatAdapter):
    """Adapter for local Gemma 4 inference using Hugging Face Transformers.

    Features:
    - Lazy model and tokenizer loading (loaded only once per process)
    - Thread-safe initialization guarded by Lock
    - Synchronous generation executed in a worker thread via asyncio.to_thread (non-blocking)
    - Extracts only newly generated tokens, excluding the input prompt
    - Maps output cleanly into the normalized platform response contract
    """

    def __init__(self, provider_id: str, config: Dict[str, Any]):
        super().__init__(provider_id, config)
        self.model_id = (
            config.get("model")
            or config.get("model_id")
            or "google/gemma-4-E2B-it"
        )
        self.device = config.get("device", "auto")
        self.max_tokens_default = int(config.get("max_tokens_default", 512))
        self._model = None
        self._tokenizer = None
        self._processor = None
        self._load_lock = Lock()

    def _load_sync(self):
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    try:
                        import torch
                        from transformers import AutoProcessor, AutoModelForMultimodalLM, AutoTokenizer, AutoModelForCausalLM

                        dtype = torch.bfloat16 if (torch.cuda.is_available() or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())) else torch.float32

                        # 1. Official Gemma 4 Multimodal path: AutoProcessor & AutoModelForMultimodalLM
                        try:
                            self._processor = AutoProcessor.from_pretrained(self.model_id)
                            self._tokenizer = getattr(self._processor, "tokenizer", self._processor)
                            self._model = AutoModelForMultimodalLM.from_pretrained(
                                self.model_id,
                                torch_dtype=dtype,
                                device_map=self.device if torch.cuda.is_available() else None,
                            )
                        except Exception:
                            # 2. Fallback to AutoTokenizer & AutoModelForCausalLM
                            self._processor = None
                            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                            self._model = AutoModelForCausalLM.from_pretrained(
                                self.model_id,
                                torch_dtype=dtype,
                                device_map=self.device if torch.cuda.is_available() else None,
                            )
                    except Exception as exc:
                        raise ProviderError(
                            self.provider_id,
                            f"Failed to load Gemma model '{self.model_id}': {exc}. "
                            "Ensure model weights are accessible and system memory is sufficient.",
                        )

    def _generate_sync(
        self,
        request: NormalizedChatRequest,
        model_meta: ModelMetadata,
    ) -> NormalizedChatResponse:
        self._load_sync()

        # Format conversation messages for chat template
        formatted_messages = []
        for m in request.messages:
            content_str = m.content if isinstance(m.content, str) else str(m.content or "")
            formatted_messages.append({"role": m.role, "content": content_str})

        try:
            # Apply chat template with processor (or tokenizer fallback)
            processor = self._processor or self._tokenizer
            if hasattr(processor, "apply_chat_template"):
                try:
                    inputs = processor.apply_chat_template(
                        formatted_messages,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                except TypeError:
                    inputs = processor.apply_chat_template(
                        formatted_messages,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                        add_generation_prompt=True,
                    )
            else:
                inputs = processor(
                    "\n".join(f"{m['role']}: {m['content']}" for m in formatted_messages),
                    return_tensors="pt",
                )

            device = getattr(self._model, "device", None)
            if device is not None and not isinstance(device, (str, torch.device)):
                device = None
            if device is not None:
                inputs = {k: v.to(device) for k, v in inputs.items()}
            input_ids = inputs["input_ids"]
            input_len = input_ids.shape[-1]

            max_new = request.max_tokens or self.max_tokens_default
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=request.temperature is not None and request.temperature > 0,
                temperature=request.temperature if (request.temperature and request.temperature > 0) else None,
                top_p=request.top_p if (request.top_p and request.top_p < 1.0) else None,
            )

            # Extract ONLY newly generated tokens (excluding prompt)
            new_tokens = outputs[0][input_len:]
            output_text = processor.decode(new_tokens, skip_special_tokens=True)
            gen_len = len(new_tokens)

        except Exception as exc:
            raise ProviderError(self.provider_id, f"Inference error with '{self.model_id}': {exc}")

        return NormalizedChatResponse(
            id=f"chatcmpl-gemma-{uuid.uuid4().hex[:8]}",
            model=model_meta.alias,
            created=int(time.time()),
            choices=[
                NormalizedChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=output_text),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=input_len,
                completion_tokens=gen_len,
                total_tokens=input_len + gen_len,
            ),
        )

    async def chat_completion(
        self,
        request: NormalizedChatRequest,
        model_meta: ModelMetadata,
    ) -> NormalizedChatResponse:
        return await asyncio.to_thread(self._generate_sync, request, model_meta)

    async def stream_chat_completion(
        self,
        request: NormalizedChatRequest,
        model_meta: ModelMetadata,
    ) -> AsyncIterator[StreamChunk]:
        # Local transformers streaming can be added via TextIteratorStreamer if requested
        raise ProviderError(
            self.provider_id,
            "Streaming is currently disabled for local Gemma Transformers adapter.",
        )
        yield  # make generator
