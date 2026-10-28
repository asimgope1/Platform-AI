import os
import json
from typing import AsyncIterator, Dict, Any, Optional
import httpx
from platform_core.contracts.capabilities import ModelMetadata
from platform_core.contracts.requests import (
    NormalizedChatRequest,
    NormalizedChatResponse,
    NormalizedChoice,
    ChatMessage,
    UsageInfo,
    ToolCall,
    FunctionCall,
    StreamChunk,
    StreamChoice,
    StreamDelta,
)
from platform_core.contracts.providers import BaseChatAdapter
from platform_core.errors.exceptions import ProviderError


class OpenAICompatibleChatAdapter(BaseChatAdapter):
    """Adapter for any OpenAI-compatible API endpoint (OpenAI, vLLM, Together, Ollama, Groq, etc.)."""

    def __init__(self, provider_id: str, config: Dict[str, Any]):
        super().__init__(provider_id, config)
        base_url_env = config.get("base_url_env")
        api_key_env = config.get("api_key_env")

        self.base_url = (os.getenv(base_url_env) if base_url_env else None) or config.get("base_url") or "https://api.openai.com/v1"
        self.api_key = (os.getenv(api_key_env) if api_key_env else None) or config.get("api_key") or ""
        self.timeout = float(config.get("timeout_seconds", 60.0))
        self._last_offline_time = 0.0

    def _get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def chat_completion(
        self,
        request: NormalizedChatRequest,
        model_meta: ModelMetadata,
    ) -> NormalizedChatResponse:
        import time
        # Fast failover if remote tunnel recently dropped
        if time.time() - self._last_offline_time < 30.0:
            raise ProviderError(self.provider_id, "Upstream tunnel endpoint temporarily offline (circuit breaker active)")

        url = f"{self.base_url.rstrip('/')}/chat/completions"

        # Format messages according to OpenAI spec
        payload_messages = []
        for m in request.messages:
            msg_dict: Dict[str, Any] = {"role": m.role}
            if m.content is not None:
                msg_dict["content"] = m.content
            if m.tool_calls:
                msg_dict["tool_calls"] = [tc.model_dump() for tc in m.tool_calls]
            if m.tool_call_id:
                msg_dict["tool_call_id"] = m.tool_call_id
            payload_messages.append(msg_dict)

        payload: Dict[str, Any] = {
            "model": model_meta.underlying_model,
            "messages": payload_messages,
            "stream": False,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.tools:
            payload["tools"] = request.tools
        if request.tool_choice:
            payload["tool_choice"] = request.tool_choice

        try:
            timeout_cfg = httpx.Timeout(self.timeout, connect=1.0)
            async with httpx.AsyncClient(timeout=timeout_cfg) as client:
                res = await client.post(url, headers=self._get_headers(), json=payload)
        except Exception as exc:
            self._last_offline_time = time.time()
            raise ProviderError(self.provider_id, f"Network connection failed: {exc}")

        if res.status_code >= 400:
            raise ProviderError(
                self.provider_id,
                f"Upstream returned HTTP {res.status_code}: {res.text}",
                status_code=res.status_code,
            )

        data = res.json()
        choices = []
        for c in data.get("choices", []):
            msg = c.get("message", {})
            tool_calls = None
            if "tool_calls" in msg and msg["tool_calls"]:
                tool_calls = [
                    ToolCall(
                        id=tc.get("id", ""),
                        type=tc.get("type", "function"),
                        function=FunctionCall(
                            name=tc.get("function", {}).get("name", ""),
                            arguments=tc.get("function", {}).get("arguments", ""),
                        ),
                    )
                    for tc in msg["tool_calls"]
                ]

            choices.append(
                NormalizedChoice(
                    index=c.get("index", 0),
                    message=ChatMessage(
                        role=msg.get("role", "assistant"),
                        content=msg.get("content"),
                        tool_calls=tool_calls,
                    ),
                    finish_reason=c.get("finish_reason", "stop"),
                )
            )

        usage_dict = data.get("usage", {})
        usage = UsageInfo(
            prompt_tokens=usage_dict.get("prompt_tokens", 0),
            completion_tokens=usage_dict.get("completion_tokens", 0),
            total_tokens=usage_dict.get("total_tokens", 0),
        )

        return NormalizedChatResponse(
            id=data.get("id", "chatcmpl-default"),
            model=model_meta.alias,
            created=data.get("created", 0),
            choices=choices,
            usage=usage,
            system_fingerprint=data.get("system_fingerprint"),
        )

    async def stream_chat_completion(
        self,
        request: NormalizedChatRequest,
        model_meta: ModelMetadata,
    ) -> AsyncIterator[StreamChunk]:
        url = f"{self.base_url.rstrip('/')}/chat/completions"

        payload_messages = []
        for m in request.messages:
            msg_dict: Dict[str, Any] = {"role": m.role}
            if m.content is not None:
                msg_dict["content"] = m.content
            if m.tool_calls:
                msg_dict["tool_calls"] = [tc.model_dump() for tc in m.tool_calls]
            if m.tool_call_id:
                msg_dict["tool_call_id"] = m.tool_call_id
            payload_messages.append(msg_dict)

        payload: Dict[str, Any] = {
            "model": model_meta.underlying_model,
            "messages": payload_messages,
            "stream": True,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.tools:
            payload["tools"] = request.tools

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream("POST", url, headers=self._get_headers(), json=payload) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    raise ProviderError(
                        self.provider_id,
                        f"Upstream returned HTTP {response.status_code}: {body.decode('utf-8', errors='ignore')}",
                        status_code=response.status_code,
                    )

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[len("data: ") :].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk_dict = json.loads(data_str)
                        choices = []
                        for c in chunk_dict.get("choices", []):
                            delta_dict = c.get("delta", {})
                            choices.append(
                                StreamChoice(
                                    index=c.get("index", 0),
                                    delta=StreamDelta(
                                        role=delta_dict.get("role"),
                                        content=delta_dict.get("content"),
                                        tool_calls=delta_dict.get("tool_calls"),
                                    ),
                                    finish_reason=c.get("finish_reason"),
                                )
                            )
                        yield StreamChunk(
                            id=chunk_dict.get("id", "chatcmpl-stream"),
                            model=model_meta.alias,
                            created=chunk_dict.get("created", 0),
                            choices=choices,
                        )
                    except json.JSONDecodeError:
                        continue
