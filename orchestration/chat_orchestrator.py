import time
from typing import AsyncIterator, Optional, Dict, Any, Union
from platform_core.contracts.requests import (
    NormalizedChatRequest,
    NormalizedChatResponse,
    StreamChunk,
)
from platform_core.security.tenant_context import TenantContext
from platform_core.observability.metrics import metrics
from platform_core.observability.logging import logger
from providers.registry import provider_registry
from orchestration.model_resolver import ModelResolver
from orchestration.capability_check import CapabilityChecker
from orchestration.context_builder import ContextBuilder
from orchestration.tool_loop import ToolLoopRunner
from orchestration.response_formatter import ResponseFormatter
from tools.executor import ToolExecutor
from retrieval.rag_pipeline import RAGPipeline


class ChatOrchestrator:
    """Core runtime orchestrator for chat completion requests."""

    def __init__(
        self,
        rag_pipeline: Optional[RAGPipeline] = None,
        tool_executor: Optional[ToolExecutor] = None,
    ):
        self.model_resolver = ModelResolver()
        self.capability_checker = CapabilityChecker()
        self.context_builder = ContextBuilder()
        self.tool_loop_runner = ToolLoopRunner(tool_executor or ToolExecutor())
        self.formatter = ResponseFormatter()
        self.rag_pipeline = rag_pipeline

    async def execute(
        self,
        request: NormalizedChatRequest,
        context: Optional[TenantContext] = None,
        enable_rag: bool = False,
    ) -> Union[NormalizedChatResponse, AsyncIterator[StreamChunk]]:
        start_time = time.perf_counter()

        # 1. Resolve requested model alias
        model_meta = self.model_resolver.resolve(request.model_alias)

        # 2. Check capabilities
        self.capability_checker.check(request, model_meta)

        # 3. Retrieve and augment RAG context if requested and pipeline available
        current_request = request
        if enable_rag and self.rag_pipeline and context:
            user_msg = ""
            for m in reversed(request.messages):
                if m.role == "user" and m.content:
                    user_msg = str(m.content)
                    break
            if user_msg:
                rag_context = await self.rag_pipeline.retrieve_context(
                    tenant_id=context.tenant_id,
                    query=user_msg,
                )
                if rag_context:
                    augmented_messages = self.context_builder.build_with_rag_context(
                        request.messages, rag_context
                    )
                    current_request = request.model_copy(update={"messages": augmented_messages})

        # 4. Get chat adapter
        adapter = provider_registry.get_chat_adapter(model_meta.provider_id)

        # 5. Handle streaming vs non-streaming
        if current_request.stream:
            # Yield streaming chunks directly
            async def stream_wrapper():
                try:
                    async for chunk in adapter.stream_chat_completion(current_request, model_meta):
                        yield chunk
                finally:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    metrics.record_request(
                        endpoint="/v1/chat/completions",
                        model=model_meta.alias,
                        latency_ms=duration_ms,
                        status_code=200,
                    )

            return stream_wrapper()

        # 6. Non-streaming with optional tool loop
        try:
            if current_request.tools:
                response = await self.tool_loop_runner.run(
                    adapter=adapter,
                    request=current_request,
                    model_meta=model_meta,
                    context=context,
                )
            else:
                response = await adapter.chat_completion(current_request, model_meta)
        except Exception as e:
            logger.warning(f"Primary model {model_meta.alias} failed ({e}); falling back to local fallback provider.")
            try:
                fallback_meta = self.model_resolver.resolve("mock-chat")
                fallback_adapter = provider_registry.get_chat_adapter(fallback_meta.provider_id)
                response = await fallback_adapter.chat_completion(current_request, fallback_meta)
            except Exception:
                raise e

        # 7. Record metrics and return
        duration_ms = (time.perf_counter() - start_time) * 1000
        metrics.record_request(
            endpoint="/v1/chat/completions",
            model=model_meta.alias,
            latency_ms=duration_ms,
            status_code=200,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )

        return response
