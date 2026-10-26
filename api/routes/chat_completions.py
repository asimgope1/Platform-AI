import json
from typing import AsyncIterator
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from api.schemas.chat import ChatCompletionRequestSchema, ChatCompletionResponseSchema
from platform_core.contracts.requests import NormalizedChatRequest, ChatMessage, ToolCall, FunctionCall
from platform_core.security.authentication import get_current_tenant
from platform_core.security.tenant_context import TenantContext
from platform_core.observability.logging import request_id_ctx, logger
from storage.database import get_db_session
from storage.repositories.usage_repo import UsageRepository
from services.api.dependencies import get_chat_orchestrator
from orchestration.chat_orchestrator import ChatOrchestrator

router = APIRouter(prefix="/v1", tags=["Chat Completions"])


@router.post("/chat/completions", response_model=None)
async def create_chat_completion(
    payload: ChatCompletionRequestSchema,
    request: Request,
    context: TenantContext = Depends(get_current_tenant),
    orchestrator: ChatOrchestrator = Depends(get_chat_orchestrator),
    session: AsyncSession = Depends(get_db_session),
):
    """OpenAI-compatible chat completion endpoint supporting streaming, tool calling, and RAG."""
    # Convert incoming schema to normalized platform contract
    normalized_messages = []
    for msg in payload.messages:
        tool_calls = None
        if msg.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.get("id", ""),
                    type=tc.get("type", "function"),
                    function=FunctionCall(
                        name=tc.get("function", {}).get("name", ""),
                        arguments=tc.get("function", {}).get("arguments", ""),
                    ),
                )
                for tc in msg.tool_calls
            ]
        normalized_messages.append(
            ChatMessage(
                role=msg.role,
                content=msg.content,
                name=msg.name,
                tool_calls=tool_calls,
                tool_call_id=msg.tool_call_id,
            )
        )

    norm_request = NormalizedChatRequest(
        model_alias=payload.model,
        messages=normalized_messages,
        stream=bool(payload.stream),
        temperature=payload.temperature,
        top_p=payload.top_p,
        max_tokens=payload.max_tokens,
        tools=payload.tools,
        tool_choice=payload.tool_choice,
        user=payload.user,
    )

    req_id = request_id_ctx.get() or "req_chat"

    # Handle streaming response
    if payload.stream:
        stream_gen = await orchestrator.execute(
            request=norm_request,
            context=context,
            enable_rag=bool(payload.enable_rag),
        )

        async def sse_event_generator() -> AsyncIterator[str]:
            try:
                async for chunk in stream_gen:
                    chunk_data = {
                        "id": chunk.id,
                        "object": "chat.completion.chunk",
                        "created": chunk.created,
                        "model": chunk.model,
                        "choices": [
                            {
                                "index": c.index,
                                "delta": {
                                    k: v
                                    for k, v in {
                                        "role": c.delta.role,
                                        "content": c.delta.content,
                                        "tool_calls": c.delta.tool_calls,
                                    }.items()
                                    if v is not None
                                },
                                "finish_reason": c.finish_reason,
                            }
                            for c in chunk.choices
                        ],
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Error during streaming: {e}")
                err_data = {"error": {"message": str(e), "type": "api_error"}}
                yield f"data: {json.dumps(err_data)}\n\n"

        return StreamingResponse(sse_event_generator(), media_type="text/event-stream")

    # Non-streaming response
    response = await orchestrator.execute(
        request=norm_request,
        context=context,
        enable_rag=bool(payload.enable_rag),
    )

    # Log usage event asynchronously in DB
    try:
        usage_repo = UsageRepository(session)
        model_meta = orchestrator.model_resolver.resolve(payload.model)
        await usage_repo.log_event(
            request_id=req_id,
            tenant_id=context.tenant_id,
            model_alias=payload.model,
            provider_id=model_meta.provider_id,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            status_code=200,
        )
    except Exception as exc:
        logger.warning(f"Could not log usage event: {exc}")

    formatted = orchestrator.formatter.format_chat_response(response)
    return JSONResponse(content=formatted)
