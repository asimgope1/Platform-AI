import json
import time
from typing import List, Optional
from platform_core.contracts.requests import (
    NormalizedChatRequest,
    NormalizedChatResponse,
    ChatMessage,
)
from platform_core.contracts.capabilities import ModelMetadata
from platform_core.contracts.providers import BaseChatAdapter
from platform_core.security.tenant_context import TenantContext
from tools.executor import ToolExecutor
from platform_core.observability.logging import logger


class ToolLoopRunner:
    """Executes multi-turn tool loops safely up to a maximum turn limit and time budget."""

    def __init__(
        self,
        tool_executor: ToolExecutor,
        max_iterations: int = 5,
        max_execution_seconds: float = 120.0,
    ):
        self.tool_executor = tool_executor
        self.max_iterations = max_iterations
        self.max_execution_seconds = max(0.01, float(max_execution_seconds))

    async def run(
        self,
        adapter: BaseChatAdapter,
        request: NormalizedChatRequest,
        model_meta: ModelMetadata,
        context: Optional[TenantContext] = None,
    ) -> NormalizedChatResponse:
        current_messages = list(request.messages)
        iteration = 0
        started_at = time.monotonic()
        # Prevent duplicate execution of identical tool invocations within a single turn loop
        turn_exec_cache: dict[tuple[str, str], object] = {}

        turn_metadata = dict(request.metadata or {})
        initial_auth_cust = turn_metadata.get("authorized_customer_id")

        while iteration < self.max_iterations:
            if iteration > 0 and (time.monotonic() - started_at) >= self.max_execution_seconds:
                logger.warning(
                    f"Tool loop hit execution time budget ({self.max_execution_seconds}s) after {iteration} iteration(s)."
                )
                break
            iteration += 1

            current_request = request.model_copy(update={"messages": current_messages, "stream": False, "metadata": turn_metadata})
            response = await adapter.chat_completion(current_request, model_meta)

            first_choice = response.choices[0] if response.choices else None
            if not first_choice or not first_choice.message.tool_calls:
                # No tool call was made, we have final answer
                return response

            # Model requested tool calls
            assistant_msg = first_choice.message
            current_messages.append(assistant_msg)

            for tool_call in assistant_msg.tool_calls or []:
                call_id = tool_call.id
                func_name = tool_call.function.name
                func_args = tool_call.function.arguments

                try:
                    canonical_args = json.dumps(json.loads(func_args or "{}"), sort_keys=True)
                except Exception:
                    canonical_args = str(func_args or "")
                dedup_key = (func_name, canonical_args)

                if dedup_key in turn_exec_cache:
                    logger.info(f"Reusing cached result for duplicate tool call '{func_name}' in turn loop.")
                    exec_result = turn_exec_cache[dedup_key]
                else:
                    logger.info(f"Executing tool '{func_name}' for call '{call_id}'")
                    exec_result = await self.tool_executor.execute_tool(
                        call_id=call_id,
                        name=func_name,
                        arguments_json=func_args,
                        context=context,
                        trusted_metadata=turn_metadata,
                    )
                    turn_exec_cache[dedup_key] = exec_result

                # If caller wasn't pre-bound and search_customer/get_customer_details identified a single customer,
                # propagate that verified customer_reference cleanly to follow-up tools & session metadata.
                if (
                    not initial_auth_cust
                    and not exec_result.is_error
                    and isinstance(exec_result.output, dict)
                    and func_name in ("search_customer", "get_customer_details")
                ):
                    out_payload = exec_result.output.get("data") if isinstance(exec_result.output.get("data"), dict) else exec_result.output
                    discovered_ref: Optional[str] = None
                    if out_payload.get("count") == 1:
                        custs = out_payload.get("customers") or out_payload.get("results") or []
                        if custs and isinstance(custs[0], dict):
                            discovered_ref = custs[0].get("customer_reference") or custs[0].get("customer_id")
                    elif out_payload.get("customer_reference") or out_payload.get("customer_id"):
                        discovered_ref = out_payload.get("customer_reference") or out_payload.get("customer_id")
                    if discovered_ref:
                        turn_metadata["authorized_customer_id"] = str(discovered_ref).strip()
                        if isinstance(request.metadata, dict):
                            request.metadata["resolved_customer_id"] = str(discovered_ref).strip()

                output_str = (
                    json.dumps(exec_result.output)
                    if not exec_result.is_error
                    else json.dumps({"error": exec_result.error_message})
                )

                # Append tool response message
                current_messages.append(
                    ChatMessage(
                        role="tool",
                        content=output_str,
                        tool_call_id=call_id,
                        name=func_name,
                    )
                )

        # Reached max iterations without finishing
        logger.warning(f"Tool loop reached maximum allowed iterations ({self.max_iterations})")
        return response
