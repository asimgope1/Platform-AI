import json
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
    """Executes multi-turn tool loops safely up to a maximum turn limit."""

    def __init__(self, tool_executor: ToolExecutor, max_iterations: int = 5):
        self.tool_executor = tool_executor
        self.max_iterations = max_iterations

    async def run(
        self,
        adapter: BaseChatAdapter,
        request: NormalizedChatRequest,
        model_meta: ModelMetadata,
        context: Optional[TenantContext] = None,
    ) -> NormalizedChatResponse:
        current_messages = list(request.messages)
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1

            current_request = request.model_copy(update={"messages": current_messages, "stream": False})
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

                logger.info(f"Executing tool '{func_name}' for call '{call_id}'")
                exec_result = await self.tool_executor.execute_tool(
                    call_id=call_id,
                    name=func_name,
                    arguments_json=func_args,
                    context=context,
                )

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
