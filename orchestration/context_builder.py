from typing import List
from platform_core.contracts.requests import ChatMessage


class ContextBuilder:
    """Builds and augments conversation messages with RAG context and system constraints."""

    def build_with_rag_context(
        self,
        messages: List[ChatMessage],
        rag_context: str,
    ) -> List[ChatMessage]:
        if not rag_context:
            return messages

        system_instruction = (
            "You have access to the following relevant context from verified platform documents:\n\n"
            f"{rag_context}\n\n"
            "Use this context to answer domain-specific questions when relevant, while maintaining your general knowledge, common sense, and conversational intelligence."
        )

        augmented_messages = list(messages)
        # 1. Update or prepend system message
        if augmented_messages and augmented_messages[0].role == "system":
            existing = augmented_messages[0].content or ""
            augmented_messages[0] = ChatMessage(
                role="system",
                content=f"{existing}\n\n{system_instruction}",
            )
        else:
            augmented_messages.insert(0, ChatMessage(role="system", content=system_instruction))

        # 2. Also inject context into the last user message for models (like Gemma) that lack native system turn support
        for i in range(len(augmented_messages) - 1, -1, -1):
            if augmented_messages[i].role == "user":
                user_msg = augmented_messages[i]
                augmented_content = (
                    f"Relevant Context (if applicable to query):\n{rag_context}\n\n"
                    f"User Message:\n{user_msg.content}"
                )
                augmented_messages[i] = ChatMessage(
                    role="user",
                    content=augmented_content,
                    name=user_msg.name,
                    tool_calls=user_msg.tool_calls,
                    tool_call_id=user_msg.tool_call_id,
                )
                break

        return augmented_messages
