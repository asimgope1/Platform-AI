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
            "Answer the user's questions truthfully and accurately based on this context. "
            "Cite sources where relevant."
        )

        augmented_messages = list(messages)
        # Check if first message is a system message
        if augmented_messages and augmented_messages[0].role == "system":
            existing = augmented_messages[0].content or ""
            augmented_messages[0] = ChatMessage(
                role="system",
                content=f"{existing}\n\n{system_instruction}",
            )
        else:
            augmented_messages.insert(0, ChatMessage(role="system", content=system_instruction))

        return augmented_messages
