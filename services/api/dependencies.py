from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from storage.database import get_db_session
from storage.repositories.document_repo import DocumentRepository
from retrieval.embedding_service import EmbeddingService
from retrieval.search import SemanticSearchService
from retrieval.rag_pipeline import RAGPipeline
from tools.executor import ToolExecutor
from orchestration.chat_orchestrator import ChatOrchestrator


def get_rag_pipeline(session: AsyncSession = Depends(get_db_session)) -> RAGPipeline:
    doc_repo = DocumentRepository(session)
    embedding_service = EmbeddingService()
    search_service = SemanticSearchService(doc_repo, embedding_service)
    return RAGPipeline(search_service)


def get_chat_orchestrator(
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> ChatOrchestrator:
    tool_executor = ToolExecutor(timeout_seconds=10.0)
    return ChatOrchestrator(rag_pipeline=rag_pipeline, tool_executor=tool_executor)
