#!/usr/bin/env python3
import asyncio
import sys
import argparse
from pathlib import Path
from storage.database import async_session, init_db
from storage.repositories.tenant_repo import TenantRepository
from storage.repositories.document_repo import DocumentRepository
from retrieval.ingestion import DocumentIngestionService
from retrieval.embedding_service import EmbeddingService
from providers.registry import provider_registry


async def main():
    parser = argparse.ArgumentParser(description="Ingest document into tenant knowledge base.")
    parser.add_argument("--tenant", type=str, required=True, help="Tenant name")
    parser.add_argument("--file", type=str, required=True, help="Path to text/markdown file")
    parser.add_argument("--title", type=str, default="", help="Document title (optional)")

    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.is_file():
        print(f"Error: File '{args.file}' does not exist.")
        sys.exit(1)

    content = file_path.read_text(encoding="utf-8")
    title = args.title or file_path.stem

    await init_db()
    provider_registry.initialize_from_file("config/providers.yaml")

    async with async_session() as session:
        tenant_repo = TenantRepository(session)
        tenant = await tenant_repo.get_by_name(args.tenant)
        if not tenant:
            print(f"Tenant '{args.tenant}' not found. Creating tenant...")
            tenant = await tenant_repo.create(args.tenant)

        doc_repo = DocumentRepository(session)
        emb_service = EmbeddingService()
        ingestion = DocumentIngestionService(doc_repo, emb_service)

        doc = await ingestion.ingest_document(
            tenant_id=tenant.id,
            title=title,
            content=content,
            source_type=file_path.suffix.lstrip(".") or "text",
        )

        print("\n" + "=" * 60)
        print("DOCUMENT INGESTED SUCCESSFULLY")
        print("=" * 60)
        print(f"Document ID:   {doc.id}")
        print(f"Tenant:        {tenant.name} ({tenant.id})")
        print(f"Title:         {doc.title}")
        print(f"Total Chunks:  {doc.total_chunks}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
