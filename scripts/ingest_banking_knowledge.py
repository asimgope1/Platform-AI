"""
Ingest Banking Knowledge Dataset into pgvector
==============================================
Usage:
    .venv/bin/python scripts/ingest_banking_knowledge.py
    .venv/bin/python scripts/ingest_banking_knowledge.py --dataset-dir data/banking_knowledge --tenant-id <uuid>
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from services.knowledge.ingestion_service import KnowledgeIngestionService
from storage.database import init_db

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("scripts.ingest_banking_knowledge")


async def run_ingestion(dataset_dir: str, tenant_id: str | None) -> dict[str, int]:
    await init_db()
    service = KnowledgeIngestionService(embedding_provider_id="semantic-default")
    return await service.ingest_directory(dataset_dir=dataset_dir, tenant_id=tenant_id)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest structured banking knowledge JSON files into pgvector.")
    parser.add_argument(
        "--dataset-dir",
        default=os.path.join(ROOT_DIR, "data", "banking_knowledge"),
        help="Path to the structured knowledge JSON directory.",
    )
    parser.add_argument(
        "--tenant-id",
        default=None,
        help="Optional override tenant_id (defaults to tenant_id inside each JSON document).",
    )
    args = parser.parse_args()

    result = asyncio.run(run_ingestion(args.dataset_dir, args.tenant_id))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

