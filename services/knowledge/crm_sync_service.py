"""
CRM Knowledge & pgvector Sync Service
====================================
Synchronizes live CRM records (Complaints, Customers, Phases, Follow-ups)
from crm_backend_db (192.168.1.124:5435) into:
1. Structured RAG knowledge documents (crm_knowledge.json)
2. Native pgvector cosine search embeddings (document_chunks table)
3. In-memory IVR status catalog (COMPLAINT_STATUS_CATALOG)
"""
from __future__ import annotations

import os
import sys
import asyncio
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("services.knowledge.crm_sync_service")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


async def sync_crm_knowledge_to_pgvector(tenant_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Extracts live CRM database records, regenerates crm_knowledge.json,
    and updates pgvector document chunks.
    """
    from scripts.build_crm_chunks import extract_and_generate_crm_knowledge
    from scripts.ingest_banking_knowledge import run_ingestion
    from providers.chat.ivr_mini_model import ivr_mini_brain

    tid = tenant_id or os.getenv("CRM_TENANT_ID", "879e9bda-8049-4824-9aa0-fc5e9f0012d9")
    try:
        logger.info("[CRMSync] 1/3 Extracting live CRM data from PostgreSQL...")
        out_path = await extract_and_generate_crm_knowledge()

        logger.info("[CRMSync] 2/3 Ingesting updated chunks into pgvector...")
        ingest_stats = await run_ingestion(dataset_dir=os.path.dirname(out_path), tenant_id=tid)

        logger.info("[CRMSync] 3/3 Refreshing IVR in-memory catalog cache...")
        await ivr_mini_brain.sync_live_crm_cache(force=True)

        logger.info(f"[CRMSync] ✓ Completed CRM pgvector sync: {ingest_stats}")
        return {
            "success": True,
            "dataset_path": out_path,
            "ingestion": ingest_stats,
        }
    except Exception as exc:
        logger.error(f"[CRMSync] Failed to sync CRM to pgvector: {exc}", exc_info=True)
        return {
            "success": False,
            "error": str(exc),
        }


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT_DIR, ".env"))
    result = asyncio.run(sync_crm_knowledge_to_pgvector())
    print(result)
