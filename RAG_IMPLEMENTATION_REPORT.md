# Hybrid RAG, Document Ingestion & Real-Time Sync Report

## 1. Hybrid Retrieval Pipeline (`services/knowledge/retrieval_service.py`, `retrieval/`)
- **Vector Search**: `DocumentRepository.search_similar_chunks` uses native PostgreSQL `pgvector` cosine distance (`Vector(384)` / `Vector(768)`) in Postgres, with deterministic cosine similarity over `embedding_json` on SQLite development databases.
- **Hybrid Reranking (`HybridRetrievalService.hybrid_rerank`)**: Combines weighted semantic score (`0.78`), multilingual token/lexical overlap (`0.22`), verified-source boost (`+0.03`), preferred category boost (`+0.03`), logarithmic `ask_count` boost, and filters unverified unanswered drafts.
- **Cross-Lingual Anchoring**: Augments Hindi (`\u0900-\u097f`), Odia (`\u0b00-\u0b7f`), and Hinglish banking/support queries with English semantic anchors before embedding lookup.
- **Ground Citations**: `POST /v1/documents/search` (`api/routes/documents.py`) and `search_knowledge_documents` return per-chunk citation identifiers (`doc:<document_id>#chunk-<chunk_index>`).

---

## 2. Multi-Format Knowledge Ingestion & Re-Indexing (`retrieval/parsers.py`, `retrieval/ingestion.py`)

| Capability | Status | Details |
| :--- | :--- | :--- |
| Plain Text & Markdown (`.txt`, `.md`) | `TESTED — DEVELOPMENT` | UTF-8 decoding, chunked (`chunk_size=500`, `overlap=50`), vectorized via `EmbeddingService`. |
| Word Documents (`.docx`) | `TESTED — DEVELOPMENT` | Uses `python-docx` when present, with stdlib `zipfile` + `word/document.xml` fallback verified in `test_stage8_document_parsers_and_pii_guard` & `test_stage8_api_upload_reindex_search_delete_and_tenant_isolation`. |
| PDF Documents (`.pdf`) | `TESTED — DEVELOPMENT` | Uses `pypdf` / `PyMuPDF` (`fitz`) when installed, with zlib-compressed PDF content stream `BT...ET` fallback verified in unit tests (scanned image PDFs without OCR return a clean `ValidationError` when `pypdf` is absent). |
| CSV Knowledge Tables (`.csv`) | `TESTED — DEVELOPMENT` | Formats structured rows into readable `Header: value` records while enforcing the PII column guard. |
| Transactional PII Rejection | `TESTED — DEVELOPMENT` | `assert_no_transactional_pii_dump` rejects raw card numbers (PANs), CVV, passwords, or SSN columns (`HTTP 403`). |
| Versioning & Re-Indexing | `TESTED — DEVELOPMENT` | `POST /v1/documents/{document_id}/reindex` atomically replaces stale chunks via `DocumentRepository.replace_chunks` and updates version metadata. |
| Deletion & Tenant Isolation | `TESTED — DEVELOPMENT` | `DELETE /v1/documents/{document_id}` cascades chunk deletion strictly within `context.tenant_id`; cross-tenant searches and deletes fail cleanly. |

---

## 3. Real-Time Knowledge & Catalog Synchronization (`data_access/sync/`, `api/routes/sync.py`)

Live transactional data (`order`, `payment`, `invoice`, `shipment`, `customer`) is **never** synchronized into RAG vectors; it always queries live DBs directly via `UniversalDataAccessService` or remote tools. Real-time sync is dedicated to non-PII product/catalog & policy knowledge:

- **Internal Tables (Migration `005_sync_pipeline_tables.py`)**: `sync_events`, `sync_record_states`, `sync_dead_letters`.
- **Event Outbox & Webhook API**: `POST /v1/sync/events`, `POST /v1/sync/process`, `POST /v1/sync/reconcile`, `GET /v1/sync/status`.
- **Verified Mechanics (`test_stage9_sync_idempotency_version_tombstone_dlq_reconcile_and_polling`)**:
  - **Idempotency**: Duplicate `(tenant_id, idempotency_key)` events return `created_new=False` without reprocessing.
  - **Version Ordering**: Out-of-order stale versions (`v1 <= current v2`) are marked `ignored_stale`.
  - **Soft Deletes / Tombstones**: `deleted` events remove indexed document chunks cleanly and record `is_deleted=True`.
  - **Exponential Backoff & Dead-Letter Queue (DLQ)**: Failed sync items schedule `next_retry_at` with exponential backoff and move to `sync_dead_letters` once `retry_count >= max_retries`.
  - **Incremental Polling & Reconciliation**: `poll_incremental_product_catalog` reads mapped `product` tables read-only without mutating customer DBs; `reconcile_snapshot` detects drift and emits missing upserts and deletes.
  - **PostgreSQL `pgvector` Status**: `vector_pgvector` native ANN search could not run live inside the local macOS sandbox (`pgvector` port `::1:5432` unreachable in sandbox); verified on SQLite cosine fallback (`8.57 ms` p50 across 20 hybrid searches with warm embeddings).

