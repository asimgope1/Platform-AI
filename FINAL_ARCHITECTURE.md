# Final Multi-Tenant Conversational AI Platform Architecture

## 1. High-Level Preserved Hybrid Architecture

FastAPI `ai-platform` on MacBook remains the **single primary AI orchestrator**, connecting to both remote Windows Django business tools (when enabled) and direct customer databases through the local database-agnostic Universal Data Access Layer.

```
Voice Caller / Client (SIP/Asterisk WebSocket, OpenAI-compatible HTTP /v1/chat/completions)
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ MacBook — FastAPI Primary AI Platform (services/api/main.py)            │
│                                                                         │
│  ┌──────────────────────┐    ┌────────────────────────────────────────┐ │
│  │ Voice & Audio Layer  │    │ Auth, Tenant Context, Rate Limiting    │ │
│  │ • Faster-Whisper STT │    │ • Bearer API key + MASTER_API_KEY      │ │
│  │ • Edge-TTS / Cache   │───▶│ • TenantContext isolation              │ │
│  │ • VAD / Barge-in     │    │ • Per-client & per-tenant rate guards  │ │
│  └──────────────────────┘    └───────────────────┬────────────────────┘ │
│                                                  ▼                      │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ Single Primary AI Orchestrator                                     │ │
│  │  ChatOrchestrator + ToolLoopRunner + RAGFirstResponseGenerator     │ │
│  │  • Gemma 4 / Ollama tool calling (windows_ollama, gemma)           │ │
│  │  • Turn execution dedup, max_iterations=5, execution time budget   │ │
│  │  • Refuses silent non-tool fallback when tools are active          │ │
│  └──────────────┬───────────────────────────────────┬─────────────────┘ │
│                 │                                   │                   │
│                 ▼                                   ▼                   │
│  ┌──────────────────────────────┐  ┌──────────────────────────────────┐ │
│  │ Hybrid RAG & Sync Engine     │  │ ToolExecutor + Business Tools    │ │
│  │ • PostgreSQL pgvector        │  │ • 10 universal business tools    │ │
│  │   (384-d / 768-d) + lexical  │  │ • Hybrid routing:                │ │
│  │ • PDF/DOCX/TXT/CSV parsers   │  │   1. Remote Django client        │ │
│  │ • Real-Time Sync Outbox /    │  │      (HMAC-SHA256, Bearer, API   │ │
│  │   Polling / DLQ / Tombstones │  │      Key, idempotency, retries)  │ │
│  └──────────────┬───────────────┘  │   2. Local Universal Data Access │ │
│                 │                  │      Service (read-only adapters)│ │
│                 ▼                  └──────────┬───────────────┬───────┘ │
│  ┌──────────────────────────────┐             │               │         │
│  │ Internal AI DB               │             │               │         │
│  │ (tenants, api_keys, docs,    │             │               │         │
│  │  tenant_database_configs,    │             │               │         │
│  │  tool_execution_audits,      │             │               │         │
│  │  sync_events, sync_dlq)      │             │               │         │
│  └──────────────────────────────┘             │               │         │
└───────────────────────────────────────────────┼───────────────┼─────────┘
                                                │               │
             ┌──────────────────────────────────┘               └───────────────────────────┐
             ▼                                                                              ▼
┌───────────────────────────────────────────┐    ┌───────────────────────────────────────────────────┐
│ Windows Host (192.168.1.141)              │    │ Heterogeneous Customer DBs (Read-Only)            │
│ • Django Business Tool Service (:8000)    │    │ • PostgreSQL, MySQL, MSSQL, Oracle, MongoDB,      │
│ • Ollama Gemma 4 (:11434)                 │    │   SQLite via per-tenant dynamic entity mapping    │
└───────────────────────────────────────────┘    └───────────────────────────────────────────────────┘
```

---

## 2. Database Separation Principle
- **Internal AI Platform DB (`storage/models/`)**: Stores only platform metadata (`tenants`, `api_keys`, `documents`, `document_chunks`, `conversations`, `call_intelligence`, `tenant_database_configs`, `tool_execution_audits`, `sync_events`, `sync_record_states`, `sync_dead_letters`). Managed via Alembic migrations `001` through `005`.
- **Customer Business DBs (`data_access/connectors/`)**: Connected read-only with encrypted credentials (`CredentialCipher`), read-only session modes, schema discovery, and canonical 8-entity mapping (`customer`, `order`, `payment`, `invoice`, `product`, `support_ticket`, `inventory`, `shipment`). AI platform migrations are never applied to customer DBs.

---

## 3. Single Primary Orchestrator Rule
- `ChatOrchestrator` (`orchestration/chat_orchestrator.py`) + `ToolLoopRunner` (`orchestration/tool_loop.py`) + `ToolExecutor` (`tools/executor.py`) form the sole tool-calling pipeline.
- Remote Windows Django serves purely as a remote tool catalog and tool executor (`/api/v1/tools/`, `/api/v1/tools/execute/`) — never as a competing second orchestrator.

