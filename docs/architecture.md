# AI Platform Architecture

The standalone AI platform is designed around strict separation of concerns, pluggable contracts, and zero vendor lock-in.

```
                    +---------------------------+
                    | Client Request (OpenAI)   |
                    +-------------+-------------+
                                  |
                                  v
                    +---------------------------+
                    | Auth & Tenant Middleware  |
                    | (API Keys, Scopes, Rate)  |
                    +-------------+-------------+
                                  |
                                  v
                    +---------------------------+
                    |   FastAPI Route Layer     |
                    +-------------+-------------+
                                  |
                                  v
                    +---------------------------+
                    |    Chat Orchestrator      |
                    +---+---------+---------+---+
                        |         |         |
         +--------------+         |         +--------------+
         v                        v                        v
+------------------+    +-------------------+    +------------------+
| Model Resolver   |    | Capability Check  |    | Context Builder  |
| & Declarative    |    | (vision, tools,   |    | & RAG Pipeline   |
| Routing YAML     |    |  streaming)       |    | (chunks/rerank)  |
+------------------+    +-------------------+    +------------------+
         |                        |                        |
         +--------------+         |         +--------------+
                        |         |         |
                        v         v         v
                    +---------------------------+
                    |    Tool Loop Runner       |
                    | (Iterative Tool Calls)    |
                    +-------------+-------------+
                                  |
                                  v
                    +---------------------------+
                    | Pluggable Provider Layer  |
                    | Mock, OpenAI, HuggingFace |
                    +-------------+-------------+
```

## Security & Multitenancy
1. **API Keys**: Stored as SHA-256 hashes in PostgreSQL/SQLite. Raw keys are never persisted.
2. **Tenant Isolation**: Every database record (`documents`, `document_chunks`, `usage_events`) has a mandatory `tenant_id` foreign key. Searches and queries enforce tenant partition boundaries.
3. **Scopes**: API keys contain comma-separated scopes (e.g. `chat:completions`, `documents:write`, `audio:transcribe`).
