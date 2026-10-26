# Standalone AI Platform

A production-grade, provider-agnostic standalone AI platform exposing standard OpenAI-compatible REST APIs for multi-tenant enterprise applications.

---

## Key Features

- **Standard OpenAI-Compatible API**: Seamless drop-in replacement for OpenAI endpoints (`/v1/chat/completions`, `/v1/models`, `/v1/embeddings`, `/v1/audio/transcriptions`, `/v1/audio/speech`).
- **Declarative YAML Routing**: Model aliases, provider definitions, and failover fallbacks configured via `config/providers.yaml` and `config/routing.yaml`.
- **Multi-Tenant Security**: Tenant isolation, SHA-256 hashed API keys, granular scope permissions (`chat:completions`, `documents:write`, `audio:transcribe`, `tools:execute`), and sliding-window rate limiting.
- **RAG & Vector Retrieval**: Automatic text chunking, embedding generation, vector similarity search with `pgvector` and in-memory fallback, and semantic reranking.
- **Autonomous Tool Calling Loop**: Native OpenAI function calling support with permission enforcement and multi-turn iterative execution.
- **Streaming & Observability**: Real-time Server-Sent Events (SSE) streaming, structured JSON logging with correlation IDs, and Prometheus metrics.

---

## Directory Structure

```
ai-platform/
├── api/                   # FastAPI route handlers, schemas, and middleware
├── audio/                 # Unified audio service (STT / TTS)
├── config/                # Declarative providers and routing YAML configurations
├── deploy/                # Dockerfile, docker-compose, and PostgreSQL setup
├── docs/                  # Architecture and API documentation
├── orchestration/         # Chat orchestration, tool calling, model resolution
├── platform_core/         # Abstract contracts, security, errors, observability
├── providers/             # Pluggable adapters (chat, embeddings, STT, TTS, vector stores, rerankers)
├── retrieval/             # Document chunking, ingestion, and RAG pipeline
├── scripts/               # Administrative CLI utilities (API key creation, migration, ingestion)
├── storage/               # SQLAlchemy async database models, repositories, and migrations
├── tests/                 # Unit, contract, security, and integration test suites
└── tools/                 # Allowlisted callable business tools
```

---

## Quickstart

### 1. Installation

```bash
# Clone and navigate to the project
cd ai-platform

# Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install platform and development dependencies
pip install -e ".[dev]"
```

### 2. Environment Configuration

Copy the example environment file and configure desired secrets:

```bash
cp .env.example .env
```

### 3. Initialize Database & Generate API Key

```bash
# Initialize database tables
python scripts/migrate_db.py

# Create a tenant API key
python scripts/create_api_key.py --tenant "AcmeCorp" --key-name "Dev Key" --scopes "*"
```

### 4. Run the API Server

```bash
uvicorn services.api.main:app --host 0.0.0.0 --port 8000 --reload
```

The interactive Swagger documentation will be available at [http://localhost:8000/docs](http://localhost:8000/docs).

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health, database readiness, and runtime metrics |
| `GET` | `/v1/models` | List all available configured model aliases |
| `GET` | `/v1/models/{alias}` | Retrieve metadata for a specific model |
| `POST` | `/v1/chat/completions` | Execute chat completion (streaming SSE, tool loop, RAG) |
| `POST` | `/v1/embeddings` | Generate vector embeddings for text |
| `POST` | `/v1/documents` | Ingest and index documents for tenant RAG retrieval |
| `GET` | `/v1/documents` | List indexed documents for current tenant |
| `POST` | `/v1/documents/search` | Execute vector similarity search |
| `DELETE` | `/v1/documents/{id}` | Delete document and corresponding chunks |
| `POST` | `/v1/audio/transcriptions` | Transcribe speech audio into text (STT) |
| `POST` | `/v1/audio/speech` | Synthesize speech audio from text (TTS) |

---

## Testing

Run the automated test suite with `pytest`:

```bash
pytest
```

---

## Production Deployment (Docker)

```bash
docker compose -f deploy/docker-compose.yml up --build -d
```
