# Performance Benchmark Report

**Platform:** Multi-Tenant Conversational AI & Voice IVR Platform (`ai-platform`)  
**Benchmark Runs Covered:**
1. **Live LAN End-to-End Benchmark (`benchmark_phase3a.py`)**: 20 iterations across 3 benchmark suites hitting the MacBook FastAPI `ChatOrchestrator`, LAN Ollama `gemma4:latest`, and the Windows Django service (`192.168.1.141:8000`).
2. **Local Database-Agnostic & RAG Concurrency Benchmark (`benchmark_stage11.py`)**: 20 warm iterations of local Hybrid RAG search and 20 concurrent `UniversalDataAccessService` tool executions backed by SQLite + warm MiniLM embeddings.

---

## 1. Executive Summary

| Benchmark Suite | Iterations / Concurrency | Success Rate | p50 Latency | p95 Latency | p99 Latency | Max Latency | Primary Bottleneck |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma 4 Single-Step Tool Selection (`gemma4:latest`) | 20 sequential | 20/20 (`0` errors) | 2,301.83 ms | 5,307.46 ms | 34,391.47 ms | 41,662.47 ms | Model warmup / cold KV cache spike (p99/max) |
| Remote Django Tool Execution (`/api/v1/tools/execute/` HTTP) | 20 sequential | 20/20 (`0` errors) | 191.62 ms | 389.24 ms | 400.51 ms | 403.33 ms | LAN Wi-Fi HTTP roundtrip |
| Remote Django Internal Query Time (`execution_time_ms`) | 20 sequential | 20/20 (`0` errors) | 3.00 ms | 18.60 ms | 42.92 ms | 49.00 ms | Local DB / fixture lookup |
| Full `ChatOrchestrator` Roundtrip (`LLM1 -> Django -> LLM2`) | 20 sequential | 20/20 (`0` fallbacks) | 20,267.44 ms | 24,406.30 ms | 25,983.95 ms | 26,378.37 ms | `final_response_generation` (~81% of turn time) |
| Local Hybrid RAG Search (warm embeddings, SQLite) | 20 sequential | 20/20 (`0` errors) | 8.57 ms | 8.95 ms | 9.12 ms | 9.16 ms | Negligible (in-process cosine + FTS) |
| Local `UniversalDataAccessService` Tool Concurrency | 20 concurrent sessions | 20/20 (`0` errors) | 59.32 ms | 137.70 ms | 157.52 ms | 162.48 ms | Async SQLite pool / serialization (`164.18 ms` total wall) |

---

## 2. Detailed Benchmark Measurements

### 2.1 Live LAN Benchmark (`benchmark_phase3a.py`, `20 x 3` runs)
- **Exit Code:** `0` (`0` errors, `0` fallbacks)
- **Benchmark 1 — Gemma 4 Single-Step Tool Selection (`20/20` tool calls emitted):**
  - `min`: `2,016.32 ms`
  - `p50`: `2,301.83 ms`
  - `mean`: `4,391.69 ms`
  - `p95`: `5,307.46 ms`
  - `p99`: `34,391.47 ms`
  - `max`: `41,662.47 ms`
- **Benchmark 2 — Remote Django `/api/v1/tools/execute/` (`20/20` HTTP 200 OK):**
  - Client HTTP roundtrip: `min=123.02 ms`, `p50=191.62 ms`, `mean=220.94 ms`, `p95=389.24 ms`, `p99=400.51 ms`, `max=403.33 ms`
  - Django internal `execution_time_ms`: `min=1.0 ms`, `p50=3.0 ms`, `mean=6.6 ms`, `p95=18.6 ms`, `p99=42.92 ms`, `max=49.0 ms`
- **Benchmark 3 — Full `ChatOrchestrator` Roundtrip (`LLM1 -> Remote Django -> LLM2`, `20/20` via `gemma4`, `0` fallbacks):**
  - `llm_tool_selection`: `min=2,281.07 ms`, `p50=3,525.43 ms`, `mean=3,415.23 ms`, `p95=4,323.75 ms`, `p99=4,637.72 ms`, `max=4,716.21 ms`
  - `remote_tool_query`: `min=63.62 ms`, `p50=157.10 ms`, `mean=324.68 ms`, `p95=1,156.06 ms`, `p99=1,344.71 ms`, `max=1,391.88 ms`
  - `final_response_generation`: `min=11,084.47 ms`, `p50=16,417.29 ms`, `mean=15,811.42 ms`, `p95=20,491.42 ms`, `p99=21,367.68 ms`, `max=21,586.74 ms`
  - `total_turn`: `min=13,451.79 ms`, `p50=20,267.44 ms`, `mean=19,551.96 ms`, `p95=24,406.30 ms`, `p99=25,983.95 ms`, `max=26,378.37 ms`

### 2.2 Local DB-Agnostic & RAG Concurrency Benchmark (`benchmark_stage11.py`)
- **Exit Code:** `0`
- **Hybrid RAG Search (`rag_hybrid_search_20x`, warm MiniLM embeddings on local SQLite):**
  - `min=8.42 ms`, `p50=8.57 ms`, `mean=8.63 ms`, `p95=8.95 ms`, `p99=9.12 ms`, `max=9.16 ms`
- **20 Concurrent `UniversalDataAccessService` Executions (`concurrent_20_local_uda_tools`):**
  - `20/20` OK
  - `min=23.80 ms`, `p50=59.32 ms`, `mean=73.66 ms`, `p95=137.70 ms`, `p99=157.52 ms`, `max=162.48 ms`
  - Total wall-clock time across all 20 concurrent executions: `164.18 ms`

---

## 3. Bottleneck Analysis & Production SLA Recommendations

1. **LLM Response Synthesis Is the Dominant Latency Driver (~81% of Total Turn Time)**
   - In live LAN execution, `remote_tool_query` adds only `157.10 ms` at p50 (`324.68 ms` mean) and `UniversalDataAccessService` local queries complete in `59.32 ms` p50 (`164.18 ms` wall for 20 concurrent calls).
   - Meanwhile, `llm_tool_selection` (`3,525.43 ms` p50) + `final_response_generation` (`16,417.29 ms` p50) drive the `20.27 s` p50 end-to-end turn latency.
   - **Recommendation for Voice IVR:** Keep filler/processing audio feedback enabled during tool+LLM waits, cap `max_tokens` for spoken responses, keep `ToolLoopRunner` within `max_execution_seconds`, and either use a faster quantized model (`gemma` 2B/4B or dedicated GPU inference with streaming TTS sentence chunking) to bring voice turn latency under a 3–5 s conversational SLA.
2. **Ollama Cold-Start / Tail Latency (`41.66 s` max on initial tool selection)**
   - Keep `keep_alive` enabled on Ollama and run a warm-up health probe on FastAPI startup to avoid cold model loads on the first customer call.
3. **Database & RAG Retrieval Headroom**
   - Local Hybrid RAG search (`8.57 ms` p50) and UDA query execution (`59.32 ms` p50 at 20x concurrency) leave substantial headroom for production traffic. External customer databases over WAN/VPN should be benchmarked separately under connection pooling (`maxPoolSize` / `pool_pre_ping`) with read-only statement timeouts configured per dialect.

