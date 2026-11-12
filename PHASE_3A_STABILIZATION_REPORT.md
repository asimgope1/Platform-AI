# Phase 3A Stabilization & Live Latency Benchmark Report

## 1. Stabilization Fixes Implemented & Verified
1. **Remote Registry Lifecycle & Stale Recovery (`tools/remote/registry_sync.py`, `tools/registry.py`)**:
   - Tracks `synced`, `stale`, `error`, and active synced tool names in `get_remote_sync_state()`.
   - Preserves previous tool snapshots safely across failed remote catalog refreshes while marking state `stale=True`.
   - Prunes removed remote tools cleanly upon successful refresh via `tool_registry.unregister(...)`.
   - Startup `lifespan.py` sync tolerates an offline Windows Django backend without blocking FastAPI API readiness.
2. **Per-Key In-Flight Concurrency Coalescing (`tools/remote/django_client.py`)**:
   - Guarded remote execution with `self._inflight_locks[stable_key]` (`asyncio.Lock`) alongside TTL deduplication cache (`_dedup_cache`), ensuring concurrent retries/calls with the same `idempotency_key` coalesce and prevent duplicate remote side effects.
3. **Orchestrator & Voice Tool-Safety Fallbacks (`orchestration/chat_orchestrator.py`, `services/conversation/response_generator.py`, `telephony/ivr_session.py`)**:
   - Refuses silent fallback to non-tool `ivr-rag-mini` when `current_request.tools` is present.
   - Emits multilingual `tool_unavailable_fallback` in `en`, `hi`, and `or` when remote business tools or Gemma fail — preventing fabricated order numbers or delivery ETAs.
   - Opt-in IVR tool wiring activates only when `remote_django` tools are actively synced in `tool_registry` and propagates `authorized_customer_id`.
   - `ToolLoopRunner` (`orchestration/tool_loop.py`) deduplicates identical `(func_name, canonical_args)` invocations within a single turn loop via `turn_exec_cache`.

---

## 2. Live LAN Benchmark Measurements (`task-6969`, MacBook → Windows `192.168.1.141`)

All 3 benchmarks executed **20 sequential iterations** over LAN Wi-Fi against `gemma4:latest` (`http://192.168.1.141:11434/v1` with `reasoning_effort: "none"`) and Windows Django (`http://192.168.1.141:8000`) using seeded fixtures (`ORD-1001`, `ORD-2005`):

### Benchmark 1: Ollama `gemma4` Single-Step Tool Selection (20/20 tool calls emitted, 0 errors)
| Metric | Value (ms) |
| :--- | :---: |
| Min | `2016.32` |
| p50 (Median) | `2301.83` |
| Mean | `4391.69` |
| p95 | `5307.46` |
| p99 | `34391.47` |
| Max (cold model warmup spike) | `41662.47` |

### Benchmark 2: Windows Django Direct Tool Execution `/api/v1/tools/execute/` (20/20 succeeded, 0 errors)
| Metric | Client HTTP Roundtrip (ms) | Django Internal `execution_time_ms` (ms) |
| :--- | :---: | :---: |
| Min | `123.02` | `1.00` |
| p50 (Median) | `191.62` | `3.00` |
| Mean | `220.94` | `6.60` |
| p95 | `389.24` | `18.60` |
| p99 | `400.51` | `42.92` |
| Max | `403.33` | `49.00` |

### Benchmark 3: Full `ChatOrchestrator` End-to-End Turn (`LLM1 -> Remote Django -> LLM2`)
- **Result**: `20/20` turns completed cleanly via `gemma4`, **`0` fallbacks**, **`0` errors**.
- **Breakdown**:

| Stage | Min (ms) | p50 (ms) | Mean (ms) | p95 (ms) | p99 (ms) | Max (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| LLM Tool Selection (`LLM1`) | `2281.07` | `3525.43` | `3415.23` | `4323.75` | `4637.72` | `4716.21` |
| Remote Django Tool Query | `63.62` | `157.10` | `324.68` | `1156.06` | `1344.71` | `1391.88` |
| Final Response Generation (`LLM2`) | `11084.47` | `16417.29` | `15811.42` | `20491.42` | `21367.68` | `21586.74` |
| **Total Turn End-to-End** | **`13451.79`** | **`20267.44`** | **`19551.96`** | **`24406.30`** | **`25983.95`** | **`26378.37`** |

### Operational Bottleneck Finding
- Remote Django DB execution itself is sub-50 ms (`3.0 ms` p50, `191.62 ms` HTTP LAN roundtrip).
- **Primary latency bottleneck**: Ollama `gemma4:latest` second-pass natural-language synthesis (`16.42 s` p50), pushing full turn p50 to `20.27 s`. Live telephony calls using `gemma4` require either persistent IVR waiting/processing audio (`IVR_PROCESSING_SOUND_ENABLED=true`) or faster GPU inference / streaming TTS synthesis.

---

## 3. Readiness Status
- **Phase 3A Stabilization & LAN Integration**: `VERIFIED — STAGING` (seeded Windows test fixtures, LAN Wi-Fi).

