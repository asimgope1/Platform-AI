# Production Readiness & Reliability Audit Report

**Audit Date:** 2026-11-11  
**Target Repository:** `ai-platform` (baseline commit `71cccde`)  
**Scope:** End-to-end Hybrid Voice Intelligence validation across telephony (`Asterisk` ↔ `IVRCallSession`), Audio Intelligence, STT (`whisper_local`), Conversation Intelligence, Decision Engine, RAG/LLM (`ChatOrchestrator` + `ivr-rag-mini`), TTS playback, and PostgreSQL/pgvector persistence.  
**Repository Mutation Policy:** No production modules were altered during this audit (`git HEAD` remains `71cccde`). Only audit documentation, load-benchmark tooling, and regression/reproducer tests were added pending fix review.

---

## Executive Verdict: **NO-GO for Unsupervised Live Production Calls**

The architecture demonstrates solid modularity, non-fatal database resilience, and deterministic policy layering in English unit/integration scenarios. However, **five production blockers** prevent unsupervised live banking IVR deployment today:

1. **Unverified & Unidirectional Asterisk Transfer Control (`Critical`):** External Asterisk server (`ws://192.168.1.133:9093`) is outside the repository; inbound transfer ACKs/failures (`transfer_failed`, `agent_unavailable`) are unhandled, `IVRState.ESCALATED` permanently locks caller microphone input with no timeout/return-to-IVR, and keypad `9` never emits `{"event": "transfer"}`.
2. **Premature Auto-Hangup During Resolution Confirmation (`Critical`):** Saying `"thank you"` while an issue is `"unresolved"` triggers `DecisionManager` `Action.CONFIRM`, yet `IVRCallSession._finalize_and_speak` immediately sends WebSocket `hangup` commands `0.8s` after asking for confirmation.
3. **Odia (`or`) Speech Recognition Failure & Hindi Script Mismatch (`Critical`):** `openai/whisper-base` (the configured `speech_to_text.default`) forces `language="hi"` for Odia, resulting in `100%` WER and `97.3%–100%` CER (Tibetan/Gujarati token hallucinations) at `7.7s–9.4s` CPU latency per turn. Even Hindi cached speech showed `57.1%–95.0%` WER on `whisper-base`.
4. **Concurrency Cold-Start Stampede & Event-Loop Blocking (`High`):** Missing double-checked locking in `SentenceTransformersEmbeddingAdapter._get_model()` caused 5-call synthetic latency `p99` to spike to `7.82s` (`916 MB` peak RSS at 20 calls). On CPU, 5 concurrent real `whisper-base` calls serialized at `p50=7.35s/turn`. Furthermore, synchronous `_tts_output_to_pcm8k` resampling inside `IVRCallSession` blocks the asyncio loop used for `20ms` frame streaming.
5. **Migration `003` vs `init_db()` Schema Drift (`High`):** Running FastAPI startup `init_db()` before `alembic upgrade head` pre-creates the 5 intelligence tables with `INTEGER` PKs instead of `BIGINT` (`003`), causing `003`'s `if table not in existing` guard to silently skip table creation.

---

## Measurement Provenance & Separation

| Evidence Category | What Was Executed | What Was NOT Executed |
| :--- | :--- | :--- |
| **Static Code Trace** | Full line-by-line audit of `telephony/`, `audio_intelligence/`, `conversation_intelligence/`, `decision_engine/`, `providers/`, `storage/`, `api/routes/`, `services/api/`. | External Asterisk server codebase at `ws://192.168.1.133:9093`. |
| **Live PostgreSQL Verification** | Executed `alembic downgrade 002_pgvector_support` → `alembic upgrade head` (`003_voice_intelligence_tables`), schema/index/constraint inspection, multi-tenant write/read isolation, RAG cross-tenant test, and `--with-db` concurrent write benchmark against `postgresql+asyncpg://localhost:5432/ai_platform_test`. | Production connection pool soak (>1h) on remote RDS/cloud PG. |
| **Local Whisper STT Benchmark** | Real `WhisperLocalSpeechToTextAdapter` (`openai/whisper-base` on CPU) over cached neural TTS `.pcm` prompts (`en`, `hi`, `or`) under clean and noise-degraded (`noise_std=1050`, 60Hz `hum=900`) conditions, plus 1-call & 5-call real Whisper concurrency runs. | GPU Whisper / `whisper-large-v3` or Indic-finetuned ASR models. |
| **Synthetic E2E Load & Fault Benchmark** | Real `AudioIntelligencePipeline`, `TelephonyIntelligencePipeline`, `DecisionManager`, `IVRCallSession`, `SentenceTransformer`, and `IVRRAGMiniModel` at `1, 5, 10, 20` concurrent calls + timeout/disconnect fault injection via `scripts/benchmark_end_to_end_voice.py`. | Real PSTN/SIP trunk RTP jitter, packet loss, and live speaker barge-in over MizoDroid/Asterisk. |

---

## 1. End-to-End Architecture Audit

**Actual Runtime Flow:**
```
Asterisk PBX WS (ws://192.168.1.133:9093) ↔ PBXConnector ↔ IVRCallSession
  ├─ on_pcm_frame (20ms s16le 8kHz) ─► VAD + 80ms preroll ─► _pcm_buffer
  │    └─ 0.9s silence gap or 8s cap ─► _flush_utterance() (under _pipeline_lock)
  │         ├─ 1. AudioIntelligencePipeline.analyze/enhance (to_thread; clean bypass ≥24dB)
  │         ├─ 2. AudioService.transcribe ─► WhisperLocalSpeechToTextAdapter (to_thread CPU)
  │         ├─ 3. Pre-AI Turn Intel: TranscriptAnalyzer + ConversationAnalyzer + DecisionManager
  │         ├─ 4. Branch: TRANSFER | CLARIFY/OFFER_TRANSFER/CONFIRM/FEEDBACK | RAG + ChatOrchestrator (fallback: ivr_mini_brain)
  │         ├─ 5. AudioService.synthesize ─► edge-tts / VITS / say ─► _tts_output_to_pcm8k() [SYNC]
  │         └─ 6. _send_pcm_reply() (20ms chunking w/ 100ms prebuffer) + post-reply turn persistence + voice learning
```

### Architecture Findings
* **Duplicate Heuristic Evaluation:** [telephony/ivr_session.py](file:///Users/asimgope/.gemini/antigravity-ide/scratch/ai-platform/telephony/ivr_session.py) and [conversation_intelligence/intent_analyzer.py](file:///Users/asimgope/.gemini/antigravity-ide/scratch/ai-platform/conversation_intelligence/intent_analyzer.py) detect closing, agent, greeting, and domain intents before [providers/chat/ivr_mini_model.py](file:///Users/asimgope/.gemini/antigravity-ide/scratch/ai-platform/providers/chat/ivr_mini_model.py) re-evaluates overlapping regexes during fallback. No circular import exists (`TYPE_CHECKING` guards are respected).
* **Duplicate VAD Pass:** `AudioQualityAnalyzer` computes frame VAD probabilities, and [audio_intelligence/speech_segmenter.py](file:///Users/asimgope/.gemini/antigravity-ide/scratch/ai-platform/audio_intelligence/speech_segmenter.py) instantiates and runs `AdaptiveTelephonyVAD` a second time on the same buffer.
* **Event-Loop Blocking:** `_tts_output_to_pcm8k()` (`scipy.signal.resample` / subprocess `afconvert`) executes synchronously inside async `_finalize_and_speak()` and `_play_prompt()` ([telephony/ivr_session.py](file:///Users/asimgope/.gemini/antigravity-ide/scratch/ai-platform/telephony/ivr_session.py#L947-L965)), stalling `20ms` pacing of all other concurrent calls on the same worker loop.
* **Acoustic Echo Cancellation (AEC) Unwired:** `AudioIntelligencePipeline` defaults `echo_cancellation_enabled` to `"false"`, and `IVRCallSession` never captures or passes far-end TTS playback frames as `reference_pcm`.
* **Session Restart & Shutdown Leaks:** `IVRCallSession.on_call_start()` does not cancel stale `_playback_task`, `_silence_task`, or `_menu_timeout_task` if invoked on reconnect, and `IntelligenceEventHandler` has no explicit drain/shutdown hook on `on_call_end()`.

---

## 2. Real Asterisk Transfer Verification

* **What Works:** When `DecisionManager` returns `Action.TRANSFER` (e.g., caller asks for a human agent verbally), `IVRCallSession._finalize_and_speak()` emits `{"event": "transfer", "uuid": ..., "reason": ..., "summary": ...}`, sets `menu_machine.state = IVRState.ESCALATED`, and queues a `CallTransfer` telemetry write.
* **Blocking Gaps:**
  1. **Asterisk Server Outside Repo:** Whether `ws://192.168.1.133:9093` parses `{"event": "transfer"}` and bridges a live SIP extension is **Not Verified** from this repository.
  2. **Keypad `9` Transfer Broken:** `IVRCallSession.on_dtmf("9")` (`Action == "HUMAN_TRANSFER"`, lines 290–293) calls `_play_transfer_notice()` only — it **never emits** `{"event": "transfer"}` and never logs `CallTransfer`.
  3. **Hardcoded Agent Availability:** `TransferPolicy(agent_available=True)` has no PBX queue/presence hook.
  4. **Zero Transfer ACK / Failure / Return-to-IVR Handling:** [telephony/pbx_connector.py](file:///Users/asimgope/.gemini/antigravity-ide/scratch/ai-platform/telephony/pbx_connector.py) handles only `call_start`, `transcript`, `dtmf`/`keypad`, `call_end`, and `hello`. Any `transfer_failed` or `agent_unavailable` event from Asterisk is dropped, while `IVRState.ESCALATED` permanently ignores caller audio with no timeout recovery back to `AI_CONVERSATION`.
  5. **Route Parity Drift:** [api/routes/ivr_websocket.py](file:///Users/asimgope/.gemini/antigravity-ide/scratch/ai-platform/api/routes/ivr_websocket.py) `/ws/ivr` omits `dtmf` text event handling entirely (present only in `pbx_connector.py`).

---

## 3. Speech Recognition Evaluation (`openai/whisper-base` CPU, 8 kHz)

Evaluated using identical `WhisperLocalSpeechToTextAdapter` settings (`openai/whisper-base`, configured default in [config/providers.yaml](file:///Users/asimgope/.gemini/antigravity-ide/scratch/ai-platform/config/providers.yaml)) across cached neural speech prompts under Clean vs Noise-Degraded (`noise_std=1050.0`, 60 Hz `hum_amp=900.0`) conditions:

| Language | Prompt Key | Clean SNR | Orig Clean WER / CER | Enh Clean WER / CER | Noisy SNR | Orig Noisy WER / CER | Enh Noisy WER / CER | Enh STT Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `en` | `human_transfer` | `29.8 dB` | `0.200 / 0.066` | `0.200 / 0.066` | `15.1 dB` | `0.200 / 0.066` | `0.200 / 0.066` | `1,158 ms` |
| `en` | `upi_ack` | `35.8 dB` | `0.125 / 0.022` | `0.125 / 0.022` | `16.2 dB` | `0.125 / 0.043` | **`0.250 / 0.076`** | `1,489 ms` |
| `en` | `invalid_input` | `34.5 dB` | `0.462 / 0.329` | `0.462 / 0.329` | `16.4 dB` | `0.462 / 0.329` | `0.462 / 0.329` | `997 ms` |
| `hi` | `human_transfer` | `35.4 dB` | `0.900 / 0.471` | `0.900 / 0.471` | `14.0 dB` | `0.900 / 0.500` | `0.850 / 0.481` | `6,136 ms` |
| `hi` | `upi_ack` | `45.0 dB` | `0.800 / 0.462` | `0.800 / 0.433` | `15.1 dB` | `0.950 / 0.510` | `0.900 / 0.519` | `5,330 ms` |
| `hi` | `timeout` | `45.0 dB` | `0.714 / 0.486` | `0.571 / 0.357` | `14.7 dB` | `0.857 / 0.571` | `0.714 / 0.543` | `4,612 ms` |
| `or` | `human_transfer` | `45.0 dB` | `1.000 / 1.000` | `1.000 / 0.973` | `18.3 dB` | `1.000 / 1.000` | `1.000 / 1.000` | `8,260 ms` |
| `or` | `upi_ack` | `45.0 dB` | `1.000 / 1.000` | `1.000 / 1.000` | `16.7 dB` | `1.000 / 1.000` | `1.000 / 1.000` | `9,074 ms` |
| `or` | `invalid_input` | `45.0 dB` | `1.000 / 0.985` | `1.000 / 1.000` | `16.8 dB` | `1.000 / 1.000` | `1.000 / 1.000` | `7,729 ms` |

### Key STT Audit Observations
* **Odia Decoding Failure:** [whisper_local.py](file:///Users/asimgope/.gemini/antigravity-ide/scratch/ai-platform/providers/speech_to_text/whisper_local.py) maps `or` → `gen_kwargs["language"] = "hi"`. `whisper-base` emitted Gujarati/Tibetan gibberish (`૯ૹ૲...` / `༽༼༻...`), failing `TranscriptAnalyzer` script verification every turn while burning `7.7s–9.4s` CPU time.
* **Spectral Distortion on Acronyms:** At `16.2 dB` SNR, spectral suppression turned `"UPI"` into `"UCI"` (`WER 0.125 → 0.250`). Conversely, it improved Hindi `timeout` (`0.714 → 0.571`).
* **Pure Noise Hallucination Hazard:** When fed low-SNR noise + 60 Hz hum without speech (`SNR=0.9 dB`), `whisper-base` hallucinated `"Thank you for your support... Bye!"` — which passes `is_closing_turn()` and can drop a noisy call.
* **Missing Confidence Diagnostics:** `WhisperLocalSpeechToTextAdapter` uses HF `pipeline("automatic-speech-recognition")` and never populates `avg_logprob` or `no_speech_prob` in `diagnostics`, leaving `TranscriptAnalyzer` anchored to its `0.78` static prior.

---

## 4. Conversation Intelligence & Decision Policy Evaluation

Verified both via architectural inspection and automated tests (`tests/audit/test_production_audit_regressions.py`):
* **Working Correctly:**
  * Frustration/sadness alone does NOT force escalation when intent is clear and quality is high (`Action.CONTINUE`).
  * Low audio quality (`< 0.35`) triggers `Action.CLARIFY` (and offers transfer after `>= 2` clarifications).
  * `explicit_agent_request=True` overrides low audio/transcript confidence and returns priority-1 `Action.TRANSFER`.
* **Critical Logic Defect (`CRITICAL-2`):**
  * When `resolution_status` is `"unresolved"`, saying `"thank you"` causes `DecisionManager` to ask confirmation (`Action.CONFIRM`, sets `awaiting_resolution_confirmation=True`). However, `IVRCallSession._finalize_and_speak()` only exempts `IVRState.FEEDBACK and self.conv_state.awaiting_feedback` from `is_closing_turn()` (`telephony/ivr_session.py#L948`), so `"thank"` triggers immediate WebSocket `hangup` `0.8s` later!

---

## 5. PostgreSQL & Tenant Isolation Verification

* **Clean Migration Verification:** On `postgresql+asyncpg://localhost:5432/ai_platform_test`, executed full `alembic downgrade 002_pgvector_support` → `alembic upgrade head` (`003_voice_intelligence_tables`). Confirmed all 5 tables (`call_intelligence`, `call_events`, `call_feedback`, `call_transfer`, `conversation_resolution`) exist with `BIGINT` primary keys, all documented indexes, and `call_feedback_rating_check (rating >= 1 AND rating <= 5)`.
* **Pre-Audit Migration Drift Discovered:** `services/api/lifespan.py` calls `init_db()` on boot using ORM models typed with `Integer` PKs; `003` skips existing tables (`if table not in existing`), leaving `INTEGER` schemas in databases initialized via `init_db()`. Additionally, `storage/migrations/env.py` lacks imports for the 5 new models, and no `ForeignKey` enforces referential integrity against `tenants.id`.
* **Tenant Isolation & Fault Tolerance:**
  * Cross-tenant live PG read/write check confirmed strict separation between `tenant-alpha`, `tenant-beta`, and `tenant-gamma` on identical `call_uuid='audit-shared-call-001'`.
  * Cross-tenant RAG suite (`tests/integration/test_rag_and_cross_tenant_paths.py`) passed `3/3`.
  * Simulated `OperationalError` across both `async_session` paths confirmed `IVRCallSession` continues serving calls cleanly without raising.
* **Collision Edge Case:** `IntelligenceRepository.record_turn()` enforces `UniqueConstraint("tenant_id", "call_uuid", "turn_number")`; any `unknown` or reused `call_uuid` after a PBX reconnect silently drops subsequent telemetry rows.

---

## 6. Load, Concurrency & Reliability Benchmarks

Executed via [benchmark_end_to_end_voice.py](file:///Users/asimgope/.gemini/antigravity-ide/scratch/ai-platform/scripts/benchmark_end_to_end_voice.py):

### A. Synthetic STT/TTS + Real Audio/Conv/Decision/MiniLM (`3 turns/call`)
| Concurrent Calls | Total Turns | Wall (ms) | Throughput (turns/s) | p50 (ms) | p95 (ms) | p99 (ms) | CPU User+Sys (s) | Peak RSS (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 3 | `276.4` | `10.85` | `84.4` | `106.4` | `106.4` | `0.30` | `289.7` |
| 5 | 15 | `8,894.9` | `1.69` | `846.1` | `7,442.6` | `7,816.1` | `9.04` | `665.1` |
| 10 | 30 | `5,671.6` | `5.29` | `1,652.7` | `2,272.1` | `2,354.0` | `8.24` | `758.3` |
| 20 | 60 | `13,186.4` | `4.55` | `3,878.5` | `4,980.1` | `5,599.4` | `17.89` | `916.7` |

> **Root Cause of 5-call `7.82s` Spike:** [providers/embeddings/sentence_transformers.py](file:///Users/asimgope/.gemini/antigravity-ide/scratch/ai-platform/providers/embeddings/sentence_transformers.py#L23-L37) omits a double-check inside `_GLOBAL_LOCK`, causing concurrent background `learn_from_voice_turn` / RAG tasks on cold start to load `all-MiniLM-L6-v2` weights sequentially one thread after another.

### B. Live PostgreSQL + pgvector RAG + Intelligence Persistence (`--with-db`, warm weights, `1 turn/call`)
| Concurrent Calls | Total Turns | p50 (ms) | p95 (ms) | Throughput (turns/s) | Peak RSS (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 1 | `144.9` | `144.9` | `5.00` | `434.7` |
| 5 | 5 | `527.4` | `555.2` | `7.89` | `434.7` |

### C. Real Local CPU Whisper STT (`--real-stt`, `openai/whisper-base`, `1 turn/call`)
| Concurrent Calls | Total Turns | p50 (ms) | p95 (ms) | Throughput (turns/s) | CPU User+Sys (s) | Peak RSS (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 1 | `1,523.3` | `1,523.3` | `0.66` | `2.88` | `1,080.2` |
| 5 | 5 | `7,349.9` | `7,388.8` | `0.67` | `31.34` | `1,646.1` |

* **Fault-Injection Outcomes:** `stt_timeout_nonfatal=True`, `llm_timeout_fallback_replied=True`, `ws_disconnect_handled_gracefully=True`.

---

## 7. Prioritized Findings Register & Proposed Fixes

| ID | Severity | Component / File | Verified Issue | Proposed Fix |
| :--- | :--- | :--- | :--- | :--- |
| **CRITICAL-1a** | Critical | `telephony/pbx_connector.py`, `telephony/ivr_session.py` | No inbound Asterisk transfer ACK/failure events; `IVRState.ESCALATED` permanently ignores caller audio; `agent_available=True` is hardcoded. | Handle `transfer_accepted` / `transfer_failed` / `agent_unavailable` in `pbx_connector.py`; add a transfer timeout timer to return caller to `AI_CONVERSATION` gracefully. Verify Asterisk WS handler on `192.168.1.133:9093`. |
| **CRITICAL-1b** | Critical | [telephony/ivr_session.py](file:///Users/asimgope/.gemini/antigravity-ide/scratch/ai-platform/telephony/ivr_session.py#L290-L293) | Pressing DTMF `9` (`HUMAN_TRANSFER`) plays notice only without emitting `{"event": "transfer"}` or logging `CallTransfer`. | Emit structured transfer event after transfer notice and record `CallTransfer` telemetry cleanly. |
| **CRITICAL-2** | Critical | [telephony/ivr_session.py](file:///Users/asimgope/.gemini/antigravity-ide/scratch/ai-platform/telephony/ivr_session.py#L947-L965) | `"thank you"` triggers `Action.CONFIRM` (`awaiting_resolution_confirmation=True`), then `_finalize_and_speak` immediately sends WebSocket `hangup` `0.8s` later. | Exempt `self.conv_state.awaiting_resolution_confirmation` (and any non-`CONTINUE`/`FEEDBACK` decision action) from `is_closing_turn` auto-hangup. |
| **CRITICAL-3** | Critical | [whisper_local.py](file:///Users/asimgope/.gemini/antigravity-ide/scratch/ai-platform/providers/speech_to_text/whisper_local.py) | Odia (`or`) forces `language="hi"` on `whisper-base`, yielding `100%` WER and `7.7s–9.4s` CPU stalls; Hindi yields `57%–95%` WER at `3.6s–6.1s`. | Route `or`/`hi` to an Indic-capable ASR backend (e.g. IndicWhisper/AI4Bharat or Azure/Google/Sarvam STT) or GPU `whisper-large-v3` behind provider registry; reject empty/hallucinated transcripts before `is_closing_turn`. |
| **HIGH-1** | High | [sentence_transformers.py](file:///Users/asimgope/.gemini/antigravity-ide/scratch/ai-platform/providers/embeddings/sentence_transformers.py#L23-L37), `lifespan.py` | Missing double-checked locking inside `_GLOBAL_LOCK` causes multi-thread `all-MiniLM-L6-v2` weight reload storm on cold concurrent calls (`7.8s` spike). | Add `if self.model_name not in self._GLOBAL_MODELS:` inside `with self._GLOBAL_LOCK:`, lock `_get_fallback_adapter()`, and pre-warm embedding adapter in `lifespan.py`. |
| **HIGH-2** | High | [telephony/ivr_session.py](file:///Users/asimgope/.gemini/antigravity-ide/scratch/ai-platform/telephony/ivr_session.py#L177-L249) | `_tts_output_to_pcm8k` runs synchronous `scipy`/subprocess resampling on event loop; `on_call_start` doesn't cancel stale tasks; `on_call_end` doesn't drain telemetry worker. | Move `_tts_output_to_pcm8k` to `asyncio.to_thread`; cancel existing tasks in `on_call_start`; expose `await handler.drain()` cleanly. |
| **HIGH-3** | High | [003_voice_intelligence_tables.py](file:///Users/asimgope/.gemini/antigravity-ide/scratch/ai-platform/storage/migrations/versions/003_voice_intelligence_tables.py), ORM models, `env.py` | `init_db()` (`Integer`) vs `003` (`BigInteger`) drift; `env.py` omits new models; missing FKs to `tenants.id`. | Align ORM PK types to `BigInteger`, import all 5 models in `storage/migrations/env.py`, and disable implicit DDL `create_all` in production mode (`alembic` sole schema authority). |
| **MEDIUM-1** | Medium | [whisper_local.py](file:///Users/asimgope/.gemini/antigravity-ide/scratch/ai-platform/providers/speech_to_text/whisper_local.py#L297-L309), `noise_suppressor.py` | Missing `avg_logprob` / `no_speech_prob` forces fallback prior `0.78`; spectral suppression distorted `"UPI"`→`"UCI"`. | Use `whisper` model `generate(..., return_dict_in_generate=True, output_scores=True)` or `faster-whisper` to extract real logprobs/no_speech probs; tune spectral floor upward or train DeepFilterNet. |
| **MEDIUM-2** | Medium | [speech_segmenter.py](file:///Users/asimgope/.gemini/antigravity-ide/scratch/ai-platform/audio_intelligence/speech_segmenter.py), `telephony/pcm_utils.py`, `api/routes/ivr_websocket.py` | Double VAD execution; `audio_bytes_to_pcm8k` hardcodes `.mp3` suffix on `afconvert`; `/ws/ivr` ignores `dtmf`; `_run_stt_ai_tts` drops `reference_pcm` AEC parameter. | Reuse VAD frame flags from `AudioQualityAnalyzer`; fix container extension detection; wire `dtmf` in `/ws/ivr`; buffer recent TX frames for optional AEC. |
| **LOW-1** | Low | [ivr_mini_model.py](file:///Users/asimgope/.gemini/antigravity-ide/scratch/ai-platform/providers/chat/ivr_mini_model.py), [intelligence_repository.py](file:///Users/asimgope/.gemini/antigravity-ide/scratch/ai-platform/storage/repositories/intelligence_repository.py) | Overlapping regex detection; repository lacks tenant-scoped read methods for `CallEvent`, `CallTransfer`, `ConversationResolution`. | Pass resolved `IntentResult` cleanly into fallback; add tenant-filtered list helpers to `IntelligenceRepository`. |

