# Hybrid Voice Intelligence — Production & Telephone Test Scenarios

Structured verification matrix covering automated regression coverage and mandatory live SIP/telephone validation before production cutover.

---

## 1. Database, Migration & Tenant Isolation Scenarios

| ID | Scenario | Preconditions / Input | Expected Result | Mode |
| :--- | :--- | :--- | :--- | :--- |
| **DB-01** | Fresh Alembic 003 Upgrade & Rollback | PostgreSQL at `002_pgvector_support` | `alembic upgrade head` creates 5 `BIGINT` tables, all indexes, and rating check constraint (`1..5`); `alembic downgrade 002_pgvector_support` drops cleanly. | Automated / CLI |
| **DB-02** | Cross-Tenant `call_uuid` Isolation | `tenant-alpha` & `tenant-beta` write identical `call_uuid` | Repository queries return strictly scoped rows; `tenant-gamma` sees `0` rows. | Automated |
| **DB-03** | PostgreSQL Connection Loss During Call | Drop PG connection mid-conversation (`OperationalError`) | `IVRCallSession` completes STT, decision, AI reply, and PCM playback with zero caller-facing errors. | Automated |
| **DB-04** | Duplicate / Missing `call_uuid` Collision | Reconnect with `"unknown"` or reused `call_uuid` | Verify telemetry does not abort call and logs non-fatal conflict diagnostic (`uq_call_intelligence_turn`). | Automated / Fault |

---

## 2. Live Telephone Audio & Multilingual STT Scenarios

| ID | Scenario | Caller Condition / Utterance | Expected Result | Mode |
| :--- | :--- | :--- | :--- | :--- |
| **TEL-01** | Clean English Banking + Digits | Quiet room, `"My UPI transaction 984512 failed for 5000 rupees"` | Clean bypass (`SNR >= 24 dB`), accurate digits & `"UPI"`, domain reply (`ACTION=CONTINUE`). | Live Phone / MizoDroid |
| **TEL-02** | Clean Hindi Banking Flow | Press `2` -> `"मेरा यूपीआई रिफंड अभी तक नहीं आया"` | Correct Devanagari/Hinglish recognition `< 1.5s` latency; feminine Hindi reply (`कर रही हूँ`), native script audio. | Live Phone (Blocked until Hindi ASR tuned) |
| **TEL-03** | Clean Odia Banking Flow | Press `3` -> `"ମୋର ଭିଡିଓ କେୱାଇସି ସ୍ଥିତି କଣ ଅଛି?"` | Odia ASR recognizes intent cleanly without Tibetan/Gujarati token hallucinations. | Live Phone (Blocked until Odia ASR integrated) |
| **TEL-04** | Noisy Background (Fan / Traffic / Street) | Ambient noise (`12–16 dB SNR`) while stating card issue | Spectral/HPF enhancement cleans hum without corrupting acronyms (`UPI`, `KYC`, `PAN`); accurate intent. | Live Phone + Synthetic |
| **TEL-05** | Soft / Fast / Short Replies | Whispering `"yes"`, `"hello"`, or fast speech `< 0.5s` | 80ms preroll preserves leading consonants; crisp speech not dropped or stuck forever. | Live Phone |
| **TEL-06** | Pure Background Noise (No Speech) | Mute mic / static hum for `2–3s` (`SNR < 5 dB`) | Rejected by VAD/SNR gate or confidence gate; never hallucinates `"Thank you... Bye!"` and cuts call. | Live Phone + Synthetic |
| **TEL-07** | Voice & DTMF Barge-In | Interrupt AI mid-sentence by speaking or pressing `9` | Playback cancels within `40–80ms`, comfort silence frames sent cleanly, immediate transition or new utterance capture. | Live Phone |

---

## 3. Conversation Intelligence & Decision Policy Scenarios

| ID | Scenario | State & Caller Utterance | Expected Result | Mode |
| :--- | :--- | :--- | :--- | :--- |
| **INT-01** | Negative Emotion With Clear Request | Angry tone, clear audio: `"Check my debit card dispatch!"` | Emotion logged as uncertain indicator (`text=frustrated`); system resolves request (`Action.CONTINUE`), does NOT premature-transfer. | Automated + Live |
| **INT-02** | Low Audio / Garbled Utterance | Heavy static (`audio_quality_score < 0.35`, unintelligible words) | `DecisionManager` returns `Action.CLARIFY`; asks caller to repeat instead of hallucinating. | Automated + Live |
| **INT-03** | Repeated Clarification Escalation | 2 consecutive `Action.CLARIFY` turns | `DecisionManager` transitions to `Action.OFFER_TRANSFER` automatically. | Automated |
| **INT-04** | Explicit Human Agent Request Priority | Even on noisy line (`audio_quality < 0.35`): `"Connect me to an agent"` | Immediate `Action.TRANSFER` (priority 1) + `{"event": "transfer"}` JSON sent. | Automated + Live |
| **INT-05** | Resolution Confirmation Flow (Blocked Bug Fix) | `"unresolved"` state + caller says `"thank you"` | System asks confirmation (`Action.CONFIRM`) AND waits on call (`NO hangup` emitted); `"yes"` advances to `FEEDBACK` `1–5` rating. | Automated (`XFAIL` until fix) + Live |
| **INT-06** | Post-Resolution Feedback Gating | `IVRState.FEEDBACK` + caller says `"5"` or `"five"` | Rating `5` saved to `call_feedback`; closing thanks spoken; call cuts cleanly afterward. | Automated + Live |

---

## 4. Asterisk Transfer Lifecycle Scenarios

| ID | Scenario | Trigger & PBX Condition | Expected Result | Mode |
| :--- | :--- | :--- | :--- | :--- |
| **XFER-01** | Verbal Transfer -> Live SIP Agent | Caller asks for agent; Asterisk accepts `{"event": "transfer"}` | Caller leaves AI hold/prompt and hears ringing → bridged live SIP extension; `call_transfer` logged with masked summary. | Live Asterisk Test |
| **XFER-02** | Keypad `9` Transfer (Blocked Bug Fix) | Press `9` during `SERVICE_SELECTION` | Plays transfer notice, emits `{"event": "transfer"}` to Asterisk, records `call_transfer`. | Automated (`XFAIL`) + Live |
| **XFER-03** | Transfer Failed / No Agent Available | Asterisk returns `transfer_failed` / no agent online (or timeout) | Session receives ACK/timeout, exits `ESCALATED`, informs caller, and returns to `AI_CONVERSATION` rather than dead air. | Live Asterisk Test |

---

## 5. Performance, Timeouts & Resilience Scenarios

| ID | Scenario | Stress / Fault Condition | Acceptance Target | Mode |
| :--- | :--- | :--- | :--- | :--- |
| **PERF-01** | Cold-Start 5/10/20 Concurrent Calls | Concurrent first-turn calls with RAG + Voice Learning | Zero duplicate MiniLM weight reloads; turn latency `< 1.0s` (excluding external LLM/STT). | `benchmark_end_to_end_voice.py` |
| **PERF-02** | Concurrent Real Whisper Capacity | `5` concurrent CPU calls vs GPU/dedicated ASR worker | Maintain `< 1.5s` STT latency at target peak concurrency (move off shared CPU event-loop worker). | `benchmark_end_to_end_voice.py --real-stt` |
| **PERF-03** | Whisper / Gemma Timeout | STT or Gemma blocks/times out | Caller never experiences unhandled exceptions; Gemma timeout falls back to `ivr-rag-mini`. | Automated Fault Test |
| **PERF-04** | Abrupt WS / SIP Disconnect | Caller hangs up mid-AI playback | Playback task stops cleanly (`_is_playing=False`), pending tasks cancelled, call summary/telemetry finalized safely. | Automated Fault Test |

