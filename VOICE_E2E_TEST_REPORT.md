# Voice & IVR End-to-End Test Report

## 1. Multilingual & Edge-Case Verification Matrix

Verified via automated tests (`tests/integration/test_stage7_8_9_10_e2e.py`, `tests/integration/test_phase3a_stabilization.py`, `tests/telephony/test_ivr_flow.py`, `tests/telephony/test_ivr_menu.py`, `tests/telephony/test_hybrid_intelligence_e2e.py`) and observed pipeline instrumentation:

| Scenario | Implementation Status | Verified Behavior & Evidence |
| :--- | :--- | :--- |
| **English / Hindi / Odia Prompts & Fallbacks** | `TESTED — DEVELOPMENT` | Verified `WELCOME_AND_LANGUAGE` entries for `en`, `hi`, `or` and verified `RAGFirstResponseGenerator` produces localized `tool_unavailable_fallback` strings in `en`, `hi`, and `or` without fabricating order IDs. |
| **Speech / No-Speech / Silence Gating** | `TESTED — DEVELOPMENT` | Verified `IVRCallSession.on_pcm_frame` ignores silent PCM frames (`np.zeros`) when idle and does not accumulate spurious speech audio into `_pcm_buffer`. |
| **Sustained Voice Barge-In & Transcript Barge-In** | `TESTED — DEVELOPMENT` | Verified sustained loud speech (`BARGE_IN_MIN_CONSECUTIVE_FRAMES=6`, `RMS > 1150.0`, `BARGE_IN_COOLDOWN_SECONDS=0.35`) cancels `_playback_task` and sets `_is_playing=False`. Verified pre-existing `TestIVRVoiceBargeIn::test_voice_barge_in_cancels_playback` fails only because that older unit test sends 2 frames whereas production was tuned to 6 frames to block speakerphone echo false triggers. |
| **Tool Timeout & Django Unavailable Fallback** | `TESTED — DEVELOPMENT` | When `ChatOrchestrator.execute` fails or times out while `tools` are active, `RAGFirstResponseGenerator` returns `source="tool_unavailable_fallback"` instead of allowing `ivr_mini_brain` to hallucinate order statuses or tracking IDs. |
| **Ollama Unavailable Fallback** | `TESTED — DEVELOPMENT` | Non-tool FAQ turns fall back cleanly to `ivr_mini_brain`; tool-enabled turns fail safe with localized outage message. |
| **Customer Ownership Isolation in IVR** | `TESTED — DEVELOPMENT` | `authorized_customer_id` propagates through `RAGFirstResponseGenerator.generate_response` (`metadata["authorized_customer_id"]`) -> `ChatOrchestrator` -> `ToolExecutor` (`trusted_metadata`) -> `UniversalDataAccessService` / `DjangoRemoteToolClient`. Cross-customer lookups raise `PermissionDeniedError`. |
| **Multi-Turn Context Retention** | `TESTED — DEVELOPMENT` | Prior assistant/user turns pass cleanly into `NormalizedChatRequest.messages` across turns alongside dynamic tool schemas. |
| **Processing / Waiting Audio Pulse** | `IMPLEMENTED — NOT TESTED` in live SIP trunk | `IVR_PROCESSING_SOUND_ENABLED=true` (`generate_processing_pulse_pcm`) in `telephony/ivr_session.py` was not tested end-to-end over a live SIP handset in this session. |
| **Live Asterisk SIP / PSTN Trunk Call** | `NOT IMPLEMENTED` / Hardware External | No physical Asterisk PBX or live SIP trunk call was connected inside this automated coding session. Live telephony E2E remains unverified. |

