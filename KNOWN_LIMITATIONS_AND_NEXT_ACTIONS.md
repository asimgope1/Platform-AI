# Known Limitations & Prioritized Next Actions

## 1. Confirmed External Blockers (Cannot Be Resolved From MacBook Workspace Alone)

1. **Windows Django Business Service Source Code & Server-Side Auth (`Stage 2`)**:
   - **Blocker**: The Django repository serving `http://192.168.1.141:8000` is hosted on a separate Windows machine not mounted in `/Users/asimgope/.gemini/antigravity-ide/scratch`.
   - **Current Exposure**: Unauthenticated HTTP access was observed on Windows dev port `8000` during earlier LAN probing.
   - **Mitigation Delivered**: Created and unit-tested [`tools/remote/django_server_auth.py`](file:///Users/asimgope/.gemini/antigravity-ide/scratch/ai-platform/tools/remote/django_server_auth.py) implementing matching API key / Bearer, HMAC-SHA256 signature, clock-skew, and nonce-replay verification.
   - **Next Action**: Deploy this middleware on Windows, set matching `REMOTE_TOOLS_API_KEY` + `REMOTE_TOOLS_HMAC_SECRET`, flip `REMOTE_TOOLS_REQUIRE_AUTH=true`, and terminate TLS before any staging/production exposure.

2. **PostgreSQL + `pgvector` Live Staging Validation**:
   - **Blocker**: Local PostgreSQL (`::1:5432`) is blocked by sandbox network rules during test runs in this environment (`test_postgres_pgvector_rag.py` connection refused/blocked).
   - **Current Status**: Native SQL queries and ORM mappings are implemented in `DocumentRepository` & `001–005` migrations, while automated runtime checks executed on SQLite fallback.
   - **Next Action**: Run `alembic upgrade head` and `pytest tests/integration/test_postgres_pgvector_rag.py` in a staging environment with live PostgreSQL + `pgvector`.

3. **External SQL / NoSQL Database Drivers & Live DB Instances**:
   - **Status**: `psycopg2`, `pymysql`, `pyodbc`, `oracledb`, and `pymongo` are not installed in the MacBook `.venv` and no live external PostgreSQL/MySQL/MSSQL/Oracle/MongoDB servers are present locally.
   - **Next Action**: Install target drivers per deployment (`pip install psycopg2-binary pymysql pyodbc oracledb pymongo`) and run live read-only onboarding tests against staging customer DB replicas.

4. **Live Asterisk PBX / SIP Trunk Call & Offline Edge-TTS**:
   - **Status**: No live Asterisk SIP call was placed; `test_audio_endpoints` failed inside the offline sandbox because `speech.platform.bing.com:443` requires external internet access for Edge-TTS neural voices (returning silent WAV fallback instead of `audio/mpeg`).
   - **Next Action**: Execute a live SIP call through Asterisk WebSocket (`IVRCallSession`) after TLS & Django auth deployment to verify real handset barge-in, processing pulse, and end-to-end speech turn latency.

---

## 2. Pre-Existing Unit Test Mismatches (Untouched by Phase 3A/3B)
- `tests/unit/test_ivr_mini_model.py::TestIVRRAGMiniModel::test_emotion_frustration_empathy`: Tests for legacy phrase `"understand your concern"` / `"don't worry"`, whereas `IVRRAGMiniModel` now outputs `"I completely understand your frustration, and I'm right here to help sort this out."`.
- `tests/unit/test_ivr_voice_barge_in.py::TestIVRVoiceBargeIn::test_voice_barge_in_cancels_playback`: Sends 2 loud frames, whereas `telephony/ivr_session.py` (`BARGE_IN_MIN_CONSECUTIVE_FRAMES = 6`) requires 6 consecutive frames (~120 ms) to suppress speakerphone echo false triggers.
- `tests/integration/test_django_comone_telephony.py`: Requires `DJANGO_SETTINGS_MODULE=comone.settings` when invoked directly.

---

## 3. Latency & Performance Recommendations
- Measured live LAN full turn p50 latency using `gemma4:latest` on `192.168.1.141` is **`20.27 s`** (`3.53 s` tool selection + `0.16 s` Django tool call + `16.42 s` final answer generation).
- **Next Action**: Move `gemma4` inference onto a dedicated low-latency GPU host or quantized faster model for conversational synthesis so voice response latency drops below `2.5 s` while keeping `IVR_PROCESSING_SOUND_ENABLED=true`.

