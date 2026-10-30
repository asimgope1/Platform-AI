# AI Platform: Multilingual IVR & Continuous Learning Telephony

This repository contains the standalone Multilingual IVR & AI Platform with:
- **Languages:** Indian English, Hindi (Devanagari with feminine concordance), and authentic native Odia (Meta MMS-TTS VITS model).
- **PBX Integration:** Real-time Asterisk WebSocket bridge (`ws://<pbx-ip>:9093`) with jitter-buffer pre-filling and polyphase anti-aliasing audio pipeline.
- **Continuous Learning RAG:** Local PostgreSQL 16 + `pgvector` knowledge store that learns novel resolutions from live telephone dialogues.
- **Sub-Millisecond Prompts:** Pre-rendered RAM and disk audio cache (`data/ivr_audio_cache/`) for instant 0 ms menu delivery.

---

## 🚀 Quickstart: Setup on a New Machine (3 Steps)

### Step 1: Install Virtual Environment & Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Step 2: Configure Environment
```bash
cp .env.example .env
```
Ensure PostgreSQL with `pgvector` is running and `DATABASE_URL` is set in `.env`.
*If you don't have PostgreSQL installed, start one with Docker in 1 second:*
```bash
docker compose -f deploy/docker-compose.yml up -d postgres
```

### Step 3: Restore All Learned Chunks & Verify Cache (1 Command)
```bash
python scripts/restore_learned_knowledge.py
```
This automatically:
- Creates all database tables and enables `pgvector`.
- Provisions the default IVR tenant (`tenant-ivr-support`) and API keys.
- Ingests and vectorizes all **109 learned knowledge chunks** from `data/ivr_chunks_dump.json`.
- Verifies that all **46 pre-rendered audio prompts** in `data/ivr_audio_cache/` are ready.

---

## 📞 Running the Servers

```bash
# 1. Start FastAPI and Asterisk PBX Connector on port 8001
uvicorn services.api.main:app --host 0.0.0.0 --port 8001 --reload

# 2. In a separate terminal, start Django Com-one on port 8000 (if using Com-one interface)
python manage.py runserver 0.0.0.0:8000
```

Dial extension **8000** on MizoDroid or any SIP client.

---

## 🔄 Keeping Learned Knowledge in Sync Across Machines

When the system learns new Q&A pairs from live telephone calls into `pgvector`:
```bash
# Export the updated database chunks to JSON:
python scripts/dump_learned_knowledge.py

# Commit and push to git:
git add data/ivr_chunks_dump.json
git commit -m "chore: sync newly learned voice call Q&A chunks"
git push
```
On any other machine running the platform, simply run:
```bash
python scripts/restore_learned_knowledge.py
```
This merges and embeds the new chunks into `pgvector` without re-indexing duplicates.
