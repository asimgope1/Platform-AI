import pytest
import json
import httpx


@pytest.mark.asyncio
async def test_health_endpoint(async_client: httpx.AsyncClient):
    resp = await async_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "database" in data["services"]
    assert "metrics" in data


@pytest.mark.asyncio
async def test_models_endpoints(async_client: httpx.AsyncClient, test_tenant_and_key):
    headers = {"Authorization": f"Bearer {test_tenant_and_key['raw_key']}"}

    # List models
    list_resp = await async_client.get("/v1/models", headers=headers)
    assert list_resp.status_code == 200
    models_data = list_resp.json()["data"]
    aliases = [m["id"] for m in models_data]
    assert "mock-chat" in aliases

    # Retrieve specific model
    get_resp = await async_client.get("/v1/models/mock-chat", headers=headers)
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == "mock-chat"


@pytest.mark.asyncio
async def test_chat_completions_non_streaming(async_client: httpx.AsyncClient, test_tenant_and_key):
    headers = {"Authorization": f"Bearer {test_tenant_and_key['raw_key']}"}
    payload = {
        "model": "mock-chat",
        "messages": [
            {"role": "user", "content": "Hello AI platform!"}
        ],
        "stream": False,
    }

    resp = await async_client.post("/v1/chat/completions", json=payload, headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "mock-chat"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert len(data["choices"][0]["message"]["content"]) > 0


@pytest.mark.asyncio
async def test_chat_completions_streaming(async_client: httpx.AsyncClient, test_tenant_and_key):
    headers = {"Authorization": f"Bearer {test_tenant_and_key['raw_key']}"}
    payload = {
        "model": "mock-chat",
        "messages": [
            {"role": "user", "content": "Stream me a message"}
        ],
        "stream": True,
    }

    resp = await async_client.post("/v1/chat/completions", json=payload, headers=headers)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")

    content = resp.text
    assert "data: " in content
    assert "[DONE]" in content


@pytest.mark.asyncio
async def test_embeddings_endpoint(async_client: httpx.AsyncClient, test_tenant_and_key):
    headers = {"Authorization": f"Bearer {test_tenant_and_key['raw_key']}"}
    payload = {
        "model": "text-embedding-3-small",
        "input": ["First sentence", "Second sentence"],
    }

    resp = await async_client.post("/v1/embeddings", json=payload, headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 2
    assert len(data["data"][0]["embedding"]) == 1536


@pytest.mark.asyncio
async def test_documents_crud_and_search(async_client: httpx.AsyncClient, test_tenant_and_key):
    headers = {"Authorization": f"Bearer {test_tenant_and_key['raw_key']}"}

    # 1. Upload document
    upload_payload = {
        "title": "Quarterly Financial Report",
        "content": "In Q3, enterprise revenue increased by 35% driven by AI subscriptions and cloud contracts.",
        "source_type": "text",
    }
    create_resp = await async_client.post("/v1/documents", json=upload_payload, headers=headers)
    assert create_resp.status_code == 201
    doc_id = create_resp.json()["id"]

    # 2. List documents
    list_resp = await async_client.get("/v1/documents", headers=headers)
    assert list_resp.status_code == 200
    docs = list_resp.json()["data"]
    assert any(d["id"] == doc_id for d in docs)

    # 3. Search document chunks
    search_payload = {
        "query": "enterprise revenue growth",
        "top_k": 3,
    }
    search_resp = await async_client.post("/v1/documents/search", json=search_payload, headers=headers)
    assert search_resp.status_code == 200
    results = search_resp.json()["data"]
    assert len(results) > 0
    assert "revenue increased" in results[0]["content"]

    # 4. Delete document
    del_resp = await async_client.delete(f"/v1/documents/{doc_id}", headers=headers)
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted"] is True


@pytest.mark.asyncio
async def test_audio_endpoints(async_client: httpx.AsyncClient, test_tenant_and_key):
    headers = {"Authorization": f"Bearer {test_tenant_and_key['raw_key']}"}

    # 1. Speech synthesis (TTS)
    tts_payload = {
        "model": "tts-1",
        "input": "Antigravity AI platform audio synthesis test.",
        "voice": "alloy",
        "response_format": "mp3",
    }
    tts_resp = await async_client.post("/v1/audio/speech", json=tts_payload, headers=headers)
    assert tts_resp.status_code == 200
    assert "audio/mpeg" in tts_resp.headers.get("content-type", "")
    assert len(tts_resp.content) > 0

    # 2. Speech transcription (STT)
    files = {"file": ("test_recording.mp3", b"dummy_audio_bytes_for_whisper", "audio/mpeg")}
    data = {"model": "whisper-1"}
    stt_resp = await async_client.post("/v1/audio/transcriptions", files=files, data=data, headers=headers)
    assert stt_resp.status_code == 200
    stt_json = stt_resp.json()
    assert "test_recording.mp3" in stt_json["text"]
