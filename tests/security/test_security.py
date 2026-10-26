import pytest
import httpx
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from api.middleware.rate_limit import RateLimitMiddleware
from platform_core.config.settings import Settings, validate_master_key, settings
from storage.models.tenant import Tenant
from storage.models.api_key import ApiKey
from platform_core.security.api_keys import hash_api_key


@pytest.mark.asyncio
async def test_unauthenticated_request_rejected(async_client: httpx.AsyncClient):
    resp = await async_client.get("/v1/models")
    assert resp.status_code == 401
    data = resp.json()
    assert "error" in data


@pytest.mark.asyncio
async def test_invalid_api_key_rejected(async_client: httpx.AsyncClient):
    headers = {"Authorization": "Bearer sk-invalid-key-that-does-not-exist"}
    resp = await async_client.get("/v1/models", headers=headers)
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_master_key_authenticated_when_configured(async_client: httpx.AsyncClient, monkeypatch):
    test_key = "sk-valid-explicit-master-key-32chars"
    monkeypatch.setattr(settings, "MASTER_API_KEY", test_key)

    headers = {"Authorization": f"Bearer {test_key}"}
    resp = await async_client.get("/v1/models", headers=headers)
    assert resp.status_code == 200
    assert "data" in resp.json()


@pytest.mark.asyncio
async def test_master_key_disabled_when_none(async_client: httpx.AsyncClient, monkeypatch):
    # When MASTER_API_KEY is None, attempting to use any master key is rejected with 401
    monkeypatch.setattr(settings, "MASTER_API_KEY", None)

    headers = {"Authorization": "Bearer sk-master-test-key"}
    resp = await async_client.get("/v1/models", headers=headers)
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_old_placeholder_rejected_by_default(async_client: httpx.AsyncClient, monkeypatch):
    # Without explicitly setting the placeholder as the key, sending it must fail with 401
    monkeypatch.setattr(settings, "MASTER_API_KEY", None)

    headers = {"Authorization": "Bearer sk-platform-master-key-change-in-production"}
    resp = await async_client.get("/v1/models", headers=headers)
    assert resp.status_code == 401


def test_validate_master_key_production_enforcement():
    # 1. Missing key in production must raise ValueError
    cfg_missing = Settings(ENVIRONMENT="production", MASTER_API_KEY=None)
    with pytest.raises(ValueError, match="Production requires an explicitly configured"):
        validate_master_key(cfg_missing)

    # 2. Key too short (< 32 chars) in production must raise ValueError
    cfg_short = Settings(ENVIRONMENT="production", MASTER_API_KEY="short-key-123")
    with pytest.raises(ValueError, match="at least 32 characters"):
        validate_master_key(cfg_short)

    # 3. Default placeholder in production must raise ValueError
    cfg_placeholder = Settings(
        ENVIRONMENT="production",
        MASTER_API_KEY="sk-platform-master-key-change-in-production"
    )
    with pytest.raises(ValueError, match="placeholder cannot be used in production"):
        validate_master_key(cfg_placeholder)

    # 4. Valid 32+ char key in production passes
    cfg_valid = Settings(
        ENVIRONMENT="production",
        MASTER_API_KEY="sk-prod-sec-key-7483920194857201948572"
    )
    validate_master_key(cfg_valid)


@pytest.mark.asyncio
async def test_cross_tenant_document_isolation(async_client: httpx.AsyncClient, db_session):
    # Seed Tenant A
    raw_key_a = "sk_live_tenant_a_key"
    tenant_a = Tenant(name="Tenant A")
    db_session.add(tenant_a)
    await db_session.commit()
    await db_session.refresh(tenant_a)
    key_a = ApiKey(tenant_id=tenant_a.id, key_hash=hash_api_key(raw_key_a), name="Key A", scopes="*")
    db_session.add(key_a)

    # Seed Tenant B
    raw_key_b = "sk_live_tenant_b_key"
    tenant_b = Tenant(name="Tenant B")
    db_session.add(tenant_b)
    await db_session.commit()
    await db_session.refresh(tenant_b)
    key_b = ApiKey(tenant_id=tenant_b.id, key_hash=hash_api_key(raw_key_b), name="Key B", scopes="*")
    db_session.add(key_b)
    await db_session.commit()

    headers_a = {"Authorization": f"Bearer {raw_key_a}"}
    headers_b = {"Authorization": f"Bearer {raw_key_b}"}

    # Tenant A creates document
    upload_resp = await async_client.post(
        "/v1/documents",
        json={"title": "Tenant A Secret", "content": "Confidential financial projections of Tenant A"},
        headers=headers_a,
    )
    assert upload_resp.status_code == 201
    doc_a_id = upload_resp.json()["id"]

    # Tenant B tries to delete Tenant A's document -> 404 (Not Found / Access Denied)
    del_resp = await async_client.delete(f"/v1/documents/{doc_a_id}", headers=headers_b)
    assert del_resp.status_code == 404

    # Tenant B lists documents -> Tenant A's document MUST NOT appear
    list_b = await async_client.get("/v1/documents", headers=headers_b)
    assert list_b.status_code == 200
    docs_b = list_b.json()["data"]
    assert not any(d["id"] == doc_a_id for d in docs_b)

    # Tenant B searches for Tenant A's secret content -> 0 matches
    search_b = await async_client.post(
        "/v1/documents/search",
        json={"query": "financial projections"},
        headers=headers_b,
    )
    assert search_b.status_code == 200
    assert len(search_b.json()["data"]) == 0

    # Tenant A searches for Tenant A's secret content -> Finds match
    search_a = await async_client.post(
        "/v1/documents/search",
        json={"query": "financial projections"},
        headers=headers_a,
    )
    assert search_a.status_code == 200
    assert len(search_a.json()["data"]) > 0


@pytest.mark.asyncio
async def test_rate_limiter_middleware_exceeded():
    async def sample_endpoint(request):
        return JSONResponse({"status": "ok"})

    mini_app = Starlette(routes=[Route("/api/test", sample_endpoint)])
    mini_app.add_middleware(RateLimitMiddleware, max_requests_per_minute=3)

    transport = httpx.ASGITransport(app=mini_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        headers = {"Authorization": "Bearer client-burst"}
        res1 = await client.get("/api/test", headers=headers)
        res2 = await client.get("/api/test", headers=headers)
        res3 = await client.get("/api/test", headers=headers)
        res4 = await client.get("/api/test", headers=headers)

        assert res1.status_code == 200
        assert res2.status_code == 200
        assert res3.status_code == 200
        assert res4.status_code == 429
        assert res4.json()["error"]["code"] == "rate_limit_exceeded"
