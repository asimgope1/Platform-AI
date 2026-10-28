import pytest
import os
import sys
import asyncio
from typing import AsyncGenerator
import httpx
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set test environment
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ["ENVIRONMENT"] = "testing"
os.environ["MASTER_API_KEY"] = "sk-master-test-key"

from storage.database import Base, get_db_session
from services.api.main import app
from providers.registry import provider_registry
from platform_core.security.api_keys import hash_api_key
from storage.models.tenant import Tenant
from storage.models.api_key import ApiKey

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"
test_engine = create_async_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestSessionLocal = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture(scope="session", autouse=True)
def init_providers():
    """Ensure providers registry is initialized from config file."""
    config_path = "config/providers.yaml"
    if os.path.exists(config_path):
        provider_registry.initialize_from_file(config_path)


@pytest.fixture(autouse=True)
async def setup_test_db():
    """Create all tables before each test and drop after."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    async with TestSessionLocal() as session:
        yield session


@pytest.fixture
async def test_tenant_and_key(db_session: AsyncSession):
    """Seed test tenant and valid API key."""
    raw_key = "sk-valid-test-key-abcdef"
    key_hash = hash_api_key(raw_key)

    tenant = Tenant(name="Test Organization", is_active=True)
    db_session.add(tenant)
    await db_session.commit()
    await db_session.refresh(tenant)

    api_key = ApiKey(
        tenant_id=tenant.id,
        key_hash=key_hash,
        name="Test API Key",
        scopes="*",
    )
    db_session.add(api_key)
    await db_session.commit()
    await db_session.refresh(api_key)

    return {
        "tenant": tenant,
        "api_key": api_key,
        "raw_key": raw_key,
    }


@pytest.fixture
async def async_client(db_session: AsyncSession) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Test HTTP client overriding db session dependency."""
    async def override_get_db_session():
        async with TestSessionLocal() as session:
            yield session

    app.dependency_overrides[get_db_session] = override_get_db_session

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    app.dependency_overrides.clear()
