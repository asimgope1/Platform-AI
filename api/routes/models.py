import time
from fastapi import APIRouter, Depends
from providers.registry import provider_registry
from platform_core.security.authentication import get_current_tenant
from platform_core.security.tenant_context import TenantContext

router = APIRouter(prefix="/v1", tags=["Models"])


@router.get("/models")
async def list_models(context: TenantContext = Depends(get_current_tenant)):
    """List all available model aliases configured on the platform."""
    models = provider_registry.list_models()
    return {
        "object": "list",
        "data": [
            {
                "id": m.alias,
                "object": "model",
                "created": 1710000000,
                "owned_by": "ai-platform",
                "provider_id": m.provider_id,
                "description": m.description,
                "capabilities": sorted([c.value for c in m.capabilities]),
            }
            for m in models
        ],
    }


@router.get("/models/{model_alias}")
async def retrieve_model(model_alias: str, context: TenantContext = Depends(get_current_tenant)):
    """Retrieve details for a specific model alias."""
    m = provider_registry.get_model_metadata(model_alias)
    return {
        "id": m.alias,
        "object": "model",
        "created": 1710000000,
        "owned_by": "ai-platform",
        "provider_id": m.provider_id,
        "description": m.description,
        "capabilities": sorted([c.value for c in m.capabilities]),
    }
