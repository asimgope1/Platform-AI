"""
Com-one Platform v2.0 - Unified ASGI Application
Routes real-time streaming, Django REST Framework, and OpenAI-compatible endpoints seamlessly.
"""
import os
import django
from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'comone.settings')
django_app = get_asgi_application()

# Import FastAPI application
from services.api.main import app as fastapi_app
from providers.registry import provider_registry
from platform_core.config.settings import settings

if os.path.exists(settings.PROVIDERS_CONFIG_PATH):
    try:
        provider_registry.initialize_from_file(settings.PROVIDERS_CONFIG_PATH)
    except Exception as e:
        pass

import threading
from apps.providers.services import ProviderBridge
threading.Thread(target=ProviderBridge.warmup_models, daemon=True).start()

async def application(scope, receive, send):
    """
    Unified ASGI Router:
    - /v1/*, /docs, /openapi.json -> FastAPI Core AI Engine
    - All other routes (/, /health/, /api/v2/*, /admin/*) -> Django v2.0 Application
    """
    if scope["type"] == "http":
        path = scope.get("path", "")
        if path.startswith(("/v1/", "/docs", "/openapi.json")):
            await fastapi_app(scope, receive, send)
            return

    await django_app(scope, receive, send)
