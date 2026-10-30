"""
Com-one Platform v2.0 - URL Configuration
"""
from django.contrib import admin
from django.urls import path, re_path, include
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import os
import httpx
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
FASTAPI_BACKEND_URL = os.getenv("FASTAPI_BACKEND_URL", "http://127.0.0.1:8001")

def home_view(request):
    html_path = BASE_DIR / 'services' / 'api' / 'static' / 'index.html'
    if html_path.exists():
        with open(html_path, 'r', encoding='utf-8') as f:
            resp = HttpResponse(f.read(), content_type='text/html')
            resp['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            resp['Pragma'] = 'no-cache'
            resp['Expires'] = '0'
            return resp
    return HttpResponse("<h1>Com-one Platform v2.0 AI Intelligence Layer</h1>")

def health_view(request):
    return JsonResponse({
        "status": "ok",
        "service": "Com-one Platform v2.0 AI Layer",
        "telephony_mode": os.getenv("COMONE_TELEPHONY_MODE", "STANDALONE_SIMULATOR"),
        "features": [
            "01_perceive_streaming_asr",
            "02_understand_intent_emotion_gemma",
            "03_respond_neural_tts_pgvector_rag",
            "04_act_crm_escalation",
            "asterisk_connector_ready"
        ]
    })

@csrf_exempt
def proxy_v1(request, path=""):
    """Transparently proxy all OpenAI-compatible /v1/ calls from UI on port 8000 to FastAPI on port 8001."""
    target_url = f"{FASTAPI_BACKEND_URL}/v1/{path}"
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.request(
                method=request.method,
                url=target_url,
                headers=headers,
                params=request.GET,
                content=request.body,
            )
            django_resp = HttpResponse(
                content=resp.content,
                status=resp.status_code,
                content_type=resp.headers.get("content-type", "application/json")
            )
            return django_resp
    except Exception as e:
        return JsonResponse({"error": f"AI Platform backend unreachable: {str(e)}"}, status=503)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home_view, name='home'),
    path('health/', health_view, name='health'),
    re_path(r'^v1/(?P<path>.*)$', proxy_v1, name='proxy_v1'),
    path('api/v2/', include('apps.api.urls')),
]
