"""
Com-one Platform v2.0 - URL Configuration
"""
from django.contrib import admin
from django.urls import path, include
from django.http import HttpResponse, JsonResponse
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

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

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home_view, name='home'),
    path('health/', health_view, name='health'),
    path('api/v2/', include('apps.api.urls')),
]
