"""
Com-one Platform v2.0 - API URL Routes
"""
from django.urls import path
from apps.api.views import (
    StartCallView,
    ProcessTurnView,
    EndCallView,
    AgentHandoverView,
    SupervisorLiveMonitorView
)

urlpatterns = [
    path('telephony/calls/start/', StartCallView.as_view(), name='start-call'),
    path('telephony/calls/<str:call_id>/turn/', ProcessTurnView.as_view(), name='process-turn'),
    path('telephony/calls/<str:call_id>/end/', EndCallView.as_view(), name='end-call'),
    path('telephony/calls/<str:call_id>/handover/', AgentHandoverView.as_view(), name='agent-handover'),
    path('telephony/supervisor/live-monitor/', SupervisorLiveMonitorView.as_view(), name='supervisor-live-monitor'),
]
