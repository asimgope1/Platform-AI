import uuid
from django.db import models
from apps.telephony.models import CallSession

class ConversationTurn(models.Model):
    SPEAKER_CHOICES = [
        ('caller', 'Caller'),
        ('ai_agent', 'AI Voice Agent'),
        ('human_agent', 'Human Agent'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    call_session = models.ForeignKey(CallSession, on_delete=models.CASCADE, related_name='turns')
    turn_index = models.IntegerField(default=1)
    speaker = models.CharField(max_length=32, choices=SPEAKER_CHOICES)
    text = models.TextField()
    detected_intent = models.CharField(max_length=128, blank=True, null=True)
    sentiment_score = models.FloatField(default=0.0)
    detected_emotion = models.CharField(max_length=32, default='calm')
    retrieved_chunks = models.JSONField(default=list, blank=True)
    latency_ms = models.IntegerField(default=350)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['turn_index']

    def __str__(self):
        return f"Turn {self.turn_index} [{self.speaker}] on {self.call_session.call_id}"
