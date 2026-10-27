import uuid
from django.db import models

class CallSession(models.Model):
    STATUS_CHOICES = [
        ('initiated', 'Initiated'),
        ('active', 'Active'),
        ('escalated', 'Escalated to Human Agent'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    call_id = models.CharField(max_length=128, unique=True, db_index=True)
    tenant_id = models.CharField(max_length=64, default='879e9bda-8049-4824-9aa0-fc5e9f0012d9', db_index=True)
    caller_number = models.CharField(max_length=32, default='+919876543210')
    did_number = models.CharField(max_length=32, default='1800-COM-ONE')
    status = models.CharField(max_length=32, choices=STATUS_CHOICES, default='initiated')
    language = models.CharField(max_length=8, default='hi')
    current_intent = models.CharField(max_length=128, blank=True, null=True)
    caller_sentiment = models.FloatField(default=0.0)
    caller_emotion = models.CharField(max_length=32, default='calm')
    is_escalated = models.BooleanField(default=False)
    escalation_reason = models.TextField(blank=True, null=True)
    assigned_agent = models.CharField(max_length=128, blank=True, null=True)
    started_at = models.DateTimeField(auto_now_add=True)
    ended_at = models.DateTimeField(blank=True, null=True)

    def __str__(self):
        return f"Call {self.call_id} [{self.caller_number}] - {self.status}"

class TelephonyEvent(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    call_session = models.ForeignKey(CallSession, on_delete=models.CASCADE, related_name='events')
    event_type = models.CharField(max_length=64)
    payload = models.JSONField(default=dict, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.event_type} on {self.call_session.call_id}"

class CDRRecord(models.Model):
    DISPOSITION_CHOICES = [
        ('RESOLVED_BY_AI', 'Resolved by AI (Pure Deflection)'),
        ('ESCALATED_TO_HUMAN', 'Escalated to Human Agent'),
        ('CALLER_ABANDONED', 'Caller Abandoned'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    call_session = models.OneToOneField(CallSession, on_delete=models.CASCADE, related_name='cdr')
    caller_number = models.CharField(max_length=32)
    duration_seconds = models.IntegerField(default=0)
    billable_seconds = models.IntegerField(default=0)
    disposition = models.CharField(max_length=32, choices=DISPOSITION_CHOICES, default='RESOLVED_BY_AI')
    csat_rating = models.IntegerField(null=True, blank=True)
    transcript_summary = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"CDR {self.call_session.call_id} - {self.disposition} ({self.duration_seconds}s)"
