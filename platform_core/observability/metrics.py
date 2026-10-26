import time
from typing import Dict, Any, Optional
from collections import defaultdict


class MetricsCollector:
    """In-memory operational metrics collector for latency, status codes, and tokens."""

    def __init__(self):
        self.request_counts: Dict[str, int] = defaultdict(int)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.total_tokens_used: Dict[str, int] = defaultdict(int)
        self.total_latency_ms: Dict[str, float] = defaultdict(float)

    def record_request(
        self,
        endpoint: str,
        model: str,
        latency_ms: float,
        status_code: int,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        key = f"{endpoint}:{model}"
        self.request_counts[key] += 1
        self.total_latency_ms[key] += latency_ms
        self.total_tokens_used[key] += (prompt_tokens + completion_tokens)
        if status_code >= 400:
            self.error_counts[key] += 1

    def get_summary(self) -> Dict[str, Any]:
        summary = {}
        for key, count in self.request_counts.items():
            total_lat = self.total_latency_ms[key]
            summary[key] = {
                "total_requests": count,
                "error_requests": self.error_counts[key],
                "avg_latency_ms": round(total_lat / count, 2) if count > 0 else 0.0,
                "total_tokens": self.total_tokens_used[key],
            }
        return summary


metrics = MetricsCollector()
