"""
Retrieval Quality & Relevance Evaluator
=======================================
Classifies hybrid RAG results into confidence tiers (strong / partial / none)
and tracks platform retrieval & fallback metrics across live calls.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class RetrievalConfidenceTier(str, Enum):
    STRONG = "strong"
    PARTIAL = "partial"
    NONE = "none"


@dataclass
class RetrievalEvaluation:
    tier: RetrievalConfidenceTier
    top_score: float
    verified_hits: int
    total_hits: int
    reason: str
    categories: List[str] = field(default_factory=list)


def _parse_chunk_meta(chunk: Any) -> Dict[str, Any]:
    if isinstance(chunk, dict):
        return chunk.get("meta") or {}
    raw = getattr(chunk, "meta_json", None)
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def evaluate_relevance(
    scored_chunks: List[Tuple[Any, float]],
    strong_threshold: float = 0.65,
    partial_threshold: float = 0.42,
) -> RetrievalEvaluation:
    """
    Evaluate retrieval quality beyond a raw single cosine threshold:
    considers top hybrid score, verified source count, and topic categories.
    """
    if not scored_chunks:
        return RetrievalEvaluation(
            tier=RetrievalConfidenceTier.NONE,
            top_score=0.0,
            verified_hits=0,
            total_hits=0,
            reason="no_matching_chunks",
        )

    top_score = float(scored_chunks[0][1])
    verified_hits = 0
    categories: List[str] = []

    for chunk, score in scored_chunks:
        if score < partial_threshold:
            continue
        meta = _parse_chunk_meta(chunk)
        # Seeded official docs and explicitly verified chunks count as authoritative sources
        is_verified = meta.get("verified")
        if is_verified is True or (is_verified is None and meta.get("type") not in ("unanswered_candidate", "call_summary")):
            verified_hits += 1
        cat = meta.get("category")
        if cat and cat not in categories:
            categories.append(str(cat))

    if top_score >= strong_threshold and verified_hits >= 1:
        tier = RetrievalConfidenceTier.STRONG
        reason = "verified_strong_match"
    elif top_score >= partial_threshold:
        tier = RetrievalConfidenceTier.PARTIAL
        reason = "partial_or_unverified_match"
    else:
        tier = RetrievalConfidenceTier.NONE
        reason = "below_relevance_threshold"

    return RetrievalEvaluation(
        tier=tier,
        top_score=round(top_score, 4),
        verified_hits=verified_hits,
        total_hits=len(scored_chunks),
        reason=reason,
        categories=categories,
    )


class RetrievalEvaluator:
    """In-memory telemetry aggregator for RAG health metrics (Recall proxy, fallback & unanswered rate)."""

    def __init__(self) -> None:
        self.total_queries: int = 0
        self.strong_hits: int = 0
        self.partial_hits: int = 0
        self.unanswered_queries: int = 0
        self.fallback_responses: int = 0
        self.transfer_count: int = 0
        self._latencies_ms: List[float] = []

    def record_turn(
        self,
        evaluation: RetrievalEvaluation,
        latency_ms: float,
        used_fallback: bool = False,
        transferred: bool = False,
    ) -> None:
        self.total_queries += 1
        if evaluation.tier == RetrievalConfidenceTier.STRONG:
            self.strong_hits += 1
        elif evaluation.tier == RetrievalConfidenceTier.PARTIAL:
            self.partial_hits += 1
        else:
            self.unanswered_queries += 1
        if used_fallback:
            self.fallback_responses += 1
        if transferred:
            self.transfer_count += 1
        self._latencies_ms.append(latency_ms)
        if len(self._latencies_ms) > 500:
            self._latencies_ms = self._latencies_ms[-500:]

    def snapshot(self) -> Dict[str, Any]:
        total = max(1, self.total_queries)
        avg_lat = round(sum(self._latencies_ms) / len(self._latencies_ms), 2) if self._latencies_ms else 0.0
        return {
            "total_queries": self.total_queries,
            "strong_hit_rate": round(self.strong_hits / total, 4),
            "partial_hit_rate": round(self.partial_hits / total, 4),
            "unanswered_query_rate": round(self.unanswered_queries / total, 4),
            "fallback_rate": round(self.fallback_responses / total, 4),
            "transfer_rate": round(self.transfer_count / total, 4),
            "avg_response_latency_ms": avg_lat,
            "timestamp": time.time(),
        }

