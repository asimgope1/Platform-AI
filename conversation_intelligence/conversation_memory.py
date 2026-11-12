"""
Conversation Memory — Tracks sliding window (last 10 turns) of emotion EMA,
emotional trend (`improving` / `stable` / `worsening`), and repeated queries.

Uses exponential moving average:
    E_t = 0.6 * E_{t-1} + 0.4 * E_current
so one anomalous turn never triggers a premature escalation.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional


EMOTION_VALENCE_MAP: Dict[str, float] = {
    "angry": -1.0,
    "frustrated": -0.75,
    "sad": -0.45,
    "concerned": -0.35,
    "neutral": 0.0,
    "happy": 0.70,
    "satisfied": 0.85,
}


@dataclass
class TurnMemorySnapshot:
    turn_number: int
    intent: str
    emotion: str
    valence: float
    satisfaction: Optional[float]
    resolution_status: str


class ConversationMemory:
    """Maintains multi-turn emotional trajectory and repeat detection."""

    def __init__(self, maxlen: int = 10) -> None:
        self.turns: Deque[TurnMemorySnapshot] = deque(maxlen=maxlen)
        self.ema_valence: float = 0.0
        self.last_intent: Optional[str] = None
        self.consecutive_same_intent: int = 0

    def record_turn(
        self,
        turn_number: int,
        intent: str,
        primary_emotion: str,
        satisfaction: Optional[float],
        resolution_status: str,
    ) -> str:
        """Update moving average and return trend ('improving', 'stable', 'worsening')."""
        current_valence = EMOTION_VALENCE_MAP.get(primary_emotion, 0.0)
        prev_ema = self.ema_valence
        if not self.turns:
            self.ema_valence = current_valence
        else:
            self.ema_valence = 0.6 * self.ema_valence + 0.4 * current_valence

        # Track repeated unresolved intent (e.g. caller repeating the same UPI problem)
        if intent not in ("greeting", "appreciation", "confirmation_yes", "confirmation_no", "unknown") and intent == self.last_intent and resolution_status != "resolved":
            self.consecutive_same_intent += 1
        elif intent not in ("confirmation_yes", "confirmation_no", "unknown"):
            self.last_intent = intent
            self.consecutive_same_intent = 0

        self.turns.append(
            TurnMemorySnapshot(
                turn_number=turn_number,
                intent=intent,
                emotion=primary_emotion,
                valence=round(current_valence, 3),
                satisfaction=satisfaction,
                resolution_status=resolution_status,
            )
        )

        delta = self.ema_valence - prev_ema
        if self.ema_valence <= -0.35 or delta <= -0.15:
            return "worsening"
        if delta >= 0.15 or (len(self.turns) >= 2 and current_valence > 0.3 > prev_ema):
            return "improving"
        return "stable"

