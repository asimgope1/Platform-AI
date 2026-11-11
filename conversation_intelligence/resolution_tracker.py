"""
Resolution Tracker — ensures customer issues are only marked `resolved` after
positive confirmation signals or explicit resolution statements.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict


RESOLVED_EXPLICIT_RE = re.compile(
    r"\b(?:resolved|problem solved|issue solved|solved my (?:problem|issue)|resolved my (?:problem|issue)|"
    r"got it|that helped|understood now|working now|all set)\b|"
    r"(?:समाधान हो गया|समझ आ गया|समस्या हल|काम हो गया|ठीक है समझ गई|ସମାଧାନ ହୋଇଗଲା|ବୁଝିପାରିଲି|ଠିକ୍ ହୋଇଗଲା)",
    re.IGNORECASE,
)

UNRESOLVED_RE = re.compile(
    r"\b(?:not resolved|still pending|still failed|didn't work|not working|same issue|no help|still stuck)\b|"
    r"(?:अभी भी नहीं|समाधान नहीं|काम नहीं कर|अब भी वही|ଏବେ ବି ହୋଇନି|ସମାଧାନ ହୋଇନି)",
    re.IGNORECASE,
)


@dataclass
class ResolutionResult:
    resolution_status: str   # "unresolved" | "potentially_resolved" | "resolved"
    customer_confirmed: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ResolutionTracker:
    """Tracks whether the caller confirmed issue resolution across turns."""

    def evaluate(
        self,
        text: str,
        intent: str,
        prior_status: str = "unresolved",
        awaiting_resolution_confirmation: bool = False,
        prior_confirmed: bool = False,
    ) -> ResolutionResult:
        clean = (text or "").strip()

        if UNRESOLVED_RE.search(clean) or (awaiting_resolution_confirmation and intent == "confirmation_no"):
            return ResolutionResult(resolution_status="unresolved", customer_confirmed=False)

        if RESOLVED_EXPLICIT_RE.search(clean) or (awaiting_resolution_confirmation and intent == "confirmation_yes"):
            return ResolutionResult(resolution_status="resolved", customer_confirmed=True)

        if intent == "appreciation" and prior_status in ("potentially_resolved", "resolved"):
            return ResolutionResult(resolution_status="resolved", customer_confirmed=True)

        if intent == "appreciation":
            return ResolutionResult(resolution_status="potentially_resolved", customer_confirmed=prior_confirmed)

        return ResolutionResult(resolution_status=prior_status or "unresolved", customer_confirmed=prior_confirmed)
