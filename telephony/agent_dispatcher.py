"""
Agent Dispatcher & Extension State Manager for Asterisk PBX.
Tracks live agent extension availability (idle vs. busy) and dispatches
transfer control events over WebSocket to connect callers to the first available extension.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Default agent extensions pool (e.g. 100, 200)
DEFAULT_EXTENSIONS = ["100", "200"]


class AgentDispatcher:
    """
    Tracks extension occupancy and selects the next available non-busy agent extension
    for live Asterisk transfers.
    """

    def __init__(self, extensions: Optional[List[str]] = None) -> None:
        raw_env = os.getenv("PBX_AGENT_EXTENSIONS", "")
        if raw_env.strip():
            self.configured_extensions = [e.strip() for e in raw_env.split(",") if e.strip()]
        else:
            self.configured_extensions = extensions or list(DEFAULT_EXTENSIONS)

        # In-memory extension status: "idle" | "ringing" | "in_use" | "busy"
        self._extension_states: Dict[str, str] = {ext: "idle" for ext in self.configured_extensions}
        self._active_calls: Dict[str, str] = {}  # call_uuid -> extension
        self._last_assigned_index: int = 0

    def get_configured_extensions(self) -> List[str]:
        return list(self.configured_extensions)

    def mark_extension_busy(self, extension: str, call_uuid: Optional[str] = None) -> None:
        ext = str(extension).strip()
        self._extension_states[ext] = "in_use"
        if call_uuid:
            self._active_calls[call_uuid] = ext
        logger.info(f"[AgentDispatcher] Extension '{ext}' marked BUSY (call={call_uuid}).")

    def mark_extension_idle(self, extension: str) -> None:
        ext = str(extension).strip()
        self._extension_states[ext] = "idle"
        # Remove any call associating this extension
        uuids_to_del = [u for u, e in self._active_calls.items() if e == ext]
        for u in uuids_to_del:
            self._active_calls.pop(u, None)
        logger.info(f"[AgentDispatcher] Extension '{ext}' marked IDLE.")

    def release_call(self, call_uuid: str) -> None:
        if call_uuid in self._active_calls:
            ext = self._active_calls.pop(call_uuid)
            self._extension_states[ext] = "idle"
            logger.info(f"[AgentDispatcher] Call {call_uuid} ended. Released extension '{ext}' to IDLE.")

    def update_from_pbx_event(self, event: Dict[str, Any]) -> None:
        """Process Asterisk PBX status / device_state / extension_status notifications."""
        etype = str(event.get("event", "")).lower()
        ext = str(event.get("extension") or event.get("exten") or event.get("target") or "").strip()
        status = str(event.get("status") or event.get("state") or "").lower()

        if not ext or ext not in self._extension_states:
            return

        if etype in ("extension_status", "device_state", "peer_status"):
            if any(s in status for s in ("idle", "not_inuse", "available", "unregistered")):
                self._extension_states[ext] = "idle"
                logger.info(f"[AgentDispatcher] PBX event: Extension {ext} is now IDLE.")
            elif any(s in status for s in ("busy", "inuse", "ringing", "unavailable", "hold")):
                self._extension_states[ext] = "busy"
                logger.info(f"[AgentDispatcher] PBX event: Extension {ext} is now BUSY ({status}).")

    def get_available_extension(self, exclude: Optional[Set[str]] = None) -> str:
        """
        Finds the first available extension that is not currently busy.
        Falls back to round-robin hunt among configured extensions if all report busy.
        """
        excluded = exclude or set()
        
        # 1. Search for explicitly idle extensions
        idle_exts = [
            ext for ext in self.configured_extensions
            if ext not in excluded and self._extension_states.get(ext, "idle") == "idle"
        ]
        if idle_exts:
            chosen = idle_exts[0]
            logger.info(f"[AgentDispatcher] Found idle extension: '{chosen}' from pool {idle_exts}.")
            return chosen

        # 2. If all are marked busy, round-robin through configured extensions
        if self.configured_extensions:
            self._last_assigned_index = (self._last_assigned_index + 1) % len(self.configured_extensions)
            chosen = self.configured_extensions[self._last_assigned_index]
            logger.warning(
                f"[AgentDispatcher] All extensions busy ({self._extension_states}); "
                f"falling back to round-robin hunt target: '{chosen}'."
            )
            return chosen

        return "1001"

    def build_transfer_payload(
        self,
        call_uuid: str,
        target_extension: str,
        reason: str,
        language: str = "en",
        caller_number: Optional[str] = None,
        summary: str = "",
    ) -> Dict[str, Any]:
        """
        Constructs standard Asterisk WebSocket transfer message recognized by Asterisk PBX.
        Includes fallback extensions to allow Asterisk hunt group dialing.
        """
        fallbacks = [ext for ext in self.configured_extensions if ext != target_extension]
        return {
            "event": "transfer",
            "action": "transfer",
            "uuid": call_uuid,
            "target": target_extension,
            "extension": target_extension,
            "channel": f"PJSIP/{target_extension}",
            "fallback_extensions": fallbacks,
            "strategy": "hunt_available",
            "reason": reason,
            "caller": caller_number or "unknown",
            "language": language,
            "summary": summary,
            "timestamp": time.time(),
        }


# Global singleton instance
agent_dispatcher = AgentDispatcher()
