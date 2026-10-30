"""
PBXWebSocketAdapter
===================
A thin shim that makes a `websockets` client connection look like a FastAPI
WebSocket to IVRCallSession — exposing the same `send_text` / `send_bytes`
interface without requiring any FastAPI dependency in the connector layer.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class PBXWebSocketAdapter:
    """
    Wraps a `websockets.WebSocketClientProtocol` (client socket to Asterisk)
    so that IVRCallSession can call `ws.send_text(...)` and `ws.send_bytes(...)`
    on it identically to a FastAPI WebSocket.
    """

    def __init__(self, ws):
        self._ws = ws

    async def send_text(self, data: str) -> None:
        await self._ws.send(data)

    async def send_bytes(self, data: bytes) -> None:
        await self._ws.send(data)

    async def close(self) -> None:
        try:
            await self._ws.close()
        except Exception:
            pass

