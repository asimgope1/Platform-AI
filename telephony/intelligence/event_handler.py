"""
Non-Blocking Background Telemetry & Event Writer for Voice Intelligence.

Uses a bounded asyncio.Queue (default maxsize=100) so database writes never block
real-time 20ms telephony audio streaming or LLM responses.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Coroutine, Dict, Optional

from storage.database import async_session
from storage.repositories.feedback_repository import FeedbackRepository
from storage.repositories.intelligence_repository import IntelligenceRepository

logger = logging.getLogger(__name__)


class IntelligenceEventHandler:
    """Bounded async background queue for call intelligence persistence."""

    def __init__(self, max_queue_size: int = 100) -> None:
        self._queue: asyncio.Queue[Callable[[], Coroutine[Any, Any, None]]] = asyncio.Queue(
            maxsize=max_queue_size
        )
        self._worker_task: Optional[asyncio.Task] = None

    def _ensure_worker(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = loop.create_task(self._worker())

    async def _worker(self) -> None:
        while True:
            job = await self._queue.get()
            try:
                await job()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug(f"[IntelligenceDB] Non-fatal persistence skip: {exc}")
            finally:
                self._queue.task_done()

    def _enqueue(self, job_factory: Callable[[], Coroutine[Any, Any, None]]) -> None:
        self._ensure_worker()
        if self._queue.full():
            logger.debug("[IntelligenceDB] Queue full; dropping non-critical telemetry write.")
            return
        self._queue.put_nowait(job_factory)

    def emit_turn(self, **kwargs: Any) -> None:
        async def _persist():
            async with async_session() as session:
                repo = IntelligenceRepository(session)
                await repo.record_turn(**kwargs)

        self._enqueue(_persist)

    def emit_event(
        self,
        tenant_id: str,
        call_uuid: str,
        event_type: str,
        event_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        async def _persist():
            async with async_session() as session:
                repo = IntelligenceRepository(session)
                await repo.record_event(
                    tenant_id=tenant_id,
                    call_uuid=call_uuid,
                    event_type=event_type,
                    event_data=event_data or {},
                )

        self._enqueue(_persist)

    def emit_feedback(self, **kwargs: Any) -> None:
        async def _persist():
            async with async_session() as session:
                repo = FeedbackRepository(session)
                await repo.record_feedback(**kwargs)

        self._enqueue(_persist)

    def emit_transfer(self, **kwargs: Any) -> None:
        async def _persist():
            async with async_session() as session:
                repo = IntelligenceRepository(session)
                await repo.record_transfer(**kwargs)

        self._enqueue(_persist)

    def emit_resolution(self, **kwargs: Any) -> None:
        async def _persist():
            async with async_session() as session:
                repo = IntelligenceRepository(session)
                await repo.record_resolution(**kwargs)

        self._enqueue(_persist)

    async def flush(self, timeout: float = 1.5) -> None:
        """Wait briefly for queued background writes (useful in tests/shutdown)."""
        if self._worker_task and not self._worker_task.done():
            try:
                await asyncio.wait_for(self._queue.join(), timeout=timeout)
            except asyncio.TimeoutError:
                pass

    def close(self) -> None:
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()

