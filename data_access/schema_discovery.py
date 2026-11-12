"""
Read-only schema discovery service for inspecting customer business databases
without assuming hardcoded table names or mutating schemas.
"""
from __future__ import annotations

from data_access.connectors.base import ConnectionConfigSpec, DiscoveredSchema
from data_access.connectors.connection_manager import ConnectionManager, connection_manager


class SchemaDiscoveryService:
    """Inspects an external database read-only and returns normalized table/column/key metadata."""

    def __init__(self, manager: ConnectionManager = connection_manager):
        self._manager = manager

    async def inspect_database(
        self,
        spec: ConnectionConfigSpec,
        *,
        timeout_seconds: float = 10.0,
    ) -> DiscoveredSchema:
        adapter = await self._manager.get_adapter(spec)
        try:
            return await adapter.discover_schema(timeout_seconds=timeout_seconds)
        except Exception:
            self._manager.metrics.record_connection_failure(spec.tenant_id)
            raise

