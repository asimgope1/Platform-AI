"""
Multi-tenant connection pool manager & runtime telemetry tracker.
Creates, caches, recycles, and closes per-tenant database adapters cleanly.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from data_access.connectors.base import BaseDatabaseAdapter, ConnectionConfigSpec
from data_access.connectors.mongodb_adapter import MongoDBAdapter
from data_access.connectors.sql_adapter import SQLAlchemyCoreAdapter


@dataclass
class ConnectionMetrics:
    total_queries: int = 0
    failed_queries: int = 0
    connection_failures: int = 0
    rejected_queries: int = 0
    total_latency_ms: float = 0.0
    by_tenant: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def record_query(self, tenant_id: str, latency_ms: float, ok: bool, rejected: bool = False) -> None:
        self.total_queries += 1
        self.total_latency_ms += latency_ms
        t = self.by_tenant.setdefault(
            tenant_id,
            {"queries": 0, "failures": 0, "rejected": 0, "total_latency_ms": 0.0},
        )
        t["queries"] += 1
        t["total_latency_ms"] += latency_ms
        if rejected:
            self.rejected_queries += 1
            t["rejected"] += 1
        if not ok:
            self.failed_queries += 1
            t["failures"] += 1

    def record_connection_failure(self, tenant_id: str) -> None:
        self.connection_failures += 1
        t = self.by_tenant.setdefault(
            tenant_id,
            {"queries": 0, "failures": 0, "rejected": 0, "total_latency_ms": 0.0},
        )
        t["failures"] += 1

    def snapshot_for_tenant(self, tenant_id: str) -> Dict[str, Any]:
        t = self.by_tenant.get(tenant_id, {"queries": 0, "failures": 0, "rejected": 0, "total_latency_ms": 0.0})
        q = t["queries"]
        return {
            "tenant_id": tenant_id,
            "queries": q,
            "failures": t["failures"],
            "rejected": t["rejected"],
            "avg_latency_ms": round(t["total_latency_ms"] / q, 2) if q else 0.0,
        }


class ConnectionManager:
    """Thread/async-safe cache and factory for tenant database adapters."""

    def __init__(self) -> None:
        self._adapters: Dict[Tuple[str, str], BaseDatabaseAdapter] = {}
        self._lock = asyncio.Lock()
        self.metrics = ConnectionMetrics()

    @staticmethod
    def create_adapter(spec: ConnectionConfigSpec) -> BaseDatabaseAdapter:
        if spec.engine.lower() == "mongodb":
            return MongoDBAdapter(spec)
        return SQLAlchemyCoreAdapter(spec)

    async def get_adapter(self, spec: ConnectionConfigSpec) -> BaseDatabaseAdapter:
        key = (spec.tenant_id, spec.config_id)
        async with self._lock:
            existing = self._adapters.get(key)
            if existing is not None:
                # Recreate if connection attributes changed
                if existing.spec.model_dump() == spec.model_dump():
                    return existing
                await existing.close()
            adapter = self.create_adapter(spec)
            self._adapters[key] = adapter
            return adapter

    async def invalidate(self, tenant_id: str, config_id: str) -> None:
        key = (tenant_id, config_id)
        async with self._lock:
            existing = self._adapters.pop(key, None)
        if existing is not None:
            await existing.close()

    async def close_all(self) -> None:
        async with self._lock:
            adapters = list(self._adapters.values())
            self._adapters.clear()
        for adapter in adapters:
            await adapter.close()


connection_manager = ConnectionManager()

