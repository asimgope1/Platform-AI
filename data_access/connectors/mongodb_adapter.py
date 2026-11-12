"""
Dedicated document database adapter for MongoDB.
Enforces read-only queries, blocks arbitrary JavaScript/operator injection ($where, $function),
infers collection schemas via safe document sampling, and normalizes documents into QueryResult.
"""
from __future__ import annotations

import asyncio
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus

from data_access.connectors.base import (
    BaseDatabaseAdapter,
    ColumnMeta,
    ConnectionConfigSpec,
    DiscoveredSchema,
    QueryResult,
    QuerySpec,
    TableMeta,
)
from data_access.security import validate_identifier
from platform_core.errors.exceptions import PermissionDeniedError, ValidationError


def _escape_regex_literal(value: str) -> str:
    return re.escape(value)


class MongoDBAdapter(BaseDatabaseAdapter):
    """Read-only MongoDB document adapter matching the BaseDatabaseAdapter contract."""

    def __init__(self, spec: ConnectionConfigSpec, client: Any = None):
        super().__init__(spec)
        if not spec.read_only:
            raise PermissionDeniedError("External MongoDB databases must remain read_only=True.")
        self._client = client

    def _build_uri(self) -> str:
        if self.spec.connection_uri:
            return self.spec.connection_uri.strip()
        host = self.spec.host or "localhost"
        port = self.spec.port or 27017
        auth = ""
        if self.spec.username and self.spec.password:
            auth = f"{quote_plus(self.spec.username)}:{quote_plus(self.spec.password)}@"
        elif self.spec.username:
            auth = f"{quote_plus(self.spec.username)}@"
        return f"mongodb://{auth}{host}:{port}/{self.spec.database_name}?readPreference=secondaryPreferred"

    def _get_client(self):
        if self._client is None:
            try:
                from pymongo import MongoClient
            except ImportError as exc:
                raise ValidationError(
                    "MongoDB adapter requires 'pymongo' to be installed in this environment."
                ) from exc
            client_kwargs: Dict[str, Any] = {
                "serverSelectionTimeoutMS": int(self.spec.options.get("timeout_ms", 5000)),
                "maxPoolSize": int(self.spec.options.get("pool_size", 10)),
                "read_preference": self.spec.options.get("read_preference"),
            }
            if "tls" in self.spec.options:
                client_kwargs["tls"] = bool(self.spec.options["tls"])
            if self.spec.options.get("ssl_ca") or self.spec.options.get("tlsCAFile"):
                client_kwargs["tlsCAFile"] = str(self.spec.options.get("tlsCAFile") or self.spec.options.get("ssl_ca"))
            self._client = MongoClient(self._build_uri(), **client_kwargs)
        return self._client

    def _ping_sync(self, timeout_ms: int) -> Tuple[bool, str]:
        client = self._get_client()
        db = client[self.spec.database_name]
        db.command({"ping": 1})
        return True, f"Connected successfully to mongodb ({self.spec.database_name})."

    async def test_connection(self, timeout_seconds: float = 5.0) -> Tuple[bool, str]:
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._ping_sync, int(timeout_seconds * 1000)),
                timeout=timeout_seconds,
            )
        except Exception as exc:
            return False, f"MongoDB connection test failed: {exc}"

    def _discover_sync(self) -> DiscoveredSchema:
        client = self._get_client()
        db = client[self.spec.database_name]
        try:
            server_info = client.server_info()
            version = str(server_info.get("version", "unknown"))
        except Exception:
            version = "unknown"

        coll_names = [c for c in db.list_collection_names() if not c.startswith("system.")]
        tables: List[TableMeta] = []
        for cname in coll_names:
            col_types: Dict[str, set[str]] = {}
            for doc in db[cname].find({}).limit(25):
                for k, v in doc.items():
                    col_types.setdefault(str(k), set()).add(type(v).__name__)
            cols = [
                ColumnMeta(
                    name=k,
                    data_type=", ".join(sorted(types)) if types else "unknown",
                    nullable=(k != "_id"),
                    primary_key=(k == "_id"),
                )
                for k, types in sorted(col_types.items())
            ]
            tables.append(
                TableMeta(
                    schema_name=self.spec.database_name,
                    name=cname,
                    columns=cols,
                    primary_keys=["_id"] if "_id" in col_types else [],
                )
            )

        return DiscoveredSchema(
            engine="mongodb",
            version=version,
            schemas=[self.spec.database_name],
            tables=tables,
            relationships=[],
        )

    async def discover_schema(self, timeout_seconds: float = 10.0) -> DiscoveredSchema:
        return await asyncio.wait_for(
            asyncio.to_thread(self._discover_sync),
            timeout=timeout_seconds,
        )

    def _execute_query_sync(self, query_spec: QuerySpec, timeout_ms: int, max_rows: int) -> List[Dict[str, Any]]:
        client = self._get_client()
        db = client[self.spec.database_name]
        coll_name = validate_identifier(query_spec.table, "collection")
        coll = db[coll_name]

        mongo_filter: Dict[str, Any] = {}
        for flt in query_spec.filters:
            if flt.table and flt.table != query_spec.table:
                continue
            safe_col = validate_identifier(flt.column, "filter field")
            if flt.operator == "eq":
                if isinstance(flt.value, str):
                    mongo_filter[safe_col] = {"$regex": f"^{_escape_regex_literal(flt.value)}$", "$options": "i"}
                else:
                    mongo_filter[safe_col] = flt.value
            elif flt.operator == "ilike":
                mongo_filter[safe_col] = {"$regex": _escape_regex_literal(str(flt.value)), "$options": "i"}
            elif flt.operator == "in":
                vals = list(flt.value) if isinstance(flt.value, (list, tuple, set)) else [flt.value]
                mongo_filter[safe_col] = {"$in": vals}

        # Enforce read-only projection
        projection: Dict[str, int] = {"_id": 0}
        for phys_col in query_spec.select_columns.values():
            projection[validate_identifier(phys_col, "projection field")] = 1
        for j in query_spec.joins:
            projection[validate_identifier(j.left_column, "join key field")] = 1

        effective_limit = max(1, min(int(query_spec.limit), int(max_rows)))
        cursor = coll.find(mongo_filter, projection).max_time_ms(timeout_ms).limit(effective_limit)
        if query_spec.order_by:
            safe_order = validate_identifier(query_spec.order_by, "sort field")
            cursor = cursor.sort(safe_order, -1 if query_spec.descending else 1)

        raw_docs = list(cursor)
        results: List[Dict[str, Any]] = []
        for doc in raw_docs:
            row: Dict[str, Any] = {}
            for out_alias, phys_col in query_spec.select_columns.items():
                val = doc.get(phys_col)
                row[out_alias] = str(val) if phys_col == "_id" and val is not None else val

            # Resolve optional read-only joins from related collections
            for j in query_spec.joins:
                r_coll = db[validate_identifier(j.right_table, "join collection")]
                join_val = doc.get(j.left_column)
                r_doc = r_coll.find_one({validate_identifier(j.right_column, "join field"): join_val}) if join_val is not None else None
                for out_alias, phys_col in j.select_columns.items():
                    row[out_alias] = r_doc.get(phys_col) if isinstance(r_doc, dict) else None
            results.append(row)

        return results

    async def execute_read_only(
        self,
        query_spec: QuerySpec,
        timeout_seconds: float = 5.0,
        max_rows: int = 50,
    ) -> QueryResult:
        t0 = time.perf_counter()
        rows = await asyncio.wait_for(
            asyncio.to_thread(
                self._execute_query_sync,
                query_spec,
                int(timeout_seconds * 1000),
                max_rows,
            ),
            timeout=timeout_seconds,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return QueryResult(
            records=rows,
            row_count=len(rows),
            engine="mongodb",
            latency_ms=round(elapsed_ms, 2),
            truncated=len(rows) >= min(query_spec.limit, max_rows),
        )

    async def close(self) -> None:
        if self._client is not None and hasattr(self._client, "close"):
            client = self._client
            self._client = None
            await asyncio.to_thread(client.close)

