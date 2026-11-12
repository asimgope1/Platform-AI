"""
SQLAlchemy Core adapter supporting PostgreSQL, MySQL, Microsoft SQL Server, Oracle, and SQLite.
Enforces connection pooling, parameterized queries, identifier allowlisting, query timeouts,
and read-only execution without mutating customer schemas.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import sqlalchemy as sa
from sqlalchemy.engine import URL

from data_access.connectors.base import (
    BaseDatabaseAdapter,
    ColumnMeta,
    ConnectionConfigSpec,
    DiscoveredSchema,
    ForeignKeyMeta,
    IndexMeta,
    QueryResult,
    QuerySpec,
    RelationshipMeta,
    TableMeta,
)
from data_access.security import validate_identifier
from platform_core.errors.exceptions import PermissionDeniedError, ValidationError


def build_sqlalchemy_url(spec: ConnectionConfigSpec) -> str | URL:
    """Construct a SQLAlchemy connection URL cleanly for supported relational engines."""
    if spec.connection_uri:
        uri = spec.connection_uri.strip()
        # Strip async-only prefixes so synchronous sqlalchemy inspect() and Core work uniformly
        if uri.startswith("sqlite+aiosqlite:///"):
            return uri.replace("sqlite+aiosqlite:///", "sqlite:///", 1)
        if uri.startswith("postgresql+asyncpg://"):
            return uri.replace("postgresql+asyncpg://", "postgresql+psycopg2://", 1)
        return uri

    eng = spec.engine.lower()
    if eng == "sqlite":
        db = spec.database_name
        if db.startswith("sqlite:///"):
            return db
        return f"sqlite:///{db}"

    driver_map = {
        "postgresql": "postgresql+psycopg2",
        "mysql": "mysql+pymysql",
        "mssql": "mssql+pyodbc",
        "oracle": "oracle+oracledb",
    }
    if eng not in driver_map:
        raise ValidationError(f"Unsupported relational database engine: '{spec.engine}'.")

    default_ports = {"postgresql": 5432, "mysql": 3306, "mssql": 1433, "oracle": 1521}
    query_params: Dict[str, str] = {}
    if eng == "mssql":
        query_params["driver"] = str(spec.options.get("odbc_driver", "ODBC Driver 18 for SQL Server"))
        if "encrypt" in spec.options:
            query_params["Encrypt"] = str(spec.options["encrypt"])
        if "trust_server_certificate" in spec.options:
            query_params["TrustServerCertificate"] = str(spec.options["trust_server_certificate"])
    if eng == "postgresql":
        for ssl_opt in ("sslmode", "sslrootcert", "sslcert", "sslkey"):
            if spec.options.get(ssl_opt):
                query_params[ssl_opt] = str(spec.options[ssl_opt])
    elif spec.options.get("sslmode"):
        query_params["sslmode"] = str(spec.options["sslmode"])

    return URL.create(
        drivername=driver_map[eng],
        username=spec.username,
        password=spec.password,
        host=spec.host or "localhost",
        port=spec.port or default_ports[eng],
        database=spec.database_name,
        query=query_params or None,
    )


class SQLAlchemyCoreAdapter(BaseDatabaseAdapter):
    """Relational database adapter using SQLAlchemy Core and pooled connections."""

    def __init__(self, spec: ConnectionConfigSpec):
        super().__init__(spec)
        if not spec.read_only:
            raise PermissionDeniedError("External customer databases must remain read_only=True by default.")
        self._url = build_sqlalchemy_url(spec)
        self._engine: Optional[sa.Engine] = None

    def _get_engine(self) -> sa.Engine:
        if self._engine is None:
            url_str = str(self._url)
            is_sqlite = url_str.startswith("sqlite")
            kwargs: Dict[str, Any] = {"pool_pre_ping": True}
            if is_sqlite:
                kwargs["connect_args"] = {"check_same_thread": False}
            else:
                kwargs["pool_size"] = int(self.spec.options.get("pool_size", 5))
                kwargs["max_overflow"] = int(self.spec.options.get("max_overflow", 5))
                kwargs["pool_recycle"] = int(self.spec.options.get("pool_recycle", 1800))
                if self.spec.engine == "mysql":
                    ssl_args = {
                        k: self.spec.options[opt]
                        for opt, k in (("ssl_ca", "ca"), ("ssl_cert", "cert"), ("ssl_key", "key"))
                        if self.spec.options.get(opt)
                    }
                    if ssl_args:
                        kwargs["connect_args"] = {"ssl": ssl_args}
            self._engine = sa.create_engine(self._url, **kwargs)
        return self._engine

    def _apply_read_only_and_timeout(self, conn: sa.Connection, timeout_ms: int) -> None:
        dialect = conn.dialect.name
        if dialect == "postgresql":
            conn.execute(sa.text("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY"))
            conn.execute(sa.text(f"SET LOCAL statement_timeout = {int(timeout_ms)}"))
        elif dialect == "mysql":
            conn.execute(sa.text("SET SESSION TRANSACTION READ ONLY"))
            conn.execute(sa.text(f"SET SESSION MAX_EXECUTION_TIME = {int(timeout_ms)}"))
        elif dialect == "mssql":
            conn.execute(sa.text(f"SET LOCK_TIMEOUT {int(timeout_ms)}"))
        elif dialect == "oracle":
            conn.execute(sa.text("SET TRANSACTION READ ONLY"))
        elif dialect == "sqlite":
            conn.execute(sa.text("PRAGMA query_only = ON"))
            conn.execute(sa.text(f"PRAGMA busy_timeout = {int(timeout_ms)}"))

    def _ping_sync(self, timeout_ms: int) -> Tuple[bool, str]:
        eng = self._get_engine()
        with eng.connect() as conn:
            self._apply_read_only_and_timeout(conn, timeout_ms=timeout_ms)
            probe_sql = "SELECT 1 FROM DUAL" if conn.dialect.name == "oracle" else "SELECT 1"
            conn.execute(sa.text(probe_sql)).scalar()
        return True, f"Connected successfully to {self.spec.engine} ({self.spec.database_name})."

    async def test_connection(self, timeout_seconds: float = 5.0) -> Tuple[bool, str]:
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._ping_sync, int(timeout_seconds * 1000)),
                timeout=timeout_seconds,
            )
        except Exception as exc:
            return False, f"Connection test failed: {exc}"

    def _inspect_version(self, conn: sa.Connection) -> str:
        dialect = conn.dialect.name
        queries = {
            "sqlite": "SELECT sqlite_version()",
            "postgresql": "SELECT version()",
            "mysql": "SELECT VERSION()",
            "mssql": "SELECT @@VERSION",
            "oracle": "SELECT banner FROM v$version WHERE ROWNUM = 1",
        }
        sql = queries.get(dialect)
        if not sql:
            return "unknown"
        try:
            val = conn.execute(sa.text(sql)).scalar()
            return str(val or "unknown")[:160]
        except Exception:
            return "unknown"

    def _discover_sync(self) -> DiscoveredSchema:
        eng = self._get_engine()
        with eng.connect() as conn:
            self._apply_read_only_and_timeout(conn, timeout_ms=10000)
            version = self._inspect_version(conn)
            insp = sa.inspect(conn)

            all_schemas: List[str] = []
            try:
                all_schemas = [s for s in insp.get_schema_names() if s not in ("information_schema", "pg_catalog")]
            except Exception:
                all_schemas = []

            target_schemas: List[Optional[str]]
            if self.spec.schema_name:
                target_schemas = [self.spec.schema_name]
            elif conn.dialect.name in ("sqlite", "mysql"):
                target_schemas = [None]
            else:
                target_schemas = [None] + [s for s in all_schemas if s not in ("public",)]

            tables: List[TableMeta] = []
            relationships: List[RelationshipMeta] = []
            seen_tables = set()

            for sch in target_schemas:
                try:
                    t_names = insp.get_table_names(schema=sch)
                except Exception:
                    continue
                for t_name in t_names:
                    key = (sch or "", t_name)
                    if key in seen_tables:
                        continue
                    seen_tables.add(key)

                    pk_info = insp.get_pk_constraint(t_name, schema=sch) or {}
                    pks = list(pk_info.get("constrained_columns") or [])

                    raw_cols = insp.get_columns(t_name, schema=sch) or []
                    cols = [
                        ColumnMeta(
                            name=c["name"],
                            data_type=str(c.get("type") or "unknown"),
                            nullable=bool(c.get("nullable", True)),
                            primary_key=c["name"] in pks,
                            default=str(c["default"]) if c.get("default") is not None else None,
                        )
                        for c in raw_cols
                    ]

                    raw_fks = insp.get_foreign_keys(t_name, schema=sch) or []
                    fks: List[ForeignKeyMeta] = []
                    for fk in raw_fks:
                        local_cols = fk.get("constrained_columns") or []
                        ref_cols = fk.get("referred_columns") or []
                        ref_table = fk.get("referred_table") or ""
                        ref_schema = fk.get("referred_schema")
                        for lc, rc in zip(local_cols, ref_cols):
                            fks.append(
                                ForeignKeyMeta(
                                    column=lc,
                                    referred_schema=ref_schema,
                                    referred_table=ref_table,
                                    referred_column=rc,
                                    constraint_name=fk.get("name"),
                                )
                            )
                            relationships.append(
                                RelationshipMeta(
                                    from_table=t_name,
                                    from_column=lc,
                                    to_table=ref_table,
                                    to_column=rc,
                                )
                            )

                    raw_idx = insp.get_indexes(t_name, schema=sch) or []
                    indexes = [
                        IndexMeta(
                            name=idx.get("name"),
                            columns=list(idx.get("column_names") or []),
                            unique=bool(idx.get("unique", False)),
                        )
                        for idx in raw_idx
                    ]

                    tables.append(
                        TableMeta(
                            schema_name=sch,
                            name=t_name,
                            columns=cols,
                            primary_keys=pks,
                            foreign_keys=fks,
                            indexes=indexes,
                        )
                    )

            return DiscoveredSchema(
                engine=self.spec.engine,
                version=version,
                schemas=all_schemas or ([self.spec.schema_name] if self.spec.schema_name else ["default"]),
                tables=tables,
                relationships=relationships,
            )

    async def discover_schema(self, timeout_seconds: float = 10.0) -> DiscoveredSchema:
        return await asyncio.wait_for(
            asyncio.to_thread(self._discover_sync),
            timeout=timeout_seconds,
        )

    def _execute_query_sync(self, query_spec: QuerySpec, timeout_ms: int, max_rows: int) -> List[Dict[str, Any]]:
        eng = self._get_engine()

        main_table_name = validate_identifier(query_spec.table, "table")
        main_schema = validate_identifier(query_spec.schema_name, "schema") if query_spec.schema_name else None
        main_tbl = sa.table(
            main_table_name,
            *[sa.column(validate_identifier(phys_col, "column")) for phys_col in query_spec.select_columns.values()],
            schema=main_schema,
        )

        # Preserve column lookup across main table + joined tables
        table_refs: Dict[str, sa.TableClause] = {main_table_name: main_tbl}
        select_exprs = []
        for out_alias, phys_col in query_spec.select_columns.items():
            safe_col = validate_identifier(phys_col, "column")
            safe_alias = validate_identifier(out_alias, "alias")
            select_exprs.append(sa.column(safe_col, _selectable=main_tbl).label(safe_alias))

        from_clause: Any = main_tbl
        for j in query_spec.joins:
            r_table_name = validate_identifier(j.right_table, "join table")
            r_schema = validate_identifier(j.right_schema, "join schema") if j.right_schema else None
            r_cols = list(set(j.select_columns.values()) | {j.right_column})
            r_tbl = sa.table(
                r_table_name,
                *[sa.column(validate_identifier(c, "join column")) for c in r_cols],
                schema=r_schema,
            )
            table_refs[r_table_name] = r_tbl

            left_col = sa.column(validate_identifier(j.left_column, "left join column"), _selectable=main_tbl)
            right_col = sa.column(validate_identifier(j.right_column, "right join column"), _selectable=r_tbl)
            from_clause = sa.join(from_clause, r_tbl, left_col == right_col, isouter=True)

            for out_alias, phys_col in j.select_columns.items():
                safe_col = validate_identifier(phys_col, "join select column")
                safe_alias = validate_identifier(out_alias, "join alias")
                select_exprs.append(sa.column(safe_col, _selectable=r_tbl).label(safe_alias))

        stmt = sa.select(*select_exprs).select_from(from_clause)

        # Bind-parameterized WHERE predicates (never string formatted)
        for flt in query_spec.filters:
            target_tbl = table_refs.get(flt.table or main_table_name, main_tbl)
            safe_col = validate_identifier(flt.column, "filter column")
            col_expr = sa.column(safe_col, _selectable=target_tbl)
            if flt.operator == "eq":
                if isinstance(flt.value, str):
                    stmt = stmt.where(sa.func.lower(col_expr) == flt.value.lower())
                else:
                    stmt = stmt.where(col_expr == flt.value)
            elif flt.operator == "ilike":
                stmt = stmt.where(col_expr.ilike(f"%{flt.value}%"))
            elif flt.operator == "in":
                vals = list(flt.value) if isinstance(flt.value, (list, tuple, set)) else [flt.value]
                stmt = stmt.where(col_expr.in_(vals))

        if query_spec.order_by:
            safe_order_col = validate_identifier(query_spec.order_by, "order_by column")
            ord_expr = sa.column(safe_order_col, _selectable=main_tbl)
            stmt = stmt.order_by(ord_expr.desc() if query_spec.descending else ord_expr.asc())

        effective_limit = max(1, min(int(query_spec.limit), int(max_rows)))
        stmt = stmt.limit(effective_limit)

        with eng.connect() as conn:
            self._apply_read_only_and_timeout(conn, timeout_ms=timeout_ms)
            rows = conn.execute(stmt).mappings().all()
            return [dict(r) for r in rows]

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
            engine=self.spec.engine,
            latency_ms=round(elapsed_ms, 2),
            truncated=len(rows) >= min(query_spec.limit, max_rows),
        )

    async def close(self) -> None:
        if self._engine is not None:
            eng = self._engine
            self._engine = None
            await asyncio.to_thread(eng.dispose)

