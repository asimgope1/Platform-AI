"""Normalized database adapter interface and metadata/query contracts."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field


class ColumnMeta(BaseModel):
    name: str
    data_type: str
    nullable: bool = True
    primary_key: bool = False
    default: Optional[str] = None


class ForeignKeyMeta(BaseModel):
    column: str
    referred_schema: Optional[str] = None
    referred_table: str
    referred_column: str
    constraint_name: Optional[str] = None


class IndexMeta(BaseModel):
    name: Optional[str] = None
    columns: List[str] = Field(default_factory=list)
    unique: bool = False


class RelationshipMeta(BaseModel):
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    cardinality: str = "many_to_one"


class TableMeta(BaseModel):
    schema_name: Optional[str] = None
    name: str
    columns: List[ColumnMeta] = Field(default_factory=list)
    primary_keys: List[str] = Field(default_factory=list)
    foreign_keys: List[ForeignKeyMeta] = Field(default_factory=list)
    indexes: List[IndexMeta] = Field(default_factory=list)


class DiscoveredSchema(BaseModel):
    engine: str
    version: str = "unknown"
    schemas: List[str] = Field(default_factory=list)
    tables: List[TableMeta] = Field(default_factory=list)
    relationships: List[RelationshipMeta] = Field(default_factory=list)


class ConnectionConfigSpec(BaseModel):
    tenant_id: str
    config_id: str
    engine: Literal["postgresql", "mysql", "mssql", "oracle", "mongodb", "sqlite"]
    host: Optional[str] = None
    port: Optional[int] = None
    database_name: str
    username: Optional[str] = None
    password: Optional[str] = None
    connection_uri: Optional[str] = None
    schema_name: Optional[str] = None
    read_only: bool = True
    options: Dict[str, Any] = Field(default_factory=dict)


class QueryFilter(BaseModel):
    column: str
    operator: Literal["eq", "ilike", "in"] = "eq"
    value: Any
    table: Optional[str] = None


class QueryJoin(BaseModel):
    right_table: str
    right_schema: Optional[str] = None
    left_column: str
    right_column: str
    # Maps output_field_name -> physical column name on right_table
    select_columns: Dict[str, str] = Field(default_factory=dict)


class QuerySpec(BaseModel):
    table: str
    schema_name: Optional[str] = None
    # Maps output_field_name -> physical column name on main table
    select_columns: Dict[str, str]
    filters: List[QueryFilter] = Field(default_factory=list)
    joins: List[QueryJoin] = Field(default_factory=list)
    order_by: Optional[str] = None
    descending: bool = True
    limit: int = 20


class QueryResult(BaseModel):
    records: List[Dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    engine: str
    latency_ms: float = 0.0
    truncated: bool = False


class BaseDatabaseAdapter(ABC):
    """Engine-agnostic read-only database adapter contract."""

    def __init__(self, spec: ConnectionConfigSpec):
        self.spec = spec

    @abstractmethod
    async def test_connection(self, timeout_seconds: float = 5.0) -> Tuple[bool, str]:
        """Verify database connectivity and read-only execution health."""

    @abstractmethod
    async def discover_schema(self, timeout_seconds: float = 10.0) -> DiscoveredSchema:
        """Read-only inspection of schemas, tables/collections, columns, keys, and relationships."""

    @abstractmethod
    async def execute_read_only(
        self,
        query_spec: QuerySpec,
        timeout_seconds: float = 5.0,
        max_rows: int = 50,
    ) -> QueryResult:
        """Execute a strictly parameterized read-only query and normalize rows."""

    @abstractmethod
    async def close(self) -> None:
        """Safely dispose pooled connections and release resources."""

