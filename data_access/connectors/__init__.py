"""Database adapters and connection pooling for external customer databases."""
from data_access.connectors.base import (
    BaseDatabaseAdapter,
    ColumnMeta,
    ConnectionConfigSpec,
    DiscoveredSchema,
    ForeignKeyMeta,
    IndexMeta,
    QueryFilter,
    QueryJoin,
    QueryResult,
    QuerySpec,
    RelationshipMeta,
    TableMeta,
)
from data_access.connectors.sql_adapter import SQLAlchemyCoreAdapter
from data_access.connectors.mongodb_adapter import MongoDBAdapter
from data_access.connectors.connection_manager import ConnectionManager, connection_manager

__all__ = [
    "BaseDatabaseAdapter",
    "ColumnMeta",
    "ConnectionConfigSpec",
    "DiscoveredSchema",
    "ForeignKeyMeta",
    "IndexMeta",
    "QueryFilter",
    "QueryJoin",
    "QueryResult",
    "QuerySpec",
    "RelationshipMeta",
    "TableMeta",
    "SQLAlchemyCoreAdapter",
    "MongoDBAdapter",
    "ConnectionManager",
    "connection_manager",
]

