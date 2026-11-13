"""
Universal Business Entity Mapping & Schema Validation Layer.

Maps tenant-specific database schemas onto six canonical business concepts:
  • customer
  • order
  • payment
  • invoice
  • product
  • support_ticket
"""
from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from data_access.connectors.base import DiscoveredSchema, TableMeta
from data_access.security import validate_identifier

UNIVERSAL_ENTITIES: Dict[str, List[str]] = {
    "customer": [
        "customer_id",
        "name",
        "phone",
        "email",
        "status",
        "tier",
    ],
    "order": [
        "order_id",
        "order_number",
        "customer_id",
        "order_status",
        "tracking_number",
        "carrier",
        "expected_delivery",
        "total_amount",
        "currency",
        "created_at",
    ],
    "payment": [
        "payment_id",
        "order_id",
        "customer_id",
        "payment_status",
        "payment_method",
        "amount",
        "paid_at",
    ],
    "invoice": [
        "invoice_id",
        "invoice_number",
        "order_id",
        "customer_id",
        "invoice_status",
        "total_amount",
        "issued_at",
        "due_at",
    ],
    "product": [
        "product_id",
        "sku",
        "name",
        "category",
        "price",
        "stock_status",
    ],
    "support_ticket": [
        "ticket_id",
        "customer_id",
        "order_id",
        "subject",
        "ticket_status",
        "ticket_status_name",
        "priority",
        "created_at",
        "updated_at",
    ],
    "complaint_phase": [
        "phase_id",
        "phase_name",
        "phase_type",
        "is_initial",
    ],
    "inventory": [
        "inventory_id",
        "product_id",
        "sku",
        "warehouse",
        "quantity_available",
        "reserved_quantity",
        "reorder_level",
        "stock_status",
        "updated_at",
    ],
    "shipment": [
        "shipment_id",
        "order_id",
        "order_number",
        "customer_id",
        "tracking_number",
        "carrier",
        "shipment_status",
        "shipped_at",
        "expected_delivery",
        "delivered_at",
        "origin",
        "destination",
    ],
}

# Minimum required lookup fields per entity when mapped
REQUIRED_ENTITY_FIELDS: Dict[str, List[str]] = {
    "customer": ["customer_id"],
    "order": ["order_number", "order_status"],
    "payment": ["payment_id", "payment_status"],
    "invoice": ["invoice_number", "invoice_status"],
    "product": ["product_id", "name"],
    "support_ticket": ["ticket_id", "ticket_status"],
    "complaint_phase": ["phase_id", "phase_name"],
    "inventory": ["sku", "quantity_available"],
    "shipment": ["tracking_number", "shipment_status"],
}


class EntityRelationSpec(BaseModel):
    target_entity: str
    local_column: str
    target_column: str


class EntityMappingSpec(BaseModel):
    table: str
    schema_name: Optional[str] = None
    primary_key: Optional[str] = None
    # canonical_field_name -> physical_column_name
    fields: Dict[str, str] = Field(default_factory=dict)
    relations: Dict[str, EntityRelationSpec] = Field(default_factory=dict)


class EntityMappingConfig(BaseModel):
    version: int = 1
    entities: Dict[str, EntityMappingSpec] = Field(default_factory=dict)


class MappingValidationResult(BaseModel):
    valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


def _find_table(discovered: DiscoveredSchema, table_name: str, schema_name: Optional[str] = None) -> Optional[TableMeta]:
    matches = [t for t in discovered.tables if t.name.lower() == table_name.lower()]
    if not matches:
        return None
    if schema_name:
        for m in matches:
            if (m.schema_name or "").lower() == schema_name.lower():
                return m
        return None
    return matches[0]


def validate_mapping_against_schema(
    mapping: EntityMappingConfig,
    discovered: DiscoveredSchema,
) -> MappingValidationResult:
    """Strictly validates every mapped table, column, identifier, and relation against the discovered schema."""
    errors: List[str] = []
    warnings: List[str] = []

    if not mapping.entities:
        errors.append("Mapping must define at least one universal business entity.")
        return MappingValidationResult(valid=False, errors=errors, warnings=warnings)

    for entity_name, spec in mapping.entities.items():
        if entity_name not in UNIVERSAL_ENTITIES:
            errors.append(
                f"Unknown universal entity '{entity_name}'. Allowed: {', '.join(sorted(UNIVERSAL_ENTITIES))}."
            )
            continue

        try:
            validate_identifier(spec.table, f"{entity_name}.table")
            if spec.schema_name:
                validate_identifier(spec.schema_name, f"{entity_name}.schema_name")
            if spec.primary_key:
                validate_identifier(spec.primary_key, f"{entity_name}.primary_key")
        except Exception as exc:
            errors.append(str(exc))
            continue

        table_meta = _find_table(discovered, spec.table, spec.schema_name)
        if not table_meta:
            errors.append(
                f"Mapped table '{spec.table}' for entity '{entity_name}' does not exist in discovered schema."
            )
            continue

        actual_cols = {c.name.lower(): c.name for c in table_meta.columns}
        allowed_fields = set(UNIVERSAL_ENTITIES[entity_name])

        for canonical_field, phys_col in spec.fields.items():
            if canonical_field not in allowed_fields:
                warnings.append(
                    f"Non-standard field '{canonical_field}' mapped under entity '{entity_name}'."
                )
            try:
                validate_identifier(phys_col, f"{entity_name}.fields.{canonical_field}")
            except Exception as exc:
                errors.append(str(exc))
                continue

            if phys_col.lower() not in actual_cols:
                errors.append(
                    f"Column '{phys_col}' mapped to '{entity_name}.{canonical_field}' "
                    f"was not found in table '{table_meta.name}'."
                )

        if spec.primary_key and spec.primary_key.lower() not in actual_cols:
            errors.append(
                f"Primary key column '{spec.primary_key}' for entity '{entity_name}' "
                f"was not found in table '{table_meta.name}'."
            )

        for req_field in REQUIRED_ENTITY_FIELDS.get(entity_name, []):
            if req_field not in spec.fields:
                errors.append(
                    f"Entity '{entity_name}' is missing required canonical field '{req_field}'."
                )

        for rel_name, rel in spec.relations.items():
            if rel.target_entity not in mapping.entities:
                errors.append(
                    f"Relation '{rel_name}' on '{entity_name}' references unmapped entity '{rel.target_entity}'."
                )
                continue
            target_spec = mapping.entities[rel.target_entity]
            target_meta = _find_table(discovered, target_spec.table, target_spec.schema_name)
            if rel.local_column.lower() not in actual_cols:
                errors.append(
                    f"Relation '{rel_name}' local_column '{rel.local_column}' missing on '{table_meta.name}'."
                )
            if target_meta:
                target_cols = {c.name.lower() for c in target_meta.columns}
                if rel.target_column.lower() not in target_cols:
                    errors.append(
                        f"Relation '{rel_name}' target_column '{rel.target_column}' missing on '{target_meta.name}'."
                    )

    return MappingValidationResult(valid=(len(errors) == 0), errors=errors, warnings=warnings)


def suggest_entity_mapping(discovered: DiscoveredSchema) -> EntityMappingConfig:
    """Heuristically propose an EntityMappingConfig from discovered table/column names."""
    table_hints = {
        "customer": ("customer", "customers", "client", "clients", "client_master", "account"),
        "order": ("order", "orders", "sales_orders", "purchase_records", "purchases"),
        "payment": ("payment", "payments", "transactions"),
        "invoice": ("invoice", "invoices", "bills"),
        "product": ("product", "products", "catalog", "items"),
        "support_ticket": ("ticket", "tickets", "support_tickets", "cases"),
        "inventory": ("inventory", "stock", "warehouse_stock", "inventory_items"),
        "shipment": ("shipment", "shipments", "deliveries", "dispatches", "fulfillments"),
    }
    col_hints: Dict[str, Dict[str, tuple[str, ...]]] = {
        "customer": {
            "customer_id": ("customer_id", "client_pk", "client_id", "id"),
            "name": ("full_name", "customer_name", "client_name", "name"),
            "phone": ("mobile_number", "phone", "phone_number", "mobile", "contact_number"),
            "email": ("email", "email_address", "mail"),
        },
        "order": {
            "order_id": ("id", "order_pk"),
            "order_number": ("order_id", "order_number", "order_no", "po_ref", "order_ref"),
            "customer_id": ("customer_id", "client_id", "cust_id"),
            "order_status": ("order_status", "current_state", "status", "state"),
            "tracking_number": ("tracking_number", "tracking_no", "awb"),
            "expected_delivery": ("expected_delivery", "delivery_eta", "ship_date", "eta"),
            "total_amount": ("total_amount", "amount", "total_cost", "price"),
        },
        "payment": {
            "payment_id": ("payment_id", "txn_id", "transaction_id", "id"),
            "order_id": ("order_id", "order_number", "order_ref"),
            "customer_id": ("customer_id", "client_id"),
            "payment_status": ("payment_status", "status", "txn_status"),
            "payment_method": ("payment_method", "method", "channel"),
            "amount": ("amount", "paid_amount", "total_amount"),
            "paid_at": ("paid_at", "payment_date", "created_at"),
        },
        "invoice": {
            "invoice_id": ("invoice_id", "id"),
            "invoice_number": ("invoice_number", "invoice_no", "bill_number", "bill_no"),
            "order_id": ("order_id", "order_number"),
            "customer_id": ("customer_id", "client_id"),
            "invoice_status": ("invoice_status", "status"),
            "total_amount": ("total_amount", "amount", "bill_amount"),
            "issued_at": ("issued_at", "issue_date", "created_at"),
            "due_at": ("due_at", "due_date"),
        },
        "inventory": {
            "inventory_id": ("inventory_id", "id"),
            "product_id": ("product_id", "item_id"),
            "sku": ("sku", "item_code", "product_code"),
            "warehouse": ("warehouse", "warehouse_code", "location"),
            "quantity_available": ("quantity_available", "available_qty", "stock_qty", "qty"),
            "stock_status": ("stock_status", "status"),
        },
        "shipment": {
            "shipment_id": ("shipment_id", "id"),
            "order_id": ("order_id", "order_pk"),
            "order_number": ("order_number", "order_no", "order_ref"),
            "customer_id": ("customer_id", "client_id"),
            "tracking_number": ("tracking_number", "tracking_no", "awb"),
            "carrier": ("carrier", "courier", "shipping_provider"),
            "shipment_status": ("shipment_status", "delivery_status", "status"),
            "expected_delivery": ("expected_delivery", "eta", "delivery_eta"),
        },
    }

    entities: Dict[str, EntityMappingSpec] = {}
    for ent, t_syns in table_hints.items():
        for tbl in discovered.tables:
            tl = tbl.name.lower()
            if tl in t_syns or any(s in tl for s in t_syns):
                cols_lower = {c.name.lower(): c.name for c in tbl.columns}
                fields: Dict[str, str] = {}
                for canon, c_syns in col_hints.get(ent, {}).items():
                    for candidate in c_syns:
                        if candidate in cols_lower:
                            fields[canon] = cols_lower[candidate]
                            break
                if fields:
                    entities[ent] = EntityMappingSpec(
                        table=tbl.name,
                        schema_name=tbl.schema_name,
                        primary_key=tbl.primary_keys[0] if tbl.primary_keys else None,
                        fields=fields,
                    )
                    break

    return EntityMappingConfig(version=1, entities=entities)

