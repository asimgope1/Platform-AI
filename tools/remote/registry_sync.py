"""
Synchronizes validated remote Django business tools into FastAPI's ToolRegistry.
"""
from __future__ import annotations

import copy
import time
from typing import Any, Callable, Awaitable, Dict, List, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession

from platform_core.observability.logging import logger
from platform_core.security.tenant_context import TenantContext
from tools.registry import ToolRegistry, tool_registry
from tools.remote.django_client import DjangoRemoteToolClient

# Snapshots of local tools overridden by remote_django, keyed by id(registry) -> name -> (defn, handler)
_LOCAL_BACKUPS: Dict[int, Dict[str, Tuple[Any, Callable[..., Awaitable[Any]]]]] = {}
_SYNC_STATE: Dict[int, Dict[str, Any]] = {}


def get_remote_sync_state(registry: ToolRegistry = tool_registry) -> Dict[str, Any]:
    """Return last sync timestamp, tool count, stale flag, and any last refresh error."""
    return dict(_SYNC_STATE.get(id(registry), {"last_synced_at": None, "tool_count": 0, "stale": False, "last_error": None}))


async def _has_local_active_db(db_session: Optional[AsyncSession], tenant_id: Optional[str]) -> bool:
    if db_session is None or not tenant_id:
        return False
    try:
        from sqlalchemy import select
        from data_access.models import TenantDatabaseConfig

        res = await db_session.execute(
            select(TenantDatabaseConfig.id).where(
                TenantDatabaseConfig.tenant_id == tenant_id,
                TenantDatabaseConfig.is_active == True,  # noqa: E712
                TenantDatabaseConfig.mapping_validated == True,  # noqa: E712
            ).limit(1)
        )
        return res.scalar_one_or_none() is not None
    except Exception:
        return False


def _make_remote_handler(
    tool_name: str,
    client: DjangoRemoteToolClient,
    *,
    local_fallback: Optional[Callable[..., Awaitable[Any]]] = None,
    prefer_remote: bool = False,
) -> Callable[..., Awaitable[Dict[str, Any]]]:
    async def _remote_tool_handler(
        *,
        tenant_context: Optional[TenantContext] = None,
        db_session: Optional[AsyncSession] = None,
        authorized_customer_id: Optional[str] = None,
        call_id: Optional[str] = None,
        **tool_kwargs: Any,
    ) -> Dict[str, Any]:
        # Preserve Phase 1 per-tenant external SQL/Mongo DB support unless remote is preferred
        if (
            not prefer_remote
            and local_fallback is not None
            and tenant_context is not None
            and await _has_local_active_db(db_session, tenant_context.tenant_id)
        ):
            local_args = dict(tool_kwargs)
            if "order_number" in local_args and "order_id" not in local_args:
                local_args["order_id"] = local_args.pop("order_number")
            if "complaint_reference" in local_args and "ticket_number" not in local_args:
                local_args["ticket_number"] = local_args.pop("complaint_reference")
            return await local_fallback(
                tenant_context=tenant_context,
                db_session=db_session,
                authorized_customer_id=authorized_customer_id,
                **local_args,
            )

        return await client.execute_tool(
            tool_name,
            tool_kwargs,
            context=tenant_context,
            authorized_customer_id=authorized_customer_id,
            call_id=call_id,
        )

    _remote_tool_handler.__name__ = f"remote_{tool_name}"
    return _remote_tool_handler


async def sync_remote_django_tools(
    client: DjangoRemoteToolClient,
    *,
    registry: ToolRegistry = tool_registry,
    prefer_remote: bool = False,
) -> List[str]:
    """
    Fetch, validate, and register Django business tools into FastAPI's ToolRegistry.
    Handles catalog refresh cleanly:
      • Preserves last known good catalog if Django refresh fails (marks state stale=True).
      • Prunes stale remote tools removed from Django, restoring snapshotted local tools if applicable.
    """
    reg_id = id(registry)
    state = _SYNC_STATE.setdefault(
        reg_id, {"last_synced_at": None, "tool_count": 0, "stale": False, "last_error": None}
    )
    try:
        definitions = await client.fetch_and_validate_catalog()
    except Exception as exc:
        if state.get("tool_count", 0) > 0:
            state["stale"] = True
        state["last_error"] = str(exc)
        raise

    backups = _LOCAL_BACKUPS.setdefault(reg_id, {})
    new_names = {d.function.name for d in definitions}

    # Prune stale previously-synced remote tools no longer present in the new catalog
    for existing_name, existing_src in list(registry.list_tools_with_sources().items()):
        if existing_src == "remote_django" and existing_name not in new_names:
            if existing_name in backups:
                orig_defn, orig_handler = backups.pop(existing_name)
                registry.register(
                    name=orig_defn.function.name,
                    description=orig_defn.function.description,
                    parameters=orig_defn.function.parameters,
                    handler=orig_handler,
                    required_scope=orig_defn.required_scope,
                    source="local",
                )
            else:
                registry.unregister(existing_name)

    # Restore shadowed local tools (e.g. register_complaint) if remote counterpart was removed
    if "create_complaint" not in new_names and "register_complaint" in backups and registry.get_tool("register_complaint") is None:
        orig_defn, orig_handler = backups.pop("register_complaint")
        registry.register(
            name=orig_defn.function.name,
            description=orig_defn.function.description,
            parameters=orig_defn.function.parameters,
            handler=orig_handler,
            required_scope=orig_defn.required_scope,
            source="local",
        )

    registered_names: List[str] = []
    for defn in definitions:
        name = defn.function.name
        existing_local = registry.get_handler(name)
        existing_defn = registry.get_tool(name)
        existing_source = registry.get_tool_source(name)

        if existing_source == "local" and existing_local is not None and existing_defn is not None:
            backups[name] = (copy.deepcopy(existing_defn), existing_local)

        local_fallback = backups[name][1] if name in backups else None
        handler = _make_remote_handler(
            name,
            client,
            local_fallback=local_fallback,
            prefer_remote=prefer_remote,
        )
        registry.register(
            name=name,
            description=defn.function.description,
            parameters=defn.function.parameters,
            handler=handler,
            required_scope=defn.required_scope,
            source="remote_django",
        )
        registered_names.append(name)

    # Suppress redundant local register_complaint when remote create_complaint is active
    if "create_complaint" in new_names and registry.get_tool_source("register_complaint") == "local":
        orig_local = registry.get_handler("register_complaint")
        orig_defn = registry.get_tool("register_complaint")
        if orig_local is not None and orig_defn is not None:
            backups.setdefault("register_complaint", (copy.deepcopy(orig_defn), orig_local))
        registry.unregister("register_complaint")

    state["last_synced_at"] = time.time()
    state["tool_count"] = len(registered_names)
    state["stale"] = False
    state["last_error"] = None

    logger.info(
        f"[RemoteTools] Synchronized {len(registered_names)} validated tools from {client.base_url}: "
        f"{', '.join(registered_names)}"
    )
    return registered_names

