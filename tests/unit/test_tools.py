import pytest
from tools.registry import tool_registry
from tools.executor import ToolExecutor
from platform_core.security.tenant_context import TenantContext


@pytest.mark.asyncio
async def test_registered_tools():
    tool_defs = tool_registry.get_all_tool_schemas()
    names = [t["function"]["name"] for t in tool_defs]
    assert "count_open_purchase_requests" in names
    assert "get_customer_info" in names


@pytest.mark.asyncio
async def test_tool_executor():
    executor = ToolExecutor()
    ctx = TenantContext(
        tenant_id="tenant-1",
        tenant_name="Corp",
        api_key_id="k-1",
        scopes={"tools:execute"},
        is_admin=False,
    )

    res = await executor.execute_tool(
        call_id="call_1",
        name="count_open_purchase_requests",
        arguments_json='{"filter_status": "open"}',
        context=ctx,
    )
    assert res.is_error is False
    assert res.output["status"] == "open"
    assert "count" in res.output
