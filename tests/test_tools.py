import pytest

from app.core.tools import ToolExecutionError, build_default_tool_registry


@pytest.mark.asyncio
async def test_business_tool_registry_executes_structured_result():
    result = await build_default_tool_registry().execute("get_leave_balance", employee_id="u-1")
    assert result["employee_id"] == "u-1"
    assert result["annual_leave_days"] == 12


@pytest.mark.asyncio
async def test_business_tool_registry_rejects_unknown_and_invalid_tools():
    registry = build_default_tool_registry()
    with pytest.raises(ToolExecutionError, match="unknown tool"):
        await registry.execute("missing")
    with pytest.raises(ToolExecutionError, match="invalid arguments"):
        await registry.execute("get_leave_balance")
