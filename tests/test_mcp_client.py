import sys

import pytest

from app.core.mcp_client import MCPStdioClient, MCPTimeoutError, MCPUnavailableError


def server_client(timeout_seconds: float = 10.0) -> MCPStdioClient:
    return MCPStdioClient(
        command=sys.executable,
        args=["-m", "app.mcp_server"],
        timeout_seconds=timeout_seconds,
    )


@pytest.mark.asyncio
async def test_mcp_server_tools_discovery_and_structured_invocation():
    async with server_client() as client:
        tools = await client.list_tools()
        result = await client.call_tool(
            "query_meeting_rooms", {"date": "明天", "time": "下午"}
        )

    assert {tool.name for tool in tools} == {"query_meeting_rooms", "query_company_notice"}
    assert result == {
        "date": "明天",
        "time": "下午",
        "rooms": ["Room-A", "Room-C"],
        "source": "demo_admin_system",
    }


@pytest.mark.asyncio
async def test_mcp_client_shutdown_closes_session():
    client = server_client()
    async with client:
        assert client.session is not None
    assert client.session is None


@pytest.mark.asyncio
async def test_mcp_server_unavailable_is_controlled():
    client = MCPStdioClient(
        command=sys.executable,
        args=["-c", "raise SystemExit(3)"],
        timeout_seconds=1.0,
    )

    with pytest.raises(MCPUnavailableError):
        await client.connect()
    assert client.session is None


@pytest.mark.asyncio
async def test_mcp_tool_timeout_is_captured():
    client = MCPStdioClient(
        command=sys.executable,
        args=["-m", "tests.mcp_slow_server"],
        timeout_seconds=10.0,
    )
    async with client:
        client.timeout_seconds = 0.05
        with pytest.raises(MCPTimeoutError):
            await client.call_tool("slow_tool", {})