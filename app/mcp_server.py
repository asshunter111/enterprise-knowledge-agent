"""Small standalone MCP tool server entry point.

The optional dependency is kept out of the normal pytest path. Install ``mcp``
and run this module to expose the administrative tools over stdio.
"""

from app.core.tools import query_meeting_rooms as _query_meeting_rooms


def create_server():
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError("MCP support requires the optional 'mcp' package") from exc

    server = FastMCP("enterprise-admin")

    @server.tool()
    async def query_meeting_rooms(date: str, time: str) -> dict:
        return await _query_meeting_rooms(date, time)

    @server.tool()
    async def query_company_notice(topic: str) -> dict:
        if not topic.strip():
            raise ValueError("topic is required")
        return {"topic": topic, "notices": [], "source": "demo_notice_system"}

    return server


if __name__ == "__main__":
    create_server().run(transport="stdio")
