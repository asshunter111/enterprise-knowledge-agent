import asyncio

from mcp.server.fastmcp import FastMCP

server = FastMCP("slow-test")


@server.tool()
async def slow_tool() -> dict:
    await asyncio.sleep(2)
    return {"status": "done"}


if __name__ == "__main__":
    server.run(transport="stdio")