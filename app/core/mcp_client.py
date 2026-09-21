import asyncio
import json
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


class MCPClientError(RuntimeError):
    pass


class MCPUnavailableError(MCPClientError):
    pass


class MCPTimeoutError(MCPClientError):
    pass


@dataclass(frozen=True)
class MCPTool:
    name: str
    description: str
    input_schema: dict[str, Any]


class MCPStdioClient:
    """短生命周期 stdio MCP Client；每次连接都保证关闭 server 子进程。"""

    def __init__(
        self,
        command: str,
        args: list[str],
        cwd: str | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self.parameters = StdioServerParameters(command=command, args=args, cwd=cwd)
        self.timeout_seconds = timeout_seconds
        self._stdio_context = None
        self._session_context = None
        self.session: ClientSession | None = None

    async def __aenter__(self) -> "MCPStdioClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.close()

    async def connect(self) -> None:
        if self.session is not None:
            return
        try:
            self._stdio_context = stdio_client(self.parameters)
            read_stream, write_stream = await self._stdio_context.__aenter__()
            self._session_context = ClientSession(read_stream, write_stream)
            self.session = await self._session_context.__aenter__()
            await self._run_with_timeout(self.session.initialize())
        except Exception as exc:
            await self.close()
            raise MCPUnavailableError(f"MCP server unavailable: {exc}") from exc

    async def list_tools(self) -> list[MCPTool]:
        session = self._require_session()
        try:
            result = await self._run_with_timeout(session.list_tools())
        except TimeoutError as exc:
            raise MCPTimeoutError("MCP list_tools timed out") from exc
        except MCPClientError:
            raise
        except Exception as exc:
            raise MCPUnavailableError(f"MCP tool discovery failed: {exc}") from exc
        return [
            MCPTool(
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.inputSchema,
            )
            for tool in result.tools
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        session = self._require_session()
        try:
            result = await self._run_with_timeout(session.call_tool(name, arguments))
        except TimeoutError as exc:
            raise MCPTimeoutError(f"MCP tool {name} timed out") from exc
        except MCPClientError:
            raise
        except Exception as exc:
            raise MCPUnavailableError(f"MCP tool {name} failed: {exc}") from exc
        if result.isError:
            raise MCPClientError(f"MCP tool {name} returned an error")
        if result.structuredContent is not None:
            return result.structuredContent
        return _content_to_object(result.content)

    async def close(self) -> None:
        session_context, stdio_context = self._session_context, self._stdio_context
        self.session = None
        self._session_context = None
        self._stdio_context = None
        if session_context is not None:
            with suppress(BaseException):
                await session_context.__aexit__(None, None, None)
        if stdio_context is not None:
            with suppress(BaseException):
                await stdio_context.__aexit__(None, None, None)

    async def _run_with_timeout(self, awaitable):
        try:
            return await asyncio.wait_for(awaitable, timeout=self.timeout_seconds)
        except TimeoutError:
            raise

    def _require_session(self) -> ClientSession:
        if self.session is None:
            raise MCPUnavailableError("MCP client is not connected")
        return self.session


def _content_to_object(content: list[Any]) -> dict[str, Any]:
    texts = [item.text for item in content if hasattr(item, "text")]
    text = "\n".join(texts).strip()
    if not text:
        return {"content": []}
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {"text": text}
    return value if isinstance(value, dict) else {"value": value}