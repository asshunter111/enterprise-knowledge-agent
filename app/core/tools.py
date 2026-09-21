import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.config import Settings


class ToolExecutionError(RuntimeError):
    pass


class ToolCallingError(RuntimeError):
    pass


class LeaveBalanceArguments(BaseModel):
    employee_id: str = Field(description="员工 ID")


class ReimbursementStatusArguments(BaseModel):
    reimbursement_id: str = Field(description="报销单 ID；未知时使用 latest")


class MeetingRoomArguments(BaseModel):
    date: str = Field(description="查询日期，例如明天或 2026-09-21")
    time: str = Field(description="查询时间段，例如下午或 14:00-16:00")


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    handler: Callable[..., Awaitable[dict[str, Any]]]
    args_schema: type[BaseModel]


class BusinessToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(
        self,
        name: str,
        description: str,
        handler: Callable[..., Awaitable[dict[str, Any]]],
        args_schema: type[BaseModel],
    ) -> None:
        if not name.strip():
            raise ValueError("tool name cannot be empty")
        self._tools[name] = ToolDefinition(name, description, handler, args_schema)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def validate_arguments(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        definition = self._tools.get(name)
        if definition is None:
            raise ToolExecutionError(f"unknown tool: {name}")
        try:
            return definition.args_schema.model_validate(arguments).model_dump()
        except Exception as exc:
            raise ToolExecutionError(f"invalid arguments for {name}: {exc}") from exc

    def langchain_tools(self) -> list[StructuredTool]:
        return [
            StructuredTool.from_function(
                coroutine=definition.handler,
                name=definition.name,
                description=definition.description,
                args_schema=definition.args_schema,
            )
            for definition in self._tools.values()
        ]

    async def execute(self, name: str, **arguments: Any) -> dict[str, Any]:
        definition = self._tools.get(name)
        if definition is None:
            raise ToolExecutionError(f"unknown tool: {name}")
        try:
            result = await definition.handler(**arguments)
        except TypeError as exc:
            raise ToolExecutionError(f"invalid arguments for {name}: {exc}") from exc
        except Exception as exc:
            raise ToolExecutionError(f"tool {name} failed: {exc}") from exc
        if not isinstance(result, dict):
            raise ToolExecutionError(f"tool {name} returned a non-object result")
        return result


async def get_leave_balance(employee_id: str) -> dict[str, Any]:
    if not employee_id.strip():
        raise ValueError("employee_id is required")
    return {"employee_id": employee_id, "annual_leave_days": 12, "source": "demo_hr_system"}


async def get_reimbursement_status(reimbursement_id: str) -> dict[str, Any]:
    if not reimbursement_id.strip():
        raise ValueError("reimbursement_id is required")
    return {
        "reimbursement_id": reimbursement_id,
        "status": "processing",
        "source": "demo_finance_system",
    }


async def query_meeting_rooms(date: str, time: str) -> dict[str, Any]:
    if not date.strip() or not time.strip():
        raise ValueError("date and time are required")
    return {
        "date": date,
        "time": time,
        "rooms": ["Room-A", "Room-C"],
        "source": "demo_admin_system",
    }


def build_default_tool_registry() -> BusinessToolRegistry:
    registry = BusinessToolRegistry()
    registry.register(
        "get_leave_balance", "查询员工剩余年假", get_leave_balance, LeaveBalanceArguments
    )
    registry.register(
        "get_reimbursement_status",
        "查询报销单状态",
        get_reimbursement_status,
        ReimbursementStatusArguments,
    )
    registry.register(
        "query_meeting_rooms",
        "查询会议室空闲情况",
        query_meeting_rooms,
        MeetingRoomArguments,
    )
    return registry


@dataclass(frozen=True)
class NativeToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: str
    assistant_message: AIMessage
    messages: list[dict[str, Any]]


class ProviderToolCaller:
    """使用 ChatOpenAI 的 provider-native tool calling，失败时由 Agent 回落。"""

    def __init__(self, settings: Settings, registry: BusinessToolRegistry) -> None:
        self.settings = settings
        self.registry = registry
        self._llm = None

    async def propose(
        self, query: str, history: list[dict], user_id: str
    ) -> NativeToolCall | None:
        if not self.settings.llm_api_key:
            raise ToolCallingError("llm api key is not configured")
        messages = self._messages(query, history, user_id)
        try:
            response = await self._get_llm().bind_tools(
                self.registry.langchain_tools(), strict=True, parallel_tool_calls=False
            ).ainvoke(messages)
        except Exception as exc:
            raise ToolCallingError(f"provider tool call failed: {exc}") from exc

        calls = getattr(response, "tool_calls", None) or []
        if not calls:
            return None
        call = calls[0]
        name = call.get("name")
        arguments = call.get("args")
        call_id = call.get("id")
        if not isinstance(name, str) or not isinstance(arguments, dict) or not call_id:
            raise ToolCallingError("provider returned an invalid tool call")
        if name not in self.registry.names():
            raise ToolCallingError(f"provider selected unknown tool: {name}")
        try:
            validated_arguments = self.registry.validate_arguments(name, arguments)
        except ToolExecutionError as exc:
            raise ToolCallingError(str(exc)) from exc
        return NativeToolCall(name, validated_arguments, call_id, response, messages)

    async def complete(self, call: NativeToolCall, result: dict[str, Any]) -> str:
        tool_message = ToolMessage(
            content=json.dumps(result, ensure_ascii=False),
            tool_call_id=call.call_id,
        )
        try:
            response = await self._get_llm().ainvoke(
                [*call.messages, call.assistant_message, tool_message]
            )
        except Exception as exc:
            raise ToolCallingError(f"provider final response failed: {exc}") from exc
        return _content_to_text(response.content)

    def _get_llm(self) -> ChatOpenAI:
        if self._llm is None:
            self._llm = ChatOpenAI(
                api_key=self.settings.llm_api_key,
                base_url=self.settings.llm_base_url,
                model=self.settings.llm_model,
                temperature=0.1,
            )
        return self._llm

    @staticmethod
    def _messages(query: str, history: list[dict], user_id: str) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "你是企业业务助手。仅当用户明确请求年假、报销状态或会议室时调用工具；"
                    "工具参数必须来自用户问题或上下文。不要为普通知识问题调用工具。"
                ),
            }
        ]
        messages.extend(history[-6:])
        messages.append({"role": "user", "content": f"用户ID={user_id}\n{query}"})
        return messages


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return str(content)
