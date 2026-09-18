from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any


class ToolExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    handler: Callable[..., Awaitable[dict[str, Any]]]


class BusinessToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(
        self,
        name: str,
        description: str,
        handler: Callable[..., Awaitable[dict[str, Any]]],
    ) -> None:
        if not name.strip():
            raise ValueError("tool name cannot be empty")
        self._tools[name] = ToolDefinition(name, description, handler)

    def names(self) -> list[str]:
        return sorted(self._tools)

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
    registry.register("get_leave_balance", "查询员工剩余年假", get_leave_balance)
    registry.register("get_reimbursement_status", "查询报销单状态", get_reimbursement_status)
    registry.register("query_meeting_rooms", "查询会议室空闲情况", query_meeting_rooms)
    return registry
