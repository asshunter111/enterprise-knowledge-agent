from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.tools import ToolExecutionError, build_default_tool_registry
from app.dependencies import verify_api_key

router = APIRouter(prefix="/api/tools", tags=["tools"], dependencies=[Depends(verify_api_key)])
_registry = build_default_tool_registry()


class ToolRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    arguments: dict[str, Any] = Field(default_factory=dict)


@router.get("")
async def list_tools() -> dict[str, list[str]]:
    return {"tools": _registry.names()}


@router.post("/execute")
async def execute_tool(body: ToolRequest) -> dict[str, Any]:
    try:
        return {"tool": body.name, "result": await _registry.execute(body.name, **body.arguments)}
    except ToolExecutionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
