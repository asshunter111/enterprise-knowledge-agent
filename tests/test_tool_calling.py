import json
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from app.config import Settings
from app.core.agent import EnterpriseRAGAgent
from app.core.mcp_client import MCPStdioClient, MCPTool
from app.core.tools import (
    NativeToolCall,
    ProviderToolCaller,
    ToolCallingError,
    ToolExecutionError,
    build_default_tool_registry,
)
from tests.fakes import FakeGenerator, FakeRetriever


class FakeProviderToolCaller:
    def __init__(self, tool_name: str, arguments: dict):
        self.tool_name = tool_name
        self.arguments = arguments
        self.proposals = []
        self.results = []

    async def propose(self, query: str, history: list[dict], user_id: str):
        self.proposals.append((query, history, user_id))
        message = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": self.tool_name,
                    "args": self.arguments,
                    "id": "call-test-1",
                    "type": "tool_call",
                }
            ],
        )
        return NativeToolCall(
            self.tool_name,
            self.arguments,
            "call-test-1",
            message,
            [{"role": "user", "content": query}],
        )

    async def complete(self, call: NativeToolCall, result: dict) -> str:
        self.results.append((call, result))
        if "error" in result:
            return f"工具执行失败：{result['error']}"
        return f"工具结果：{json.dumps(result, ensure_ascii=False)}"


class FakeBoundChatModel:
    def __init__(self):
        self.bound_tools = []
        self.bind_options = {}
        self.messages = []

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = tools
        self.bind_options = kwargs
        return self

    async def ainvoke(self, messages):
        self.messages.append(messages)
        if isinstance(messages[-1], ToolMessage):
            return AIMessage(content="工具结果已处理")
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "query_meeting_rooms",
                    "args": {"date": "明天", "time": "下午"},
                    "id": "call-native-1",
                    "type": "tool_call",
                }
            ],
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "expected_tool"),
    [
        ("我还有多少年假？", "get_leave_balance"),
        ("报销单 R-100 当前到哪一步了？", "get_reimbursement_status"),
        ("明天下午有没有空闲会议室？", "query_meeting_rooms"),
    ],
)
async def test_deterministic_fallback_identifies_business_tools(query, expected_tool):
    agent = EnterpriseRAGAgent(Settings(), FakeRetriever(), FakeGenerator())

    result = await agent.run(query, user_id="employee-1", role="finance")

    assert result["tool"] == expected_tool
    assert result["tool_source"] == (
        "mcp" if expected_tool == "query_meeting_rooms" else "local"
    )
    assert "native" not in " ".join(result["trace"])


@pytest.mark.asyncio
async def test_provider_tool_call_uses_structured_arguments_and_final_response():
    caller = FakeProviderToolCaller("get_leave_balance", {"employee_id": "employee-9"})
    agent = EnterpriseRAGAgent(
        Settings(llm_api_key="test-key"),
        FakeRetriever(),
        FakeGenerator(),
        tool_caller=caller,
    )

    result = await agent.run("请查一下我的年假余额", user_id="employee-9")

    assert result["tool"] == "get_leave_balance"
    assert caller.results[0][1] == {
        "employee_id": "employee-9",
        "annual_leave_days": 12,
        "source": "demo_hr_system",
    }
    assert "工具结果" in result["answer"]
    assert "tool: native response completed" in result["trace"]


@pytest.mark.asyncio
async def test_provider_tool_call_execution_failure_is_returned_without_500():
    caller = FakeProviderToolCaller("get_leave_balance", {"employee_id": ""})
    agent = EnterpriseRAGAgent(
        Settings(llm_api_key="test-key"),
        FakeRetriever(),
        FakeGenerator(),
        tool_caller=caller,
    )

    result = await agent.run("查询年假", user_id="employee-1")

    assert result["answer"] == "工具执行失败：当前无法完成该业务查询。"
    assert "tool: failed get_leave_balance" in result["trace"]


@pytest.mark.asyncio
async def test_provider_tool_caller_uses_bind_tools_and_tool_message_round_trip():
    model = FakeBoundChatModel()
    caller = ProviderToolCaller(Settings(llm_api_key="test-key"), build_default_tool_registry())
    caller._llm = model

    call = await caller.propose("明天下午有没有空闲会议室？", [], "employee-1")
    answer = await caller.complete(call, {"date": "明天", "time": "下午", "rooms": ["Room-A"]})

    assert call.name == "query_meeting_rooms"
    assert call.arguments == {"date": "明天", "time": "下午"}
    assert {tool.name for tool in model.bound_tools} == {
        "get_leave_balance",
        "get_reimbursement_status",
        "query_meeting_rooms",
    }
    assert model.bind_options == {"strict": True, "parallel_tool_calls": False}
    assert isinstance(model.messages[-1][-1], ToolMessage)
    assert answer == "工具结果已处理"


@pytest.mark.asyncio
async def test_invalid_provider_arguments_raise_controlled_tool_calling_error():
    class InvalidArgumentsModel(FakeBoundChatModel):
        async def ainvoke(self, messages):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_leave_balance",
                        "args": {},
                        "id": "call-invalid-1",
                        "type": "tool_call",
                    }
                ],
            )

    caller = ProviderToolCaller(Settings(llm_api_key="test-key"), build_default_tool_registry())
    caller._llm = InvalidArgumentsModel()

    with pytest.raises(ToolCallingError, match="invalid arguments"):
        await caller.propose("查询年假", [], "employee-1")


@pytest.mark.asyncio
async def test_no_api_key_keeps_deterministic_tool_fallback():
    agent = EnterpriseRAGAgent(Settings(), FakeRetriever(), FakeGenerator())

    result = await agent.run("报销单 R-7 当前状态是什么？", role="finance")

    assert result["tool"] == "get_reimbursement_status"
    assert result["answer"] == "报销单 R-7 当前状态是：processing。"
    assert result["trace"][0] == "context_router: tool (get_reimbursement_status)"


def test_default_tools_expose_standard_structured_schemas():
    tools = {tool.name: tool for tool in build_default_tool_registry().langchain_tools()}

    assert set(tools) == {
        "get_leave_balance",
        "get_reimbursement_status",
        "query_meeting_rooms",
    }
    assert tools["get_leave_balance"].args_schema.model_validate(
        {"employee_id": "employee-1"}
    ).employee_id == "employee-1"
    assert tools["query_meeting_rooms"].args_schema.model_validate(
        {"date": "明天", "time": "下午"}
    ).time == "下午"


@pytest.mark.asyncio
async def test_tool_registry_execution_failure_is_controlled():
    with pytest.raises(ToolExecutionError, match="invalid arguments"):
        await build_default_tool_registry().execute("query_meeting_rooms", date="明天")


@pytest.mark.asyncio
async def test_provider_no_tool_call_preserves_normal_rag_path():
    class NoToolProvider(FakeProviderToolCaller):
        async def propose(self, query: str, history: list[dict], user_id: str):
            return None

    agent = EnterpriseRAGAgent(
        Settings(llm_api_key="test-key", min_relevance_score=0.05),
        FakeRetriever(),
        FakeGenerator(),
        tool_caller=NoToolProvider("get_leave_balance", {}),
    )

    result = await agent.run("报销期限是多少")

    assert result["tool"] is None
    assert result["retrieved_count"] == 1
    assert "rerank: selected 1 chunks" in result["trace"]


@pytest.mark.asyncio
async def test_mcp_unavailable_returns_error_without_fabricating_rooms():
    def unavailable_client():
        return MCPStdioClient(
            command=sys.executable,
            args=["-c", "raise SystemExit(3)"],
            timeout_seconds=1.0,
        )

    agent = EnterpriseRAGAgent(
        Settings(),
        FakeRetriever(),
        FakeGenerator(),
        mcp_client_factory=unavailable_client,
    )

    result = await agent.run("明天下午有没有空闲会议室？")

    assert result["tool"] == "query_meeting_rooms"
    assert result["tool_source"] == "mcp"
    assert result["answer"] == "MCP 业务工具当前不可用，请稍后重试。"
    assert "Room-A" not in result["answer"]


@pytest.mark.asyncio
async def test_agent_calls_mcp_with_plural_meeting_room_tool_name():
    class RecordingMCPClient:
        def __init__(self):
            self.calls = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_value, traceback):
            return None

        async def list_tools(self):
            return [MCPTool("query_meeting_rooms", "", {"type": "object"})]

        async def call_tool(self, name, arguments):
            self.calls.append((name, arguments))
            return {
                "date": arguments["date"],
                "time": arguments["time"],
                "rooms": ["Room-A"],
            }

    client = RecordingMCPClient()
    agent = EnterpriseRAGAgent(
        Settings(),
        FakeRetriever(),
        FakeGenerator(),
        mcp_client_factory=lambda: client,
    )

    result = await agent.run("明天下午有没有空闲会议室？")

    assert client.calls == [("query_meeting_rooms", {"date": "明天", "time": "下午"})]
    assert result["answer"] == "明天 下午 可用会议室：Room-A。"


def test_tool_calling_evaluation_cases_are_separate_from_retrieval_dataset():
    path = Path(__file__).parents[1] / "evaluation" / "tool_calling.json"
    cases = json.loads(path.read_text(encoding="utf-8"))

    assert {case["category"] for case in cases} == {
        "leave",
        "reimbursement",
        "meeting_room",
        "knowledge",
        "execution_failure",
        "ambiguous",
    }
    assert len(cases) == 7
