import pytest

from app.config import Settings
from app.core.agent import EnterpriseRAGAgent
from tests.fakes import FakeGenerator, FakeRetriever


@pytest.mark.asyncio
async def test_agent_runs_complete_langgraph_flow():
    agent = EnterpriseRAGAgent(Settings(min_relevance_score=0.05), FakeRetriever(), FakeGenerator())

    result = await agent.run("报销期限是多少")

    assert result["answer"].startswith("报销应在十个工作日内")
    assert result["retrieved_count"] == 1
    assert result["citations"][0]["document_name"] == "财务制度.md"
    assert result["trace"] == [
        "retrieve: 1 candidates",
        "verify: 1 candidates passed threshold",
        "rerank: selected 1 chunks",
        "generate: answer completed",
    ]


@pytest.mark.asyncio
async def test_agent_uses_no_context_branch():
    agent = EnterpriseRAGAgent(Settings(min_relevance_score=0.05), FakeRetriever(), FakeGenerator())

    result = await agent.run("missing information")

    assert result["answer"] == "资料中未找到相关信息。"
    assert result["citations"] == []
    assert result["trace"][-1] == "no_context: no reliable evidence"
