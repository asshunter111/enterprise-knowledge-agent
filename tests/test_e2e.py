import pytest

from app.config import Settings
from app.core.agent import EnterpriseRAGAgent
from app.core.embedding import EmbeddingService
from app.core.generator import AnswerGenerator
from app.core.reranker import Reranker
from app.core.retriever import Retriever
from app.core.vector_store import VectorStore
from app.models.database import AsyncSessionLocal, init_db
from app.services.chat_service import ChatService
from app.services.memory_service import MemoryService
from tests.fakes import FakeGenerator, FakeRetriever


class FinanceRetriever(FakeRetriever):
    async def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        return [
            {
                "id": "finance:0",
                "content": "财务部报销流程需要提交申请。",
                "metadata": {
                    "document_id": "finance",
                    "document_name": "finance.md",
                    "chunk_index": 0,
                    "department": "finance",
                },
                "score": 0.9,
            }
        ]


@pytest.mark.asyncio
async def test_agent_tool_call_is_real_registry_execution():
    agent = EnterpriseRAGAgent(Settings(), FakeRetriever(), FakeGenerator())
    result = await agent.run("我还有多少天年假？", user_id="employee-1")
    assert result["tool"] == "get_leave_balance"
    assert result["answer"] == "员工 employee-1 还剩 12 天年假。"
    assert "tool: executed get_leave_balance" in result["trace"]


@pytest.mark.asyncio
async def test_real_hash_rag_e2e_returns_citation(tmp_path):
    settings = Settings(
        chroma_persist_dir=tmp_path / "chroma",
        chroma_collection="e2e_rag",
        embedding_backend="hash",
        rerank_backend="lexical",
        min_relevance_score=0.05,
    )
    embedding = EmbeddingService(settings)
    store = VectorStore(settings)
    chunks = [{"content": "兰州是甘肃省的省会城市。", "metadata": {"chunk_index": 0, "chunk_total": 1}}]
    vectors = await embedding.embed_documents([chunks[0]["content"]])
    await store.upsert(chunks, vectors, "company", "company.md")
    retriever = Retriever(settings, embedding, store, Reranker(settings))
    agent = EnterpriseRAGAgent(settings, retriever, AnswerGenerator(settings))

    result = await agent.run("兰州是哪个省的省会？")

    assert "甘肃省" in result["answer"]
    assert result["citations"][0]["document_name"] == "company.md"
    assert "generate: answer completed" in result["trace"]


@pytest.mark.asyncio
async def test_tool_permission_denial_is_inside_agent_path():
    agent = EnterpriseRAGAgent(Settings(), FakeRetriever(), FakeGenerator())
    result = await agent.run("我的报销申请到哪一步了？", user_id="employee-1")
    assert result["tool"] == "get_reimbursement_status"
    assert result["answer"] == "当前角色无权查询报销单状态。"
    assert "tool: denied get_reimbursement_status" in result["trace"]


@pytest.mark.asyncio
async def test_memory_cross_session_and_user_isolation():
    await init_db()
    memory = MemoryService(AsyncSessionLocal, Settings(memory_retrieval_limit=5))
    agent = EnterpriseRAGAgent(Settings(), FakeRetriever(), FakeGenerator(), memory_service=memory)
    service = ChatService(AsyncSessionLocal, agent, memory)

    await service.answer(None, "我是财务部员工。", user_id="memory-user-a")
    result = await service.answer(None, "和我相关的报销规定有哪些？", user_id="memory-user-a")
    other = await service.answer(None, "和我相关的报销规定有哪些？", user_id="memory-user-b")

    assert result["memory_used"] is True
    assert other["memory_used"] is False
    assert await memory.delete("memory-user-a", "department")
    assert await memory.retrieve("memory-user-a", "报销") == []


@pytest.mark.asyncio
async def test_agent_filters_restricted_documents_before_generation():
    agent = EnterpriseRAGAgent(Settings(), FinanceRetriever(), FakeGenerator())
    employee = await agent.run("报销流程是什么？", user_id="employee-2", role="employee")
    finance = await agent.run("报销流程是什么？", user_id="finance-1", role="finance")
    assert employee["answer"] == "资料中未找到相关信息。"
    assert employee["citations"] == []
    assert finance["retrieved_count"] == 1
    assert employee["diagnostics"]["permission_allowed_count"] == 0
