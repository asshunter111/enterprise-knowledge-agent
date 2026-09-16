from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from app.config import Settings
from app.core.agent import EnterpriseRAGAgent
from app.dependencies import get_chat_service, get_document_service, get_rag_agent
from app.main import app
from app.models.database import AsyncSessionLocal, init_db
from app.models.document import Document
from app.models.session import ChatMessage, ChatSession
from app.services.chat_service import ChatService
from tests.fakes import FakeGenerator, RecordingRetriever

B_GROUP_DOCUMENT = {
    "id": "eval-schedule:1",
    "content": "B组负责本周一至周五的下午班次。",
    "metadata": {
        "document_id": "eval-schedule",
        "document_name": "schedule.md",
        "chunk_index": 1,
    },
    "score": 0.71,
}

RESOLUTIONS = {
    "那么B组呢？": {
        "intent": "schedule_query",
        "slots": {"group": "B组", "period": "本周"},
        "retrieval_query": "B组本周的排班情况怎么样？",
    },
    "报销期限是多少？": {
        "intent": "finance_policy",
        "slots": {"policy_type": "expense_reimbursement"},
        "retrieval_query": "员工报销需要在费用发生后多久提交？",
    },
}


class ByQueryResolver:
    """按 Query 返回预设解析结果。"""

    def __init__(self, mapping: dict[str, dict]) -> None:
        self.mapping = mapping

    async def resolve(self, query: str, history: list[dict], active_context: dict | None) -> dict:
        return dict(self.mapping[query])


class FakeDocumentService:
    async def process(self, document_id: str, path: Path, filename: str) -> None:
        async with AsyncSessionLocal() as session:
            document = await session.get(Document, document_id)
            document.status = "ready"
            document.chunk_count = 1
            await session.commit()

    async def delete(self, document: Document, path: Path) -> None:
        if path.exists():
            path.unlink()
        async with AsyncSessionLocal() as session:
            stored = await session.get(Document, document.id)
            await session.delete(stored)
            await session.commit()


async def stored_context(session_id: str) -> dict | None:
    async with AsyncSessionLocal() as session:
        chat_session = await session.get(ChatSession, session_id)
        return chat_session.active_context


@pytest.fixture
async def client():
    await init_db()
    retriever = RecordingRetriever(
        {
            "那么B组呢？": [B_GROUP_DOCUMENT],
            "B组本周的排班情况怎么样？": [B_GROUP_DOCUMENT],
            "报销期限是多少？": [B_GROUP_DOCUMENT],
            "员工报销需要在费用发生后多久提交？": [B_GROUP_DOCUMENT],
        }
    )
    agent = EnterpriseRAGAgent(
        Settings(min_relevance_score=0.05),
        retriever,
        FakeGenerator(),
        resolver=ByQueryResolver(RESOLUTIONS),
    )
    chat_service = ChatService(AsyncSessionLocal, agent)
    app.dependency_overrides[get_chat_service] = lambda: chat_service
    app.dependency_overrides[get_rag_agent] = lambda: agent
    app.dependency_overrides[get_document_service] = lambda: FakeDocumentService()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as test_client:
        test_client.retriever = retriever
        yield test_client
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_direct_response_goes_through_api_without_retrieval(client: AsyncClient):
    response = await client.post("/api/chat", json={"query": "你好"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "你好，有什么可以帮你？"
    assert payload["citations"] == []
    assert payload["retrieved_count"] == 0
    assert client.retriever.queries == []
    assert payload["trace"] == ["context_router: direct response", "direct_response: completed"]


@pytest.mark.asyncio
async def test_active_context_is_persisted_and_switched_on_new_intent(client: AsyncClient):
    first = await client.post("/api/chat", json={"query": "你好"})
    session_id = first.json()["session_id"]
    assert await stored_context(session_id) is None

    second = await client.post(
        "/api/chat", json={"session_id": session_id, "query": "那么B组呢？"}
    )
    assert second.status_code == 200
    assert "context_resolve: schedule_query" in second.json()["trace"]
    assert "B组本周的排班情况怎么样？" in " ".join(second.json()["trace"])
    assert await stored_context(session_id) == {
        "intent": "schedule_query",
        "slots": {"group": "B组", "period": "本周"},
    }

    third = await client.post(
        "/api/chat", json={"session_id": session_id, "query": "报销期限是多少？"}
    )
    assert third.status_code == 200
    assert third.json()["trace"][0] == "context_router: new_intent"
    assert await stored_context(session_id) == {
        "intent": "finance_policy",
        "slots": {"policy_type": "expense_reimbursement"},
    }

    # 新意图只切断 active_context，完整聊天记录仍然保留
    async with AsyncSessionLocal() as session:
        messages = list(
            (
                await session.scalars(
                    select(ChatMessage).where(ChatMessage.session_id == session_id)
                )
            ).all()
        )
        assert len(messages) == 6
