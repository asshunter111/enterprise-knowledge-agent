from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import Settings
from app.core.agent import EnterpriseRAGAgent
from app.dependencies import get_chat_service, get_document_service, get_rag_agent
from app.main import app
from app.models.database import AsyncSessionLocal, init_db
from app.models.document import Document
from app.services.chat_service import ChatService
from tests.fakes import FakeGenerator, FakeRetriever


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


@pytest.fixture
async def client():
    await init_db()
    agent = EnterpriseRAGAgent(Settings(min_relevance_score=0.05), FakeRetriever(), FakeGenerator())
    chat_service = ChatService(AsyncSessionLocal, agent)
    app.dependency_overrides[get_chat_service] = lambda: chat_service
    app.dependency_overrides[get_rag_agent] = lambda: agent
    app.dependency_overrides[get_document_service] = lambda: FakeDocumentService()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_chat_citations_and_feedback(client: AsyncClient):
    response = await client.post("/api/chat", json={"query": "报销期限是多少"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["retrieved_count"] == 1
    assert payload["citations"][0]["document_id"] == "doc-1"
    assert "rerank" in " ".join(payload["trace"])

    feedback = await client.post(
        f"/api/messages/{payload['message_id']}/feedback", json={"feedback": "like"}
    )
    assert feedback.status_code == 200


@pytest.mark.asyncio
async def test_stream_endpoint_returns_real_sse_events(client: AsyncClient):
    async with client.stream(
        "POST", "/api/chat/stream", json={"query": "报销期限是多少"}
    ) as response:
        body = "".join([chunk async for chunk in response.aiter_text()])

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: token" in body
    assert "event: citations" in body
    assert "event: done" in body


@pytest.mark.asyncio
async def test_document_upload_is_processed_in_background(client: AsyncClient):
    upload = await client.post(
        "/api/documents/upload",
        files={"file": ("policy.md", "报销制度".encode(), "text/markdown")},
    )
    assert upload.status_code == 202
    document_id = upload.json()["id"]

    detail = await client.get(f"/api/documents/{document_id}")
    assert detail.status_code == 200
    assert detail.json()["status"] == "ready"
    assert detail.json()["chunk_count"] == 1


@pytest.mark.asyncio
async def test_rejects_unsupported_file_type(client: AsyncClient):
    response = await client.post(
        "/api/documents/upload",
        files={"file": ("legacy.doc", b"content", "application/msword")},
    )
    assert response.status_code == 415
