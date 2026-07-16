from functools import lru_cache

from fastapi import Header, HTTPException

from app.config import get_settings
from app.core.agent import EnterpriseRAGAgent, get_agent
from app.core.embedding import get_embedding_service
from app.core.vector_store import get_vector_store
from app.models.database import AsyncSessionLocal
from app.services.chat_service import ChatService
from app.services.document_service import DocumentService


async def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = get_settings().app_api_key
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="invalid API key")


@lru_cache
def get_chat_service() -> ChatService:
    return ChatService(AsyncSessionLocal, get_agent())


@lru_cache
def get_document_service() -> DocumentService:
    return DocumentService(AsyncSessionLocal, get_embedding_service(), get_vector_store())


def get_rag_agent() -> EnterpriseRAGAgent:
    return get_agent()
