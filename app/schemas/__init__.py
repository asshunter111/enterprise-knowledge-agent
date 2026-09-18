from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    file_type: str
    file_size: int
    chunk_count: int
    status: str
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class SessionCreate(BaseModel):
    title: str = Field(default="新对话", min_length=1, max_length=255)


class SessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    created_at: datetime
    updated_at: datetime


class Citation(BaseModel):
    index: int
    document_id: str
    document_name: str
    chunk_index: int
    content_preview: str
    score: float


class ChatRequest(BaseModel):
    session_id: str | None = None
    user_id: str = Field(default="anonymous", min_length=1, max_length=100)
    query: str = Field(min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    session_id: str
    message_id: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)
    retrieved_count: int = 0
    tool: str | None = None
    memory_used: bool = False
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class FeedbackRequest(BaseModel):
    feedback: str = Field(pattern="^(like|dislike)$")


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class RetrievalResult(BaseModel):
    content: str
    score: float
    metadata: dict[str, Any]


class MemoryWrite(BaseModel):
    user_id: str = Field(min_length=1, max_length=100)
    key: str = Field(min_length=1, max_length=100)
    value: str = Field(min_length=1, max_length=2000)
    memory_type: str = Field(default="preference", min_length=1, max_length=50)


class MemoryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: str
    key: str
    value: str
    memory_type: str
    created_at: datetime
    updated_at: datetime
