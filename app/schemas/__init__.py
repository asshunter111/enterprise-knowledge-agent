"""Pydantic 请求/响应模型"""
from pydantic import BaseModel, Field


# ── 文档 ──
class DocumentOut(BaseModel):
    id: str
    filename: str
    file_type: str
    file_size: int
    chunk_count: int
    status: str
    created_at: str

    class Config:
        from_attributes = True


# ── 会话 ──
class SessionCreate(BaseModel):
    title: str = "新对话"


class SessionOut(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


# ── 聊天 ──
class Citation(BaseModel):
    index: int
    doc_name: str
    chunk_index: int
    content_preview: str
    score: float


class ChatRequest(BaseModel):
    session_id: str | None = None
    query: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    citations: list[Citation] = []
    steps: list[str] = []
    retrieved_count: int = 0


# ── 反馈 ──
class FeedbackRequest(BaseModel):
    message_id: str
    feedback: str = Field(..., pattern="^(like|dislike)$")
