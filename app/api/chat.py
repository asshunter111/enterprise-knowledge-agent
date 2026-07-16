"""聊天 API — 会话管理 + 知识问答 + 流式"""
import json
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import get_db
from app.models.session import ChatSession, ChatMessage
from app.schemas import (
    SessionCreate, SessionOut, ChatRequest, ChatResponse, Citation,
)
from app.core.agent import run_agent

router = APIRouter(tags=["智能问答"])


# ── 会话管理 ──

@router.post("/sessions", response_model=SessionOut)
async def create_session(
    body: SessionCreate, db: AsyncSession = Depends(get_db),
):
    session = ChatSession(title=body.title)
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return SessionOut(
        id=session.id, title=session.title,
        created_at=str(session.created_at), updated_at=str(session.updated_at),
    )


@router.get("/sessions", response_model=list[SessionOut])
async def list_sessions(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(ChatSession).order_by(ChatSession.updated_at.desc())
    )
    sessions = result.scalars().all()
    return [
        SessionOut(
            id=s.id, title=s.title,
            created_at=str(s.created_at), updated_at=str(s.updated_at),
        )
        for s in sessions
    ]


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(ChatSession).where(ChatSession.id == session_id)
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(404, "会话不存在")
    await db.delete(session)
    await db.commit()
    return {"message": "删除成功"}


# ── 问答 ──

@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, db: AsyncSession = Depends(get_db)):
    """知识问答（非流式）"""
    # 获取或创建会话
    session = await _get_or_create_session(body.session_id, db)

    # 取历史消息
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session.id)
        .order_by(ChatMessage.created_at.asc())
    )
    messages = result.scalars().all()
    history = [
        {"role": m.role, "content": m.content} for m in messages[-10:]
    ]

    # 运行 Agent
    result = await run_agent(query=body.query, history=history)

    # 保存消息
    user_msg = ChatMessage(session_id=session.id, role="user", content=body.query)
    db.add(user_msg)
    assistant_msg = ChatMessage(
        session_id=session.id,
        role="assistant",
        content=result["answer"],
        citations=json.dumps(result["citations"], ensure_ascii=False),
    )
    db.add(assistant_msg)
    await db.commit()

    return ChatResponse(
        session_id=session.id,
        answer=result["answer"],
        citations=[
            Citation(
                index=c["index"], doc_name=c["doc_name"],
                chunk_index=c["chunk_index"],
                content_preview=c["content_preview"], score=c["score"],
            )
            for c in result["citations"]
        ],
        steps=result["steps"],
        retrieved_count=result["retrieved_count"],
    )


@router.post("/chat/stream")
async def chat_stream(body: ChatRequest, db: AsyncSession = Depends(get_db)):
    """知识问答（SSE 流式）"""
    session = await _get_or_create_session(body.session_id, db)

    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session.id)
        .order_by(ChatMessage.created_at.asc())
    )
    messages = result.scalars().all()
    history = [{"role": m.role, "content": m.content} for m in messages[-10:]]

    # 保存用户消息
    user_msg = ChatMessage(session_id=session.id, role="user", content=body.query)
    db.add(user_msg)
    await db.commit()

    agent_result = await run_agent(query=body.query, history=history)
    full_answer = agent_result["answer"]

    # 保存助手消息
    assistant_msg = ChatMessage(
        session_id=session.id,
        role="assistant",
        content=full_answer,
        citations=json.dumps(agent_result["citations"], ensure_ascii=False),
    )
    db.add(assistant_msg)
    await db.commit()

    async def generate():
        # 模拟流式（逐词输出）
        for word in full_answer:
            yield word
        # 最后发 citations
        yield "\n\n__CITATIONS__:" + json.dumps(agent_result["citations"], ensure_ascii=False)

    return StreamingResponse(generate(), media_type="text/plain")


# ── 反馈 ──

@router.post("/chat/{message_id}/feedback")
async def submit_feedback(
    message_id: str,
    feedback: str,  # like or dislike
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(ChatMessage).where(ChatMessage.id == message_id)
    )
    msg = result.scalar_one_or_none()
    if not msg:
        raise HTTPException(404, "消息不存在")
    if feedback not in ("like", "dislike"):
        raise HTTPException(400, "feedback 必须是 like 或 dislike")
    msg.feedback = feedback
    await db.commit()
    return {"message": "反馈提交成功"}


async def _get_or_create_session(session_id: str | None, db: AsyncSession) -> ChatSession:
    if session_id:
        result = await db.execute(
            select(ChatSession).where(ChatSession.id == session_id)
        )
        session = result.scalar_one_or_none()
        if session:
            return session
    session = ChatSession(title="新对话")
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session
