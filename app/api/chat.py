"""会话 & 问答 API"""
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


# ── 会话 ──

@router.post("/sessions", response_model=SessionOut)
async def create_session(body: SessionCreate, db: AsyncSession = Depends(get_db)):
    s = ChatSession(title=body.title)
    db.add(s)
    await db.commit()
    await db.refresh(s)
    return SessionOut(
        id=s.id, title=s.title,
        created_at=str(s.created_at), updated_at=str(s.updated_at),
    )


@router.get("/sessions", response_model=list[SessionOut])
async def list_sessions(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(ChatSession).order_by(ChatSession.updated_at.desc())
    )
    return [
        SessionOut(
            id=s.id, title=s.title,
            created_at=str(s.created_at), updated_at=str(s.updated_at),
        )
        for s in result.scalars().all()
    ]


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(ChatSession).where(ChatSession.id == session_id))
    s = result.scalar_one_or_none()
    if not s:
        raise HTTPException(404, "会话不存在")
    await db.delete(s)
    await db.commit()
    return {"message": "已删除"}


# ── 问答 ──

async def _get_session(session_id: str | None, db: AsyncSession) -> ChatSession:
    if session_id:
        result = await db.execute(
            select(ChatSession).where(ChatSession.id == session_id)
        )
        s = result.scalar_one_or_none()
        if s:
            return s
    s = ChatSession(title="新对话")
    db.add(s)
    await db.commit()
    await db.refresh(s)
    return s


async def _build_history(session_id: str, db: AsyncSession) -> list[dict]:
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())
    )
    return [
        {"role": m.role, "content": m.content}
        for m in result.scalars().all()[-10:]
    ]


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, db: AsyncSession = Depends(get_db)):
    session = await _get_session(body.session_id, db)
    history = await _build_history(session.id, db)

    result = await run_agent(query=body.query, history=history)

    # 保存消息
    db.add(ChatMessage(session_id=session.id, role="user", content=body.query))
    db.add(ChatMessage(
        session_id=session.id,
        role="assistant",
        content=result["answer"],
        citations=json.dumps(result["citations"], ensure_ascii=False),
    ))
    await db.commit()

    return ChatResponse(
        session_id=session.id,
        answer=result["answer"],
        citations=[
            Citation(
                index=c["index"], doc_name=c["doc_name"],
                content_preview=c["content_preview"], score=c["score"],
            )
            for c in result["citations"]
        ],
        steps=result["steps"],
        retrieved_count=result["retrieved_count"],
    )


@router.post("/chat/stream")
async def chat_stream(body: ChatRequest, db: AsyncSession = Depends(get_db)):
    session = await _get_session(body.session_id, db)
    history = await _build_history(session.id, db)

    db.add(ChatMessage(session_id=session.id, role="user", content=body.query))
    await db.commit()

    result = await run_agent(query=body.query, history=history)

    db.add(ChatMessage(
        session_id=session.id,
        role="assistant",
        content=result["answer"],
        citations=json.dumps(result["citations"], ensure_ascii=False),
    ))
    await db.commit()

    # 模拟流式逐词输出
    async def stream():
        for word in result["answer"]:
            yield word
        yield "\n\n__CITATIONS__: " + json.dumps(result["citations"], ensure_ascii=False)

    return StreamingResponse(stream(), media_type="text/plain")


@router.post("/feedback/{message_id}")
async def feedback(message_id: str, fb: str = "like", db: AsyncSession = Depends(get_db)):
    if fb not in ("like", "dislike"):
        raise HTTPException(400, "feedback 只能是 like 或 dislike")

    result = await db.execute(select(ChatMessage).where(ChatMessage.id == message_id))
    msg = result.scalar_one_or_none()
    if not msg:
        raise HTTPException(404, "消息不存在")

    msg.feedback = fb
    await db.commit()
    return {"message": "反馈已记录"}
