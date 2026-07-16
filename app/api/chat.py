"""Conversation & QA APIs"""
import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import get_db
from app.models.session import ChatSession, ChatMessage
from app.schemas import (
    SessionCreate, SessionOut, ChatRequest, ChatResponse, Citation,
)
from app.core.retriever import retrieve_with_rerank
from app.core.generator import generate_answer

router = APIRouter(tags=["智能问答"])


# ── sessions ──

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
        raise HTTPException(404, "session not found")
    await db.delete(s)
    await db.commit()
    return {"message": "deleted"}


# ── helpers ──

async def _get_session(sid: str | None, db: AsyncSession) -> ChatSession:
    if sid:
        result = await db.execute(select(ChatSession).where(ChatSession.id == sid))
        s = result.scalar_one_or_none()
        if s:
            return s
    s = ChatSession(title="New Chat")
    db.add(s)
    await db.commit()
    await db.refresh(s)
    return s


async def _load_history(sid: str, db: AsyncSession) -> list[dict]:
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == sid)
        .order_by(ChatMessage.created_at.asc())
    )
    return [
        {"role": m.role, "content": m.content}
        for m in result.scalars().all()[-10:]
    ]


# ── chat ──

@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, db: AsyncSession = Depends(get_db)):
    session = await _get_session(body.session_id, db)
    history = await _load_history(session.id, db)

    docs = await retrieve_with_rerank(body.query)
    result = await generate_answer(query=body.query, retrieved_docs=docs, history=history)

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
        steps=[f"retrieved {len(docs)} chunks"],
        retrieved_count=len(docs),
    )


@router.post("/chat/stream")
async def chat_stream(body: ChatRequest, db: AsyncSession = Depends(get_db)):
    session = await _get_session(body.session_id, db)
    history = await _load_history(session.id, db)

    db.add(ChatMessage(session_id=session.id, role="user", content=body.query))
    await db.commit()

    docs = await retrieve_with_rerank(body.query)
    result = await generate_answer(query=body.query, retrieved_docs=docs, history=history)

    db.add(ChatMessage(
        session_id=session.id,
        role="assistant",
        content=result["answer"],
        citations=json.dumps(result["citations"], ensure_ascii=False),
    ))
    await db.commit()

    async def stream():
        for word in result["answer"]:
            yield word
        yield "\n\n__CITATIONS__:" + json.dumps(result["citations"], ensure_ascii=False)

    return StreamingResponse(stream(), media_type="text/plain")


@router.post("/feedback/{message_id}")
async def feedback(message_id: str, fb: str = "like", db: AsyncSession = Depends(get_db)):
    if fb not in ("like", "dislike"):
        raise HTTPException(400, "must be like or dislike")
    result = await db.execute(select(ChatMessage).where(ChatMessage.id == message_id))
    msg = result.scalar_one_or_none()
    if not msg:
        raise HTTPException(404, "message not found")
    msg.feedback = fb
    await db.commit()
    return {"message": "ok"}
