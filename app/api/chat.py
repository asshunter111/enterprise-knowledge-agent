import json
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.dependencies import get_chat_service, verify_api_key
from app.schemas import ChatRequest, ChatResponse, FeedbackRequest, SessionCreate, SessionOut
from app.services.chat_service import ChatService

router = APIRouter(prefix="/api", tags=["chat"], dependencies=[Depends(verify_api_key)])
ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]


@router.post("/sessions", response_model=SessionOut)
async def create_session(body: SessionCreate, service: ChatServiceDep):
    return await service.create_session(body.title)


@router.get("/sessions", response_model=list[SessionOut])
async def list_sessions(service: ChatServiceDep):
    return await service.list_sessions()


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str, service: ChatServiceDep):
    if not await service.delete_session(session_id):
        raise HTTPException(status_code=404, detail="session not found")


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, service: ChatServiceDep):
    try:
        return await service.answer(body.session_id, body.query)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/chat/stream")
async def chat_stream(body: ChatRequest, service: ChatServiceDep) -> StreamingResponse:
    try:
        chat_session, history = await service.begin_exchange(body.session_id, body.query)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    prepared = await service.agent.prepare_stream(body.query, history)

    async def event_stream() -> AsyncIterator[str]:
        yield _sse("meta", {"session_id": chat_session.id, "trace": prepared["trace"]})
        chunks: list[str] = []
        try:
            async for token in service.agent.generator.stream(
                body.query, prepared["documents"], history
            ):
                chunks.append(token)
                yield _sse("token", {"text": token})
        except Exception as exc:
            yield _sse("error", {"message": str(exc)})
            return

        answer = "".join(chunks)
        message = await service.save_assistant_message(
            chat_session.id, answer, prepared["citations"], prepared["trace"]
        )
        yield _sse("citations", {"items": prepared["citations"]})
        yield _sse("done", {"message_id": message.id})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/messages/{message_id}/feedback")
async def feedback(
    message_id: str,
    body: FeedbackRequest,
    service: ChatServiceDep,
):
    if not await service.set_feedback(message_id, body.feedback):
        raise HTTPException(status_code=404, detail="assistant message not found")
    return {"status": "ok"}


def _sse(event: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"
