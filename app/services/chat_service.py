from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.agent import EnterpriseRAGAgent
from app.models.session import ChatMessage, ChatSession


class ChatService:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        agent: EnterpriseRAGAgent,
    ) -> None:
        self.session_factory = session_factory
        self.agent = agent

    async def create_session(self, title: str) -> ChatSession:
        async with self.session_factory() as session:
            chat_session = ChatSession(title=title)
            session.add(chat_session)
            await session.commit()
            await session.refresh(chat_session)
            return chat_session

    async def list_sessions(self) -> list[ChatSession]:
        async with self.session_factory() as session:
            statement = select(ChatSession).order_by(ChatSession.updated_at.desc())
            return list((await session.scalars(statement)).all())

    async def delete_session(self, session_id: str) -> bool:
        async with self.session_factory() as session:
            chat_session = await session.get(ChatSession, session_id)
            if chat_session is None:
                return False
            await session.delete(chat_session)
            await session.commit()
            return True

    async def answer(self, session_id: str | None, query: str) -> dict:
        chat_session, history = await self.begin_exchange(session_id, query)
        result = await self.agent.run(query, history)
        message = await self.save_assistant_message(
            chat_session.id, result["answer"], result["citations"], result["trace"]
        )
        return {"session_id": chat_session.id, "message_id": message.id, **result}

    async def begin_exchange(
        self, session_id: str | None, query: str
    ) -> tuple[ChatSession, list[dict]]:
        async with self.session_factory() as session:
            chat_session = await self._get_or_create_session(session, session_id)
            history = await self._load_history(session, chat_session.id)
            session.add(ChatMessage(session_id=chat_session.id, role="user", content=query))
            if chat_session.title == "新对话":
                chat_session.title = query[:40]
            chat_session.updated_at = datetime.now(UTC)
            await session.commit()
            return chat_session, history

    async def save_assistant_message(
        self,
        session_id: str,
        answer: str,
        citations: list[dict],
        trace: list[str],
    ) -> ChatMessage:
        async with self.session_factory() as session:
            message = ChatMessage(
                id=str(uuid4()),
                session_id=session_id,
                role="assistant",
                content=answer,
                citations=citations,
                trace=trace,
            )
            session.add(message)
            chat_session = await session.get(ChatSession, session_id)
            if chat_session is not None:
                chat_session.updated_at = datetime.now(UTC)
            await session.commit()
            await session.refresh(message)
            return message

    async def set_feedback(self, message_id: str, feedback: str) -> bool:
        async with self.session_factory() as session:
            message = await session.get(ChatMessage, message_id)
            if message is None or message.role != "assistant":
                return False
            message.feedback = feedback
            await session.commit()
            return True

    async def _get_or_create_session(
        self, session: AsyncSession, session_id: str | None
    ) -> ChatSession:
        if session_id:
            chat_session = await session.get(ChatSession, session_id)
            if chat_session is None:
                raise LookupError("session not found")
            return chat_session
        chat_session = ChatSession()
        session.add(chat_session)
        await session.flush()
        return chat_session

    @staticmethod
    async def _load_history(session: AsyncSession, session_id: str, limit: int = 10) -> list[dict]:
        statement = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.desc())
            .limit(limit)
        )
        messages = list((await session.scalars(statement)).all())
        messages.reverse()
        return [{"role": item.role, "content": item.content} for item in messages]
