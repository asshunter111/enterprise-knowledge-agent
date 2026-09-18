from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import Settings
from app.models.memory import UserMemory


class MemoryService:
    def __init__(
        self, session_factory: async_sessionmaker[AsyncSession], settings: Settings
    ) -> None:
        self.session_factory = session_factory
        self.settings = settings

    async def upsert(
        self, user_id: str, key: str, value: str, memory_type: str = "preference"
    ) -> UserMemory:
        async with self.session_factory() as session:
            result = await session.scalar(
                select(UserMemory).where(
                    UserMemory.user_id == user_id, UserMemory.key == key
                )
            )
            if result is None:
                result = UserMemory(user_id=user_id, key=key, value=value, memory_type=memory_type)
                session.add(result)
            else:
                result.value = value
                result.memory_type = memory_type
            await session.commit()
            await session.refresh(result)
            return result

    async def retrieve(self, user_id: str, query: str | None = None) -> list[UserMemory]:
        async with self.session_factory() as session:
            statement = (
                select(UserMemory)
                .where(UserMemory.user_id == user_id)
                .order_by(UserMemory.updated_at.desc())
                .limit(self.settings.memory_retrieval_limit)
            )
            memories = list((await session.scalars(statement)).all())
        if not query:
            return memories
        terms = {term.lower() for term in query.split() if term.strip()}
        aliases = {
            "department": {"部门", "报销", "审批", "制度", "工作"},
        }
        return [
            item
            for item in memories
            if not terms
            or any(
                term in item.key.lower() or term in item.value.lower() for term in terms
            )
            or any(alias in query for alias in aliases.get(item.key, set()))
        ]

    async def delete(self, user_id: str, key: str) -> bool:
        async with self.session_factory() as session:
            result = await session.execute(
                delete(UserMemory).where(
                    UserMemory.user_id == user_id, UserMemory.key == key
                )
            )
            await session.commit()
            return result.rowcount > 0
