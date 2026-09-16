from collections.abc import AsyncIterator
from importlib import import_module

from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings


class Base(DeclarativeBase):
    pass


settings = get_settings()
engine_options: dict = {"pool_pre_ping": True}
if settings.database_url.startswith("sqlite"):
    engine_options["connect_args"] = {"check_same_thread": False}
else:
    engine_options.update({"pool_size": 10, "max_overflow": 20})

engine = create_async_engine(settings.database_url, **engine_options)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_db() -> AsyncIterator[AsyncSession]:
    async with AsyncSessionLocal() as session:
        yield session


async def init_db() -> None:
    settings.prepare_directories()
    import_module("app.models")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
        await connection.run_sync(_add_missing_columns)


def _add_missing_columns(connection) -> None:
    """create_all 只建新表，不会给已存在的表补列，老库需要单独升级。"""

    if "chat_sessions" not in inspect(connection).get_table_names():
        return
    existing = {column["name"] for column in inspect(connection).get_columns("chat_sessions")}
    if "active_context" not in existing:
        connection.execute(text("ALTER TABLE chat_sessions ADD COLUMN active_context JSON"))
