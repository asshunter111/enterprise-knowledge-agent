import pytest

from app.config import Settings
from app.models.database import AsyncSessionLocal, init_db
from app.services.memory_service import MemoryService


@pytest.mark.asyncio
async def test_memory_upsert_retrieve_and_delete():
    await init_db()
    service = MemoryService(AsyncSessionLocal, Settings(memory_retrieval_limit=5))
    await service.upsert("user-1", "default_city", "兰州")
    await service.upsert("user-1", "default_city", "西安")
    memories = await service.retrieve("user-1", "default_city")
    assert len(memories) == 1
    assert memories[0].value == "西安"
    assert await service.delete("user-1", "default_city")
    assert await service.retrieve("user-1") == []
