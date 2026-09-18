from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.dependencies import get_memory_service, verify_api_key
from app.schemas import MemoryOut, MemoryWrite
from app.services.memory_service import MemoryService

router = APIRouter(prefix="/api/memory", tags=["memory"], dependencies=[Depends(verify_api_key)])
MemoryServiceDep = Annotated[MemoryService, Depends(get_memory_service)]


@router.post("", response_model=MemoryOut)
async def write_memory(body: MemoryWrite, service: MemoryServiceDep) -> MemoryOut:
    return await service.upsert(body.user_id, body.key, body.value, body.memory_type)


@router.get("", response_model=list[MemoryOut])
async def list_memory(
    user_id: str, service: MemoryServiceDep, query: str | None = Query(default=None)
) -> list[MemoryOut]:
    return await service.retrieve(user_id, query)


@router.delete("/{user_id}/{key}", status_code=204)
async def delete_memory(user_id: str, key: str, service: MemoryServiceDep) -> None:
    await service.delete(user_id, key)
