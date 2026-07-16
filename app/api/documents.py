import asyncio
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import aiofiles
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.document_parser import SUPPORTED_EXTENSIONS
from app.dependencies import get_document_service, verify_api_key
from app.models.database import get_db
from app.models.document import Document
from app.schemas import DocumentOut
from app.services.document_service import DocumentService

router = APIRouter(
    prefix="/api/documents", tags=["documents"], dependencies=[Depends(verify_api_key)]
)
settings = get_settings()
DatabaseSession = Annotated[AsyncSession, Depends(get_db)]
DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]


@router.post("/upload", response_model=DocumentOut, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    file: Annotated[UploadFile, File()],
    background_tasks: BackgroundTasks,
    db: DatabaseSession,
    service: DocumentServiceDep,
) -> Document:
    filename = file.filename or "unnamed"
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported file type: {suffix or 'none'}",
        )

    document_id = str(uuid4())
    stored_name = f"{document_id}{suffix}"
    destination = settings.upload_dir / stored_name
    total_size = 0

    try:
        async with aiofiles.open(destination, "wb") as output:
            while chunk := await file.read(1024 * 1024):
                total_size += len(chunk)
                if total_size > settings.max_upload_size_bytes:
                    raise HTTPException(status_code=413, detail="file is too large")
                await output.write(chunk)
    except Exception:
        if destination.exists():
            await asyncio.to_thread(destination.unlink)
        raise

    document = Document(
        id=document_id,
        filename=filename,
        stored_name=stored_name,
        file_type=suffix,
        file_size=total_size,
        status="processing",
    )
    db.add(document)
    try:
        await db.commit()
        await db.refresh(document)
    except Exception:
        await asyncio.to_thread(destination.unlink, missing_ok=True)
        raise

    background_tasks.add_task(service.process, document.id, destination, filename)
    return document


@router.get("", response_model=list[DocumentOut])
async def list_documents(db: DatabaseSession) -> list[Document]:
    statement = select(Document).order_by(Document.created_at.desc())
    return list((await db.scalars(statement)).all())


@router.get("/{document_id}", response_model=DocumentOut)
async def get_document(document_id: str, db: DatabaseSession) -> Document:
    document = await db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="document not found")
    return document


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: str,
    db: DatabaseSession,
    service: DocumentServiceDep,
) -> None:
    document = await db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="document not found")
    await service.delete(document, settings.upload_dir / document.stored_name)
