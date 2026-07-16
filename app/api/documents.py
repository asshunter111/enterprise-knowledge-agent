"""文档管理 API — 上传 / 列表 / 删除"""
import os
import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import UPLOAD_DIR, ALLOWED_EXTENSIONS, MAX_UPLOAD_SIZE
from app.models.database import get_db
from app.models.document import Document
from app.schemas import DocumentOut
from app.core.document_parser import DocumentParser
from app.core.vector_store import add_chunks, delete_by_doc_id

router = APIRouter(prefix="/documents", tags=["文档管理"])


@router.post("/upload", response_model=DocumentOut)
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """上传文档：解析 → 分块 → 向量化 → 入库"""
    # 校验扩展名
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"不支持的文件格式: {ext}，支持: {ALLOWED_EXTENSIONS}")

    # 保存文件
    doc_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{doc_id}{ext}"
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(400, f"文件超过最大限制 {MAX_UPLOAD_SIZE // 1024 // 1024}MB")
    save_path.write_bytes(content)

    # 数据库记录
    doc = Document(
        id=doc_id, filename=file.filename, file_type=ext,
        file_size=len(content), status="processing",
    )
    db.add(doc)
    await db.commit()

    # 解析 + 分块 + 向量化
    try:
        text = DocumentParser.parse(save_path)
        chunks = DocumentParser.chunk(text)
        count = await add_chunks(chunks, doc_id, file.filename)
        doc.chunk_count = count
        doc.status = "ready"
    except Exception as e:
        doc.status = "error"
        doc.error_message = str(e)
    await db.commit()
    await db.refresh(doc)

    return DocumentOut(
        id=doc.id, filename=doc.filename, file_type=doc.file_type,
        file_size=doc.file_size, chunk_count=doc.chunk_count,
        status=doc.status, created_at=str(doc.created_at),
    )


@router.get("/", response_model=list[DocumentOut])
async def list_documents(db: AsyncSession = Depends(get_db)):
    """获取文档列表"""
    result = await db.execute(
        select(Document).order_by(Document.created_at.desc())
    )
    docs = result.scalars().all()
    return [
        DocumentOut(
            id=d.id, filename=d.filename, file_type=d.file_type,
            file_size=d.file_size, chunk_count=d.chunk_count,
            status=d.status, created_at=str(d.created_at),
        )
        for d in docs
    ]


@router.delete("/{doc_id}")
async def delete_document(doc_id: str, db: AsyncSession = Depends(get_db)):
    """删除文档及其向量数据"""
    result = await db.execute(select(Document).where(Document.id == doc_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(404, "文档不存在")

    # 删向量
    await delete_by_doc_id(doc_id)
    # 删文件
    file_path = UPLOAD_DIR / f"{doc_id}{doc.file_type}"
    if file_path.exists():
        os.remove(file_path)
    # 删数据库记录
    await db.delete(doc)
    await db.commit()
    return {"message": "删除成功"}
