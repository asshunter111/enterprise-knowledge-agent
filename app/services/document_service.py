import asyncio
import logging
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.document_parser import chunk_text, parse_document
from app.core.embedding import EmbeddingService
from app.core.vector_store import VectorStore
from app.models.document import Document

logger = logging.getLogger(__name__)


class DocumentService:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
    ) -> None:
        self.session_factory = session_factory
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    async def process(self, document_id: str, path: Path, filename: str) -> None:
        try:
            text = await asyncio.to_thread(parse_document, path)
            if not text.strip():
                raise ValueError("document contains no readable text")
            chunks = await asyncio.to_thread(chunk_text, text)
            embeddings = await self.embedding_service.embed_documents(
                [chunk["content"] for chunk in chunks]
            )
            count = await self.vector_store.upsert(chunks, embeddings, document_id, filename)
            await self._set_status(document_id, "ready", chunk_count=count)
        except Exception as exc:
            try:
                await self.vector_store.delete_document(document_id)
            except Exception:
                logger.exception(
                    "failed to clean partial vectors", extra={"document_id": document_id}
                )
            await self._set_status(document_id, "error", error_message=str(exc))

    async def delete(self, document: Document, path: Path) -> None:
        await self.vector_store.delete_document(document.id)
        if path.exists():
            await asyncio.to_thread(path.unlink)
        async with self.session_factory() as session:
            stored = await session.get(Document, document.id)
            if stored is not None:
                await session.delete(stored)
                await session.commit()

    async def _set_status(
        self,
        document_id: str,
        status: str,
        *,
        chunk_count: int = 0,
        error_message: str | None = None,
    ) -> None:
        async with self.session_factory() as session:
            document = await session.get(Document, document_id)
            if document is None:
                return
            document.status = status
            document.chunk_count = chunk_count
            document.error_message = error_message
            await session.commit()
